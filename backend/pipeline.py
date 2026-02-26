import json
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from ultralytics import YOLO
import supervision as sv

# Use lightweight YOLOv8 model for local processing
MODEL_PATH = "yolov8n.pt"

ALLOWED_CLASSES = np.array([2, 3, 5, 7], dtype=int)
# Vehicle types mapped for the specific output format requirement
OUTPUT_VEHICLE_MAP = {2: "sedan", 3: "motorcycle", 5: "bus", 7: "pickup"}  

def cls_name(c: int) -> str:
    m = {2: "sedan", 3: "motorcycle", 5: "bus", 7: "pickup"}
    return m.get(int(c), f"cls_{int(c)}")

def make_empty_detections():
    return sv.Detections(
        xyxy=np.zeros((0, 4), dtype=float),
        confidence=np.zeros((0,), dtype=float),
        class_id=np.zeros((0,), dtype=int),
    )

def point_in_poly(poly_xy: np.ndarray, x: int, y: int) -> bool:
    return cv2.pointPolygonTest(poly_xy, (float(x), float(y)), False) >= 0

def line_intersect(A, B, C, D):
    # Returns True if line segment AB (track) intersects CD (user drawn line segment)
    def ccw(p1, p2, p3):
        return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def check_polyline_intersection(A, B, polyline_pts):
    # Check if segment AB intersects any segment of a polyline
    for i in range(len(polyline_pts) - 1):
        if line_intersect(A, B, polyline_pts[i], polyline_pts[i+1]):
            return True
    return False

def draw_geometries(frame, polygons, lines, colors):
    for name, poly in polygons.items():
        cv2.polylines(frame, [poly], isClosed=True, color=colors[name], thickness=2)
        px, py = int(poly[0, 0]), int(poly[0, 1])
        cv2.putText(frame, str(name), (px, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[name], 2, cv2.LINE_AA)
                    
    for name, line_pts in lines.items():
        cv2.polylines(frame, [line_pts], isClosed=False, color=colors[name], thickness=2)
        px, py = int(line_pts[0, 0]), int(line_pts[0, 1])
        cv2.putText(frame, str(name), (px, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[name], 2, cv2.LINE_AA)

def draw_summary_top_left(frame, counts_by_area, output_mapping, max_lines=25):
    x0, y0 = 20, 35
    lh = 24

    lines = []
    lines.append("COUNT BY AREA")
    for direction in output_mapping:
        lines.append(f"[{direction}]")
        lane_id = output_mapping[direction]
        if lane_id in counts_by_area:
            area_counts = counts_by_area[lane_id]
            for k in sorted(area_counts.keys()):
                lines.append(f"  {k}: {area_counts[k]}")
        lines.append("")

    if len(lines) > max_lines:
        lines = lines[:max_lines-1] + ["..."]

    overlay = frame.copy()
    box_w = 420
    box_h = (len(lines) + 1) * lh + 10
    cv2.rectangle(overlay, (x0 - 10, y0 - 25), (x0 - 10 + box_w, int(y0 - 25 + box_h)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = y0
    for text in lines:
        if text == "":
            y += lh // 2
            continue
        is_header = (text.startswith("COUNT") or text.startswith("["))
        color = (255, 255, 255) if is_header else (0, 255, 0)
        cv2.putText(frame, text, (x0, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += lh

def run_tracking_pipeline(task_id: str, video_path: str, spec_data: dict, out_dir: str, model_type: str = "yolo", progress_dict: dict = None):
    video_file = Path(video_path)
    output_video_path = Path(out_dir) / (video_file.stem + "_processed.mp4")
    output_json_path = Path(out_dir) / (video_file.stem + "_output.json")

    # 1. Parse JSON to get polygons and lane mapping
    polygons = {}
    lines = {}
    geom_colors = {}
    features = spec_data.get("zones", {}).get("features", [])
    color_palette = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    for idx, feature in enumerate(features):
        name = feature.get("properties", {}).get("name", str(idx))
        geom_type = feature.get("geometry", {}).get("type", "Polygon")
        coords = feature.get("geometry", {}).get("coordinates", [])
        
        if not coords:
            continue
            
        if geom_type == "Polygon" and len(coords) > 0:
            poly_np = np.array(coords[0], dtype=np.int32)
            polygons[name] = poly_np
            geom_colors[name] = color_palette[idx % len(color_palette)]
        elif geom_type == "LineString" and len(coords) >= 2:
            line_np = np.array(coords, dtype=np.int32)
            lines[name] = line_np
            geom_colors[name] = color_palette[idx % len(color_palette)]

    lane_set = spec_data.get("lane_set", {})
    output_mapping = {}
    for direction, lanes in lane_set.items():
        if isinstance(lanes, list) and len(lanes) > 0:
            output_mapping[direction] = lanes[0]

    # 2. Init Model
    conf_thres = 0.2
    model = None
    if model_type == "dino":
        try:
            # Look for the DINO model file in the current backend directory or the parent directory
            local_dino_name = "dinov3_convnext_tiny_ltdetr_coco_251113_3a90352e.pt"
            possible_paths = [Path(local_dino_name), Path("..") / local_dino_name]
            
            DINO_PATH = None
            for p in possible_paths:
                if p.exists():
                    DINO_PATH = str(p.resolve())
                    break
            
            if not DINO_PATH:
                raise FileNotFoundError(f"Could not find DINO model '{local_dino_name}' in the project directory.")
                
            print(f"[INFO] Loading DINO model: {DINO_PATH}")
            import lightly_train
            model = lightly_train.load_model(DINO_PATH)
        except ImportError:
            raise RuntimeError("The 'lightly_train' module is required to run the DINO model, but it is not installed locally. Please run YOLOv8 or install the lightly_train module.")
        except Exception as e:
            raise RuntimeError(f"Failed to load DINO model: {e}")
    else:
        print(f"[INFO] Loading YOLO model: {MODEL_PATH}")
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)

    # 3. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ENHANCEMENT: Use avc1 codec for web browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # 4. Init ByteTrack
    tracker = sv.ByteTrack(
        track_activation_threshold=0.5,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=int(fps),
    )

    counts_by_area = {name: defaultdict(int) for name in list(polygons.keys()) + list(lines.keys())}

    # Dictionary to track transitional paths (Line A -> Line B)
    paths_count = defaultdict(lambda: defaultdict(int))
    track_state = {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Processing {total_frames} frames...")

    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if progress_dict is not None and total_frames > 0:
                progress_dict["progress"] = int((frame_idx / total_frames) * 100)

            draw_geometries(frame_bgr, polygons, lines, geom_colors)

            if model_type == "dino":
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)
                
                pred = model.predict(img_tensor, threshold=conf_thres)
                labels = pred["labels"].cpu().numpy()
                boxes  = pred["bboxes"].cpu().numpy()
                scores = pred["scores"].cpu().numpy()
            else:
                results = model(frame_bgr, verbose=False, conf=conf_thres)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                labels = results.boxes.cls.cpu().numpy().astype(int)

            keep = (scores >= conf_thres) & np.isin(labels, ALLOWED_CLASSES)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if len(boxes) == 0:
                _ = tracker.update_with_detections(make_empty_detections())
                draw_summary_top_left(frame_bgr, counts_by_area, output_mapping)
                writer.write(frame_bgr)
                continue

            detections = sv.Detections(xyxy=boxes, confidence=scores, class_id=labels)
            tracked = tracker.update_with_detections(detections)

            tids = getattr(tracked, "tracker_id", None)
            if tids is None:
                tids = [None] * len(tracked.xyxy)
            else:
                tids = tids.tolist() if hasattr(tids, "tolist") else list(tids)

            for (xmin, ymin, xmax, ymax), sc, c, tid in zip(tracked.xyxy, tracked.confidence, tracked.class_id, tids):
                if tid is None:
                    continue

                track_id = int(tid)
                c_int = int(c)
                cname = cls_name(c_int)

                x1, y1, x2, y2 = map(int, [xmin, ymin, xmax, ymax])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                st = track_state.get(track_id)
                if st is None:
                    st = {"counted_areas": set(), "path": [], "prev_pt": (cx, cy)}
                    track_state[track_id] = st

                prev_pt = st["prev_pt"]
                matched_area = None
                
                # Check Polygons (Areas)
                for name, poly in polygons.items():
                    if point_in_poly(poly, cx, cy):
                        matched_area = name
                        break
                        
                # Check Lines (Trajectories)
                if matched_area is None:
                    for name, line_pts in lines.items():
                        if check_polyline_intersection(prev_pt, (cx, cy), line_pts):
                            matched_area = name
                            break

                if matched_area is not None and matched_area not in st["counted_areas"]:
                    counts_by_area[matched_area][cname] += 1
                    st["counted_areas"].add(matched_area)
                    st["path"].append(matched_area)
                    
                    # Store Trajectory (e.g. "From 1 to 2")
                    if len(st["path"]) >= 2:
                        path_name = f"From {st['path'][-2]} to {st['path'][-1]}"
                        paths_count[path_name][cname] += 1
                        
                # Update prev_pt for next frame
                st["prev_pt"] = (cx, cy)

                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame_bgr, (cx, cy), 4, (0, 0, 255), -1)

                area_txt = matched_area if matched_area is not None else "-"
                cv2.putText(
                    frame_bgr, f"ID:{track_id} {cname} {float(sc):.2f} area={area_txt}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA
                )

            draw_summary_top_left(frame_bgr, counts_by_area, output_mapping)
            writer.write(frame_bgr)

    cap.release()
    writer.release()
    print("[INFO] Done processing video.")

    final_output = {
        "areas": {},
        "paths": {}
    }
    
    for direction, lane_id in output_mapping.items():
        if lane_id in counts_by_area:
            direction_list = []
            for v_class, cnt in sorted(counts_by_area[lane_id].items()):
                direction_list.append({v_class: cnt})
            final_output["areas"][direction] = direction_list
        else:
            final_output["areas"][direction] = []
            
    # Include path transitions in final JSON format
    for path_name, path_counts in paths_count.items():
        transition_list = []
        for v_class, cnt in sorted(path_counts.items()):
            transition_list.append({v_class: cnt})
        final_output["paths"][path_name] = transition_list

    with open(output_json_path, 'w') as f:
        json.dump(final_output, f, indent=2)

    # --- Generate Accumulated SQLite DB Entries ---
    import database
    database.save_tracking_results(task_id, final_output)

    # --- Generate Accumulated CSV Output (On Disk) ---
    import csv
    from datetime import datetime
    
    # Always write to a single master CSV file in the output directory
    output_csv_path = Path(out_dir) / "accumulated_traffic_data.csv"
    
    # Get current timestamp for this run
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    csv_rows = []
    
    # Flatten Areas
    for area_name, counts_list in final_output.get("areas", {}).items():
        for count_dict in counts_list:
            for v_class, cnt in count_dict.items():
                csv_rows.append([current_time, video_file.name, "Area", area_name, v_class, cnt])
                
    # Flatten Paths
    for path_name, counts_list in final_output.get("paths", {}).items():
        for count_dict in counts_list:
            for v_class, cnt in count_dict.items():
                csv_rows.append([current_time, video_file.name, "Trajectory", path_name, v_class, cnt])
                
    # Check if file exists to determine if we need a header
    file_exists = output_csv_path.exists()
    
    with open(output_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Source Video", "Geometry Type", "Name", "Vehicle Class", "Count"])
        writer.writerows(csv_rows)
        
    if progress_dict is not None:
        progress_dict["progress"] = 100

    return output_video_path, output_json_path, output_csv_path

