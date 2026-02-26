import os
import json
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel

# Import our tracking pipeline function
from pipeline import run_tracking_pipeline

app = FastAPI(title="Vehicle Counting App")

# Define directories
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BACKEND_DIR / "uploads"
OUTPUT_DIR = BACKEND_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global store for tracking progress
progress_store = {}

def process_video_task(task_id: str, video_path: str, spec_data: dict, out_dir: str, model_type: str):
    import database
    try:
        out_video, out_json, out_csv = run_tracking_pipeline(task_id, video_path, spec_data, out_dir, model_type, progress_store[task_id])
        progress_store[task_id]["output_video"] = Path(out_video).name
        progress_store[task_id]["output_json"] = Path(out_json).name
        # Use the physical file directly generated from pipeline.py again
        progress_store[task_id]["output_csv"] = Path(out_csv).name
        progress_store[task_id]["status"] = "complete"
        database.update_task_status(task_id, "complete")
    except Exception as e:
        progress_store[task_id]["status"] = "error"
        progress_store[task_id]["error"] = str(e)
        database.update_task_status(task_id, "error", str(e))


@app.post("/api/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    json_spec: UploadFile = File(...),
    model_type: str = Form("yolo")
):
    try:
        task_id = str(uuid.uuid4())
        progress_store[task_id] = {"status": "processing", "progress": 0, "output_video": None, "output_json": None, "output_csv": None, "error": None}

        video_path = UPLOAD_DIR / video.filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        json_content = await json_spec.read()
        json_path = UPLOAD_DIR / "input.json"
        with open(json_path, "wb") as f:
            f.write(json_content)

        try:
            spec_data = json.loads(json_content.decode("utf-8"))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file provided.")

        import database
        # Create Task Record
        database.create_task(task_id, video.filename, model_type)
        # Save Defined Zones (Polygons/Lines)
        database.save_task_zones(task_id, spec_data)

        print(f"Queueing task {task_id}: Processing video {video.filename} with {model_type}")
        background_tasks.add_task(process_video_task, task_id, str(video_path), spec_data, str(OUTPUT_DIR), model_type)

        return {"task_id": task_id, "message": "Inference task queued successfully"}

    except Exception as e:
        print(f"Error during upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/progress/{task_id}")
async def get_progress(task_id: str):
    if task_id not in progress_store:
        raise HTTPException(status_code=404, detail="Task not found")
    return progress_store[task_id]


@app.get("/api/download/video/{filename}")
async def download_video(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found.")
    return FileResponse(file_path, media_type="video/mp4", filename=filename, headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/api/download/json/{filename}")
async def download_json(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output JSON not found.")
    return FileResponse(file_path, media_type="application/json", filename=filename, headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/api/download/csv/{filename}")
async def download_csv(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output CSV not found.")
    return FileResponse(file_path, media_type="text/csv", filename=filename, headers={"Content-Disposition": f"attachment; filename={filename}"})


# Mount the frontend directory so we can access index.html
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
