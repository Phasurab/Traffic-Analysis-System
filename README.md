# AI Traffic Analysis System of Bangkok

> **DevOps Highlights**: This application is fully containerized using a lightweight Python-slim Docker image. Orchestration is managed via Docker Compose, utilizing bounded volume mounts to ensure SQLite database persistence and CSV state accumulation across container restarts.

This project provides a comprehensive local web application for analyzing traffic videos, extracting vehicle counts based on user-defined polygon areas and trajectory lines, and generating visual and statistical outputs. It is built using FastAPI for the backend and HTML/JS/CSS for the frontend, featuring a responsive, glassmorphism UI with an Emerald Green aesthetic tailored for NECTEC and the Bangkok Metropolitan Administration.

### Demonstration
[![Watch the demonstration video](https://img.youtube.com/vi/0M-j-yNAENg/maxresdefault.jpg)](https://youtu.be/0M-j-yNAENg)

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/your-username/traffic-analysis-system.git
cd traffic-analysis-system

# Spin up the containerized application
docker compose up --build
```
*Note: Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is running. Once built, open `http://localhost:8000` in your browser!*

## Key Features
- **Interactive Drawing Tool**: Integrated Leaflet.js canvas allows users to draw counting areas (polygons) and trajectory tracking lines (polylines) directly over the first frame of the uploaded video.
- **Multi-Model Inference Pipeline**: Choose between Ultralytics YOLOv8 and DINOv3 LTDETR models for object detection and tracking using PyTorch, Supervision, and ByteTrack.
- **Advanced Tracking Logic**: Supports traditional area-based counting and vector-math-based polyline intersection (e.g. tracking vehicles moving "from Area 1 to Area 2").
- **Live Processing UI**: Dynamic progress bar polls the FastAPI backend to provide real-time inference status updates.
- **Comprehensive Data Export**: The system manages dual-layer storage. It logs all runtime metadata, user-drawn geometries, and frame-by-frame JSON tracking data dynamically into an internal SQLite database (`traffic.db`), while simultaneously compiling and appending the tracking data into an `accumulated_traffic_data.csv` spreadsheet for seamless data analytics across multiple video uploads.

## Requirements
Ensure you have the following installed in your local Python 3.9+ environment:
- `fastapi`, `uvicorn`, `python-multipart`
- `opencv-python`
- `torch`, `torchvision`
- `supervision`, `ultralytics`
- `lightly` (Required for DINO model)
- `tqdm`

## Running the Web App (Docker - Recommended)
The easiest way to run this application on any computer is via Docker Compose.
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. From the root of this project folder, simply run:
   ```bash
   docker compose up --build
   ```
3. Open a browser and navigate to `http://localhost:8000`.

*Note: The `outputs` and `uploads` folder are mapped directly to your host machine, ensuring your SQLite Database (`traffic.db`) and `accumulated_traffic_data.csv` history aren't wiped when the container closes!*

## Running the Web App (Local Python)
1. Navigate to the `backend` folder:
   ```bash
   cd vehicle_counting_app/backend
   ```
2. Start the FastAPI development server:
   ```bash
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
3. Open a browser and navigate to `http://localhost:8000`.

## Architecture Notes
- Custom Leaflet coordinate mappers in `frontend/script.js` automatically flip the Y-axis to bridge the gap between Leaflet's traditional bottom-up mapping style and OpenCV/Supervision's top-down video matrices.
- SQLite Database automatically instantiates inside `backend/outputs/traffic.db` to maintain a relational history of tasks, drawing zones, and tracking inferences.
- CSV Exports dynamically accumulate across multiple video uploads to `backend/outputs/accumulated_traffic_data.csv` keeping an easily accessible master record.
