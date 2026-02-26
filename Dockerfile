FROM python:3.9-slim

# Set environment variables to avoid python buffering (helps with seeing logs in real-time)
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required by OpenCV and general utilities
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY backend/ backend/
COPY frontend/ frontend/

# Ensure output directory exists inside container for SQLite and CSVs
RUN mkdir -p backend/outputs

# Expose the API port
EXPOSE 8000

# Start Uvicorn when the container launches
# Navigate directly to the backend folder first where main.py lives
WORKDIR /app/backend
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
