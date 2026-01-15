# Optimized Dockerfile for Railway - Python AI Backend
# Uses slim base image and caching to reduce size

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies with cache optimization
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p chroma_db data

# Expose port (Railway sets PORT env var)
ENV PORT=8000
EXPOSE 8000

# No CMD - Railway provides start command via dashboard:
# gunicorn server:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 180
