# Optimized Dockerfile for Railway - Python AI Backend
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p chroma_db data

# Railway sets PORT dynamically, default to 8000
ENV PORT=8000
EXPOSE 8000

# MUST use shell form with sh -c to expand $PORT
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port $PORT"]
