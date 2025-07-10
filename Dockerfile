# Multi-stage build for Railway deployment optimization
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies to user directory for easy copying
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Create directories
RUN mkdir -p uploads static/uploads

# Copy application code
COPY web_app/ ./web_app/
COPY runs/train/weights/ ./runs/train/weights/
COPY dataset.yaml .
COPY run_web_app.py .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE $PORT

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--timeout", "120", "web_app.app:app"] 