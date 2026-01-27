FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Create cache directory
RUN mkdir -p /tmp/model_cache

# Environment variables
ENV TRANSFORMERS_CACHE=/tmp/model_cache
ENV HF_HOME=/tmp/model_cache
ENV PORT=7860

# Expose port
EXPOSE 7860

# Run application
CMD ["python", "app.py"]
