FROM python:3.9-slim

# Install lib dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Gunicorn will be invoked via Procfile on Render
CMD ["gunicorn", "app:app", "--workers", "1", "--bind", "0.0.0.0:5000"]
