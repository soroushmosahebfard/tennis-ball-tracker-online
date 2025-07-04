FROM python:3.9-slim

# install OS deps for OpenCV
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["gunicorn", "app:app", "--workers", "1", "--bind", "0.0.0.0:5000"]
