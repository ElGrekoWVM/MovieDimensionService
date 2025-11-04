FROM python:3.11-slim

# Install system deps for OpenCV
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      libglib2.0-0 \
      libsm6 \
      libxrender1 \
      libxext6 \
      ffmpeg \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV VIDEO_ROOT=/video

EXPOSE 8080

# Use gunicorn for production
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "app:app"]
