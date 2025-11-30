FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install ffmpeg and other small deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       gifsicle \
       imagemagick \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .

ENV PYTHONUNBUFFERED=1
CMD ["python", "main.py"]
