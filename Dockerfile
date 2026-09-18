FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Protocol packs are loaded relative to the working directory (retrieval.py), so they ship with the app.
COPY src ./src
COPY protocols ./protocols
COPY data/few_shot_library.yaml ./data/few_shot_library.yaml

# Cloud Run injects PORT (8080 by default). --proxy-headers so URLs behind Google's front end are correct.
CMD ["sh", "-c", "exec uvicorn src.voice_interface.media_stream_server:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*'"]
