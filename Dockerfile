FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY Website/requirements.txt .
RUN pip install -r requirements.txt

# Copy the backend (app.py, models/, lookup spreadsheet)
COPY Website/ .

# Cloud Run supplies $PORT (8080 by default)
CMD exec gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 120
