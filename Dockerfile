# Simulator, training, tests and the read-only API. Live play needs macOS (screen capture,
# clicks and Apple Vision OCR) and runs natively, not in a container: see README.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "pytest", "-q"]
