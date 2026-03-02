# src/etl_db/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything needed for the Hub
COPY src/core /app/src/core
COPY src/etl_db /app/src/etl_db

# Set PYTHONPATH so it can find the 'src' package root
ENV PYTHONPATH=/app

# Updated CMD to match your actual folder structure
CMD ["python", "-m", "src.etl_db.src.main"]