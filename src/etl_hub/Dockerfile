# src/etl_hub/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory
COPY src/ /app/src/

# Tell Python where the root module path is
ENV PYTHONPATH=/app/src

# Run the ETL Hub module
CMD ["python", "-m", "etl_hub.main"]