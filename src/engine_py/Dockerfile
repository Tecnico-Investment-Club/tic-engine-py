# src/engine_py/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared core and the engine app
COPY src/core /app/src/core
COPY src/engine_py /app/src/engine_py

# Ensure Python treats /app as the root module path
ENV PYTHONPATH=/app

# Run the engine
CMD ["python", "-m", "src.engine_py.src.main"]