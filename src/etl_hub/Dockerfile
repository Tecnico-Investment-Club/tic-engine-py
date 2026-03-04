FROM python:3.12-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Set the working directory to the project root
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ENTIRE src folder into /app/src
# This ensures etl_db can see core, and engine can see core.
COPY src/ /app/src/

# IMPORTANT: Set PYTHONPATH to /app so 'import src.core...' works
ENV PYTHONPATH=/app

# Start the module
# Use the -m flag without the .py extension
# Adjust this based on your actual file path (e.g., src.etl_db.src.main)
CMD ["python", "-m", "src.etl_db.src.main"]