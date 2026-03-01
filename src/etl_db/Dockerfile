# Use a lightweight Python image
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    psycopg2-binary \
    pydantic \
    pyyaml \
    requests \
    alpaca-py \
    python-dotenv

# Copy the entire source tree (Core + ETL)
COPY src/ /app/src/

# Add /app to PYTHONPATH so "from src.core..." works
ENV PYTHONPATH=/app

# Run the ETL Entrypoint
CMD ["python", "src/etl_db/main.py"]