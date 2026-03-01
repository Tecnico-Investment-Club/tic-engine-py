# Dockerfile for ETL+DB container
# Combines Postgres database and ETL ingestion service in a single container

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql \
    postgresql-contrib \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY .env* ./

# Create directory for Postgres data
RUN mkdir -p /var/lib/postgresql/data && \
    chown -R postgres:postgres /var/lib/postgresql/data

# Copy entrypoint script
COPY src/etl_db/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose Postgres port (optional, for debugging)
EXPOSE 5432

# Run entrypoint
CMD ["/app/entrypoint.sh"]

