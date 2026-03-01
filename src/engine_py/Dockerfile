# Use a lightweight Python image
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr (so logs show up immediately)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies strictly required for the Engine
RUN pip install --no-cache-dir \
    pandas \
    schedule \
    alpaca-py \
    psycopg2-binary \
    pyyaml \
    python-dotenv \
    requests

# Copy the entire source tree (Core + Engine)
COPY src/ /app/src/

# Add /app to PYTHONPATH so "from src.core..." works
ENV PYTHONPATH=/app

# Run the Engine Entrypoint
CMD ["python", "src/engine_py/main.py"]