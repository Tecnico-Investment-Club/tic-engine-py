import os
import time
import psycopg2
from psycopg2.extensions import connection
import logging

logger = logging.getLogger(__name__)

def get_db_connection(max_retries: int = 5) -> connection:
    """
    Establishes a connection to the Postgres DB with retry logic.
    Fails immediately if DATABASE_URL is not defined.
    """

    # Get the DSN from environment variables
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        # Fail fast if the environment variable is missing
        raise ValueError("DATABASE_URL environment variable is not set.")

    # For loop to try and connect
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(dsn)
            return conn
        except psycopg2.OperationalError as e:
            logger.warning(f"DB connection failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2)

    raise Exception("Could not connect to the database after multiple retries.")
