import logging
import os
import time

import psycopg2
from psycopg2.extensions import connection

logger = logging.getLogger("ETL.DB_CONNECTION")


def get_db_connection(max_retries: int = 5, dsn: str | None = None) -> connection:
    """
    Establishes a connection to the Postgres DB with retry logic.

    If `dsn` is not provided, it falls back to the DATABASE_URL environment
    variable for backwards compatibility.
    """
    if dsn is None:
        dsn = os.getenv("DATABASE_URL")

    if not dsn:
        # Fail fast if the DSN is missing
        raise ValueError("Database DSN is not set (DATABASE_URL or explicit dsn).")

    # For loop to try and connect
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(dsn)
            return conn
        except psycopg2.OperationalError as e:
            logger.warning(
                f"DB connection failed (attempt {attempt + 1}/{max_retries}): {e}"
            )
            time.sleep(2)

    raise Exception("Could not connect to the database after multiple retries.")
