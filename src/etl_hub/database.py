import logging
import os
import time
from typing import List, Optional
from datetime import datetime

import psycopg2
from psycopg2.extensions import connection
from psycopg2.extras import execute_values

from core.datatypes import Candle
from etl_hub.interfaces.IDatabase import IDatabase

logger = logging.getLogger("ETL.DATABASE")

class PostgresDatabase(IDatabase):
    """
    Handles Postgres connection pooling and data persistence.
    """
    def __init__(self, dsn: Optional[str] = None, max_retries: int = 5):
        self.dsn = dsn or os.getenv("DATABASE_URL")
        self.max_retries = max_retries
        self.conn: Optional[connection] = None
        
        if not self.dsn:
            raise ValueError("Database DSN is not set (DATABASE_URL or explicit dsn).")

    def connect(self) -> None:
        if self.conn and not self.conn.closed:
            return

        for attempt in range(self.max_retries):
            try:
                self.conn = psycopg2.connect(self.dsn)
                logger.info("Successfully connected to the internal database.")
                return
            except psycopg2.OperationalError as e:
                logger.warning(f"DB connection failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(2)

        raise Exception("Could not connect to the database after multiple retries.")

    def save_candles(self, candles: List[Candle], table_name: str = "candles_1h") -> None:
        if not candles:
            return

        self._ensure_connection()

        data = [(c.symbol, c.timestamp, c.open, c.high, c.low, c.close, c.volume) for c in candles]

        query = f"""
            INSERT INTO {table_name} (symbol, timestamp, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (symbol, timestamp) 
            DO UPDATE SET 
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume;
        """

        try:
            with self.conn.cursor() as cursor:
                execute_values(cursor, query, data)
            self.conn.commit()
            logger.info(f"Saved {len(candles)} candles to {table_name} from asset {candles[0].symbol}.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to save candles: {e}")
            raise e

    def get_latest_timestamp(self, symbol: str, table_name: str = "candles_1h") -> Optional[datetime]:
        self._ensure_connection()
        query = f"SELECT MAX(timestamp) FROM {table_name} WHERE symbol = %s;"
        with self.conn.cursor() as cursor:
            cursor.execute(query, (symbol,))
            result = cursor.fetchone()
            if result and result[0]:
                return result[0]
        return None

    def disconnect(self) -> None:
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Database connection closed.")
            
    def _ensure_connection(self):
        if not self.conn or self.conn.closed:
            self.connect()


class DataJanitor:
    """
    Dedicated class to handle database pruning and cleanup.
    Keeps the database footprint lean.
    """
    def __init__(self, db: PostgresDatabase):
        self.db = db

    def run_janitor(self, days_to_keep: int = 500) -> int:
        self.db._ensure_connection()
        
        query = f"""
            WITH deleted_1h AS (
                DELETE FROM candles_1h WHERE timestamp < NOW() - INTERVAL '{days_to_keep} days' RETURNING 1
            ),
            deleted_1d AS (
                DELETE FROM candles_1d WHERE timestamp < NOW() - INTERVAL '{days_to_keep} days' RETURNING 1
            )
            SELECT (SELECT COUNT(*) FROM deleted_1h) + (SELECT COUNT(*) FROM deleted_1d);
        """
        
        try:
            with self.db.conn.cursor() as cursor:
                cursor.execute(query)
                total_deleted = cursor.fetchone()[0]
                
            self.db.conn.commit()
            if total_deleted > 0:
                logger.info(f"Janitor pruned {total_deleted} old records.")
            return total_deleted
            
        except Exception as e:
            self.db.conn.rollback()
            logger.error(f"Janitor failed to prune data: {e}")
            return 0