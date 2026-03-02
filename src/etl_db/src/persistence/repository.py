from typing import List
import logging
from psycopg2.extras import execute_values
from src.core.types import Candle

logger = logging.getLogger(__name__)

class MarketDataRepository:
    def __init__(self, conn):
        self.conn = conn

    def save_candles(self, candles: List[Candle], table_name: str = "candles_1h"):
        """
        High-performance bulk upsert using execute_values.
        This reduces network overhead by sending one massive SQL packet.
        """
        if not candles:
            return

        # Prepare the Data
        # We convert Pydantic models to a list of tuples
        data = []
        for c in candles:
            data.append((c.symbol, c.timestamp, c.open, c.high, c.low, c.close, c.volume))

        # Define the Query
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
                # Execute Fast Batch Insert
                execute_values(cursor, query, data)
            
            # Commit and Log
            self.conn.commit()
            logger.info(f"Saved {len(candles)} candles to {table_name}")
            
        except Exception as e:
            # There was a problem
            self.conn.rollback()
            logger.error(f"Failed to save candles: {e}")
            raise e