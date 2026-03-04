from typing import List
import logging
from datetime import datetime
from psycopg2.extras import execute_values
from src.core.types import Candle

logger = logging.getLogger("ETL.REPOSITORY")

class MarketDataRepository:
    def __init__(self, conn):
        self.conn = conn

    def save_candles(self, candles: List[Candle], table_name: str = "candles_1h"):
        """
        Bulk upsert using execute_values.
        """
        if not candles:
            return

        # Prepare the Data
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
            logger.info(f"Saved {len(candles)} candles to {table_name} from asset {candles[0].symbol}.")
            
        except Exception as e:
            # There was a problem
            self.conn.rollback()
            logger.error(f"Failed to save candles: {e}")
            raise e
    
    def prune_old_data(self, retention_days: int):
        query = f"""
            DELETE FROM candles_1h WHERE timestamp < NOW() - INTERVAL '{retention_days} days';
            DELETE FROM candles_1d WHERE timestamp < NOW() - INTERVAL '{retention_days} days';
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Janitor failed to prune data: {e}")
    
    def get_latest_timestamp(self, symbol: str, table_name: str) -> datetime:
            """
            Returns the timestamp of the most recent candle for a given symbol.
            Returns None if no data exists.
            """
            query = f"SELECT MAX(timestamp) FROM {table_name} WHERE symbol = %s;"
            with self.conn.cursor() as cursor:
                cursor.execute(query, (symbol,))
                result = cursor.fetchone()
                if result and result[0]:
                    return result[0]
            return None