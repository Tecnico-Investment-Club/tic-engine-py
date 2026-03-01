import time
import logging
import yaml
from typing import List

from src.etl_db.data_source.base import DataSource
from src.etl_db.persistence.repository import MarketDataRepository

logger = logging.getLogger(__name__)

class IngestionJob:
    def __init__(self, source: DataSource, repo: MarketDataRepository, config_path: str = "src/etl_db/config.yaml"):
        self.source = source
        self.repo = repo
        self.config = self._load_config(config_path)

    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def run_loop(self):
        """
        The main infinite loop. 
        Fetches data -> Saves to DB -> Sleeps.
        """
        logger.info("--- Starting ETL Ingestion Loop ---")
        
        while True:
            # Load the internal parameters
            assets = self.config.get("assets", [])
            sleep_time = self.config["job_settings"].get("sleep_seconds", 3600)
            limit = self.config["job_settings"].get("lookback_limit", 100)

            # Log that the new cycle is starting
            logger.info(f"Starting cycle for {len(assets)} assets.")

            for symbol in assets:
                try:
                    # Fetch 1-Hour Candles
                    candles_1h = self.source.fetch_candles(symbol=symbol, timeframe="1h", limit=limit)
                    if candles_1h:
                        self.repo.save_candles(candles_1h, table_name="candles_1h")
                    
                    # Fetch 1-Day Candles
                    candles_1d = self.source.fetch_candles(symbol=symbol, timeframe="1d", limit=limit)
                    if candles_1d:
                        self.repo.save_candles(candles_1d, table_name="candles_1d")

                except Exception as e:
                    # Log the error and move on to the next symbol
                    logger.error(f"Failed to process {symbol}: {e}")

            # Job is done, log and sleep
            logger.info(f"Cycle complete. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)