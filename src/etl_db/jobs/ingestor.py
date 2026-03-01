import time
import logging
import yaml
from typing import Dict

from src.etl_db.data_source.base import DataSource
from src.etl_db.persistence.repository import MarketDataRepository

logger = logging.getLogger(__name__)

class IngestionJob:
    def __init__(self, sources: Dict[str, DataSource], repo: MarketDataRepository, config_path: str = "src/etl_db/config.yaml"):
        self.sources = sources
        self.repo = repo
        self.config = self._load_config(config_path)

    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def run_loop(self):
        logger.info("--- Starting ETL Ingestion Loop ---")
        
        while True:
            asset_groups = self.config.get("assets", {})
            sleep_time = self.config["job_settings"].get("sleep_seconds", 3600)
            limit = self.config["job_settings"].get("lookback_limit", 100)

            # Loop over each exchange defined in config (e.g., 'alpaca', 'binance')
            for exchange_name, symbols in asset_groups.items():
                
                # Get the correct API client
                source = self.sources.get(exchange_name)
                if not source:
                    logger.error(f"No data source configured for '{exchange_name}'")
                    continue

                logger.info(f"Starting cycle for {len(symbols)} assets on {exchange_name.upper()}.")

                for symbol in symbols:
                    try:
                        # Fetch and Save 1H
                        candles_1h = source.fetch_candles(symbol, "1h", limit)
                        if candles_1h:
                            self.repo.save_candles(candles_1h, "candles_1h")
                        
                        # Fetch and Save 1D
                        candles_1d = source.fetch_candles(symbol, "1d", limit)
                        if candles_1d:
                            self.repo.save_candles(candles_1d, "candles_1d")

                    except Exception as e:
                        logger.error(f"Failed to process {symbol} on {exchange_name}: {e}")

            logger.info(f"Cycle complete. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)