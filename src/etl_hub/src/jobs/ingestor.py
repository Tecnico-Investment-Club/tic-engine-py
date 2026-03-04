import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Literal

from src.core.config import load_yaml_config
from src.core.utils import normalize_symbol, timeframe_to_table
from src.etl_db.src.data_source.base import DataSource
from src.etl_db.src.persistence.repository import MarketDataRepository

logger = logging.getLogger("ETL.Ingestor")


class IngestionJob:
    def __init__(
        self,
        sources: Dict[str, DataSource],
        repo: MarketDataRepository,
        config_path: str = "src/etl_db/config.yaml",
    ):
        self.sources = sources
        self.repo = repo
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        return load_yaml_config(self.config_path)

    def _process_symbol_timeframe(
        self, source: DataSource, symbol: str, timeframe: Literal["1h", "1d"]
    ):
        # Initialize variables
        table_name = timeframe_to_table(timeframe)
        internal_symbol = normalize_symbol(symbol)

        # Get the most recent timestamp
        latest_ts = self.repo.get_latest_timestamp(internal_symbol, table_name)
        
        # Determine fetch limit based on whether this is a bootstrap or update
        bootstrap = self.config["job_settings"].get("bootstrap_limit", 500)
        update = self.config["job_settings"].get("update_limit", 10)
        
        fetch_limit = bootstrap if latest_ts is None else update
        
        candles = source.fetch_candles(symbol, timeframe, limit=fetch_limit)
        if candles:
            self.repo.save_candles(candles, table_name)

    def run_once(self):
        """Logic to run a single ETL cycle."""
        logger.info("--- Starting Scheduled ETL Cycle ---")
        
        # Reload the config on every cycle to pick up changes without restarting the container
        self.config = self._load_config()
        asset_groups = self.config.get("assets", {})

        for exchange_name, symbols in asset_groups.items():
            # Get the corresponding data source
            source = self.sources.get(exchange_name)
            if not source: continue

            logger.info(f"Processing {len(symbols)} assets on {exchange_name.upper()}.")

            # Process each symbol for both timeframes
            for symbol in symbols:
                try:
                    self._process_symbol_timeframe(source, symbol, "1h")
                    self._process_symbol_timeframe(source, symbol, "1d")
                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {e}")
        
        # Prune old candles after each cycle
        days = self.config["job_settings"].get("retention_days", 365)
        self.repo.prune_old_data(retention_days=days)

        logger.info("--- Cycle Complete ---")

    def run_loop(self):
        logger.info("--- Initializing ETL Scheduler ---")

        # Run immediately on startup so we don't wait an hour to see if it works
        self.run_once()

        logger.info("Scheduler Active. Next run at :01 past the hour.")

        # Sleep until :01 of each hour, then run a cycle.
        while True:
            now = datetime.now(timezone.utc)
            next_run = now.replace(minute=1, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(hours=1)
            time.sleep((next_run - now).total_seconds())
            self.run_once()