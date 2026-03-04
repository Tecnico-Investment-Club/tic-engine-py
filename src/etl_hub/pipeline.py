import logging
from typing import Dict, Literal
from core.messaging import PostgresNotifier

from core.utils import normalize_symbol
from etl_hub.interfaces.IProvider import IProvider
from etl_hub.interfaces.IDatabase import IDatabase

logger = logging.getLogger("ETL.PIPELINE")

class ETLPipeline:
    def __init__(self, db, providers, janitor, db_url): 
        self.db = db
        self.providers = providers
        self.janitor = janitor
        self.notifier = PostgresNotifier(db_url)

    def _sync_asset_candles(self, provider: IProvider, symbol: str, timeframe: Literal["1h", "1d"], config: dict):
        """
        Calculates the missing delta and synchronizes the local database.
        """
        table_name = f"candles_{timeframe}"
        internal_symbol = normalize_symbol(symbol)

        # Check local DB state
        latest_ts = self.db.get_latest_timestamp(internal_symbol, table_name)
        
        # Determine fetch limits
        job_settings = config.get("job_settings", {})
        bootstrap = job_settings.get("bootstrap_limit", 500)
        update = job_settings.get("update_limit", 10)
        
        fetch_limit = bootstrap if latest_ts is None else update
        
        # Extract and Load the Candles from a given Symbol in a given Provider with a given timeframe
        candles = provider.fetch_candles(symbol, timeframe, limit=fetch_limit)
        if candles:
            self.db.save_candles(candles, table_name)

    def execute(self, current_config: dict):
        """
        Executes a full synchronization pipeline for all configured assets.
        """
        logger.info("\n--- Starting ETL Pipeline Execution ---")
        
        # Connect to the Database
        self.db.connect()

        # Fetch the desired assets
        asset_groups = current_config.get("assets", {})

        # Loop through them all
        for provider_name, symbols in asset_groups.items():
            # Get provider
            provider = self.providers.get(provider_name)

            # Safety check
            if not provider: 
                logger.warning(f"Provider '{provider_name}' is not in the ETL.")
                continue
            
            logger.info(f"Synchronizing {len(symbols)} assets via {provider.get_provider_name()}.")

            # For each symbol, fetch the candles
            for symbol in symbols:
                try:
                    self._sync_asset_candles(provider, symbol, "1h", current_config)
                    self._sync_asset_candles(provider, symbol, "1d", current_config)
                except Exception as e:
                    logger.error(f"Failed to sync {symbol} on {provider_name}: {e}")
        
        # Prune old data
        retention_days = current_config.get("job_settings", {}).get("retention_days", 365)
        self.janitor.run_janitor(days_to_keep=retention_days)

        self.notifier.notify(
            channel="market_data_ready", 
            payload={"status": "success", "message": "New candles available"}
        )

        logger.info("--- ETL Pipeline Execution Complete ---\n")