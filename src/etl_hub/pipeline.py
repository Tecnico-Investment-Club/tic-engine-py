import logging
from typing import Dict, Literal, List
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

    def _sync_batch(self, provider: IProvider, symbols: List[str], timeframe: Literal["1h", "1d"], config: dict):
        """
        Groups symbols by their required fetch limits, then syncs them in batches.
        """
        table_name = f"candles_{timeframe}"
        job_settings = config.get("job_settings", {})
        bootstrap_limit = job_settings.get("bootstrap_limit", 500)
        update_limit = job_settings.get("update_limit", 10)

        bootstrap_symbols = []
        update_symbols = []

        # Group symbols based on local DB state
        for symbol in symbols:
            internal_symbol = normalize_symbol(symbol)
            latest_ts = self.db.get_latest_timestamp(internal_symbol, table_name)
            
            if latest_ts is None:
                bootstrap_symbols.append(symbol)
            else:
                update_symbols.append(symbol)

        # Process both groups in chunks
        if bootstrap_symbols:
            logger.info(f"Bootstrapping {len(bootstrap_symbols)} symbols ({bootstrap_limit} candles, {timeframe}).")
            self._fetch_and_save_chunks(provider, bootstrap_symbols, timeframe, bootstrap_limit, table_name)

        if update_symbols:
            logger.info(f"Updating {len(update_symbols)} symbols ({update_limit} candles, {timeframe}).")
            self._fetch_and_save_chunks(provider, update_symbols, timeframe, update_limit, table_name)

    def _fetch_and_save_chunks(self, provider: IProvider, symbols: List[str], timeframe: str, limit: int, table_name: str):
        """
        Breaks a list of symbols into manageable chunks for the provider API.
        """
        chunk_size = 250  # Safe size for most API limits
        
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            try:
                # Fetch dictionary of {symbol: [Candles]}
                batch_results = provider.fetch_candles(chunk, timeframe, limit=limit)
                
                # Flatten the dictionary into a single large list of Candles
                all_chunk_candles = []
                for _, candles in batch_results.items():
                    all_chunk_candles.extend(candles)
                
                # Batch insert to DB
                if all_chunk_candles:
                    self.db.save_candles(all_chunk_candles, table_name)
                    logger.info(f"Saved chunk of {len(all_chunk_candles)} total candles to {table_name}.")
                    
            except Exception as e:
                logger.error(f"Failed to sync chunk on {provider.get_provider_name()}: {e}")

    def execute(self, current_config: dict):
        """
        Executes a full synchronization pipeline for all configured assets.
        """
        logger.info("\n--- Starting ETL Pipeline Execution ---")
        
        self.db.connect()
        asset_groups = current_config.get("assets", {})

        for provider_name, symbols in asset_groups.items():
            provider = self.providers.get(provider_name)

            if not provider: 
                logger.warning(f"Provider '{provider_name}' is not in the ETL.")
                continue
            
            logger.info(f"Synchronizing {len(symbols)} assets via {provider.get_provider_name()}.")

            # Execute batch syncing for both timeframes
            self._sync_batch(provider, symbols, "1h", current_config)
            self._sync_batch(provider, symbols, "1d", current_config)
        
        # Prune old data
        retention_days = current_config.get("job_settings", {}).get("retention_days", 365)
        self.janitor.run_janitor(days_to_keep=retention_days)

        self.notifier.notify(
            channel="market_data_ready", 
            payload={"status": "success", "message": "New candles available"}
        )

        logger.info("--- ETL Pipeline Execution Complete ---\n")