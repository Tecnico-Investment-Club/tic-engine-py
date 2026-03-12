import logging
import time
from datetime import datetime, timedelta, timezone

from core.config import settings, load_yaml_config
from core.discord import setup_discord_logging

# Cleaned up imports
from etl_hub.database import PostgresDatabase, DataJanitor
from etl_hub.providers.alpaca import AlpacaProvider
from etl_hub.providers.binance import BinanceProvider
from etl_hub.pipeline import ETLPipeline

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Attach Discord logging handler for ERROR+ logs (no-op if webhook not set)
setup_discord_logging(settings.discord_webhook_url, level=logging.ERROR)

logger = logging.getLogger("ETL.MAIN")

def main():
    logger.info("Initializing ETL Hub Infrastructure...")

    # logger.error("TEST ETL ERROR: Intentional crash to verify Discord logging.")
    
    # Initialize DB and Janitor
    db = PostgresDatabase(settings.database_url)
    janitor = DataJanitor(db)
    
    # Initialize Providers
    alpaca = AlpacaProvider(api_key=settings.alpaca_api_key, api_secret=settings.alpaca_api_secret)
    binance = BinanceProvider()
    
    # Assemble the Pipeline
    pipeline = ETLPipeline(
        db=db,
        providers={"alpaca": alpaca, "binance": binance},
        janitor=janitor,
        db_url=settings.database_url
    )

    logger.info("ETL Hub Scheduler Active.")

    # The Infinite Loop
    while True:
        try:
            # Reload YAML inside the loop so assets can be updated without rebuilding
            config = load_yaml_config("src/etl_hub/config.yaml")
            
            # Execute the database synchronization
            pipeline.execute(config)
            
        except Exception as e:
            logger.error(f"Critical failure in ETL loop: {e}")

        # Sscheduling Logic
        now = datetime.now(timezone.utc)
        
        # Calculate the exact target time (minute 1, second 0)
        next_run = now.replace(minute=1, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(hours=1)
        
        sleep_seconds = (next_run - now).total_seconds()
        logger.info(f"Sleeping for {sleep_seconds / 60:.2f} minutes until exactly {next_run.strftime('%H:%M:%S')} UTC...")
        
        # Wait Loop
        while datetime.now(timezone.utc) < next_run:
            time.sleep(1) # Check the clock every 1 second

if __name__ == "__main__":
    main()