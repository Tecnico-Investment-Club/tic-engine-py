import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to python path so imports work inside Docker
sys.path.append(os.getcwd())


from src.etl_db.persistence.connection import get_db_connection
from src.etl_db.persistence.repository import MarketDataRepository
from src.etl_db.data_source.alpaca import AlpacaSource
from src.etl_db.data_source.binance import BinanceSource
from src.etl_db.jobs.ingestor import IngestionJob

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ETL_Main")

def main():
    # Load Secrets
    api_key = os.getenv("ALPACA_KEY")
    api_secret = os.getenv("ALPACA_SECRET")

    # BINANCE DOES NOT REQUIRE API KEYS FOR PUBLIC DATA
    
    if not api_key or not api_secret:
        logger.fatal("Missing ALPACA_KEY or ALPACA_SECRET env vars.")
        sys.exit(1)

    # Initialize Infrastructure
    try:
        db_conn = get_db_connection()
        repo = MarketDataRepository(db_conn)
        
        # Initialize Sources
        alpaca_source = AlpacaSource(api_key, api_secret)
        binance_source = BinanceSource()
        
        # Package the sources into a dictionary
        sources = {
            "alpaca": alpaca_source,
            "binance": binance_source
        }
        
        # Log the successful initialization of all components
        logger.info("Connected to DB, Alpaca, and Binance successfully.")
    except Exception as e:
        logger.fatal(f"Initialization failed: {e}")
        sys.exit(1)

    # Start the Job
    job = IngestionJob(sources, repo)
    job.run_loop()

if __name__ == "__main__":
    main()