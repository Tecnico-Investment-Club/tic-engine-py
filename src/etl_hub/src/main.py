import logging
import os
import sys

from src.core.config import settings
from src.etl_db.src.data_source.alpaca import AlpacaSource
from src.etl_db.src.data_source.binance import BinanceSource
from src.etl_db.src.jobs.ingestor import IngestionJob
from src.etl_db.src.persistence.connection import get_db_connection
from src.etl_db.src.persistence.repository import MarketDataRepository

# #region agent log
with open('/home/mv/tic/tic-engine-py/.cursor/debug.log', 'a') as f:
    import json
    handlers_before = len(logging.root.handlers) if hasattr(logging.root, 'handlers') else 0
    f.write(json.dumps({"id": f"log_etl_module_{os.getpid()}", "timestamp": int(__import__('time').time() * 1000), "location": "etl_db/src/main.py:MODULE_LEVEL", "message": "Module-level code executing", "data": {"__name__": __name__, "pid": os.getpid(), "handlers_before": handlers_before}, "runId": "debug1", "hypothesisId": "A,C"}) + '\n')
# #endregion

# Configure Logging
# force=True ensures we reset any handlers that might have been attached
# before this module configures logging, preventing duplicate log lines.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# #region agent log
with open('/home/mv/tic/tic-engine-py/.cursor/debug.log', 'a') as f:
    import json
    f.write(json.dumps({"id": f"log_etl_basicconfig_{os.getpid()}", "timestamp": int(__import__('time').time() * 1000), "location": "etl_db/src/main.py:AFTER_BASICCONFIG", "message": "basicConfig called", "data": {"handlers_after": len(logging.root.handlers), "handler_ids": [id(h) for h in logging.root.handlers], "pid": os.getpid()}, "runId": "debug1", "hypothesisId": "B"}) + '\n')
# #endregion

logger = logging.getLogger("ETL.MAIN")


def main():
    # #region agent log
    with open('/home/mv/tic/tic-engine-py/.cursor/debug.log', 'a') as f:
        import json
        f.write(json.dumps({"id": f"log_etl_main_entry_{os.getpid()}", "timestamp": int(__import__('time').time() * 1000), "location": "etl_db/src/main.py:main()", "message": "main() function called", "data": {"pid": os.getpid(), "handlers_count": len(logging.root.handlers)}, "runId": "debug1", "hypothesisId": "A"}) + '\n')
    # #endregion
    
    # Load Secrets from shared GlobalSettings
    api_key = settings.alpaca_key
    api_secret = settings.alpaca_secret
    db_url = settings.database_url

    # Initialize Infrastructure
    try:
        db_conn = get_db_connection(dsn=db_url)
        repo = MarketDataRepository(db_conn)

        # Initialize Sources
        alpaca_source = AlpacaSource(api_key, api_secret)
        binance_source = BinanceSource()

        # Package the sources into a dictionary
        sources = {
            "alpaca": alpaca_source,
            "binance": binance_source,
        }

        # Log the successful initialization of all components
        # #region agent log
        with open('/home/mv/tic/tic-engine-py/.cursor/debug.log', 'a') as f:
            import json
            f.write(json.dumps({"id": f"log_etl_before_info_{os.getpid()}", "timestamp": int(__import__('time').time() * 1000), "location": "etl_db/src/main.py:BEFORE_INFO_LOG", "message": "About to call logger.info", "data": {"pid": os.getpid(), "handlers_count": len(logging.root.handlers)}, "runId": "debug1", "hypothesisId": "D"}) + '\n')
        # #endregion
        logger.info("Connected to DB, Alpaca, and Binance successfully.")
        # #region agent log
        with open('/home/mv/tic/tic-engine-py/.cursor/debug.log', 'a') as f:
            import json
            f.write(json.dumps({"id": f"log_etl_after_info_{os.getpid()}", "timestamp": int(__import__('time').time() * 1000), "location": "etl_db/src/main.py:AFTER_INFO_LOG", "message": "After calling logger.info", "data": {"pid": os.getpid()}, "runId": "debug1", "hypothesisId": "D"}) + '\n')
        # #endregion
    except Exception as e:
        logger.fatal(f"Initialization failed: {e}")
        raise

    # Start the Job
    job = IngestionJob(sources, repo)
    job.run_loop()


if __name__ == "__main__":
    # #region agent log
    with open('/home/mv/tic/tic-engine-py/.cursor/debug.log', 'a') as f:
        import json
        f.write(json.dumps({"id": f"log_etl_name_main_{os.getpid()}", "timestamp": int(__import__('time').time() * 1000), "location": "etl_db/src/main.py:__main__", "message": "__name__ == '__main__' block executing", "data": {"__name__": __name__, "pid": os.getpid()}, "runId": "debug1", "hypothesisId": "A"}) + '\n')
    # #endregion
    main()