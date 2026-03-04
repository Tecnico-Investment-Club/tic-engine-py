import logging
import sys

# Load your configuration parser 
from core.config import settings, load_yaml_config
from core.messaging import PostgresListener # ADD THIS IMPORT
from core.discord import setup_discord_logging

# Concrete Implementations
from trading_pod.ingestion import DataIngestion
from trading_pod.strategy.factory import get_strategy
from trading_pod.transformer import StandardTransformer
from trading_pod.execution import AlpacaExecution
from trading_pod.pipeline import TradingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Attach Discord logging handler for ERROR+ logs (no-op if webhook not set)
setup_discord_logging(settings.discord_webhook_url, level=logging.ERROR)

logger = logging.getLogger("TRADING.MAIN")

def main():
    logger.info("Initializing Trading Pod Infrastructure...")

    
    #logger.critical("TEST TRADING CRITICAL: Intentional crash to verify Discord logging.")

    # 1. Load configuration 
    config = load_yaml_config("src/trading_pod/config.yaml")
    
    lookback = config.get("lookback", 50)
    symbols = config.get("assets", [])
    
    strat_cfg = config.get("strategy", {})
    strategy_name = strat_cfg.get("name", "PingPongStrat")
    timeframe = strat_cfg.get("timeframe", "1h")
    trade_every = strat_cfg.get("trade_every", None) # Grab the cooldown from config
    strat_params = strat_cfg.get("params", {})

    # 2. Instantiate dependencies
    ingestor = DataIngestion(
        db_url=settings.database_url,
        alpaca_key=settings.alpaca_api_key,
        alpaca_secret=settings.alpaca_api_secret
    )
    strategy = get_strategy(strategy_name=strategy_name, params=strat_params)
    transformer = StandardTransformer(buffer_pct=0.05)
    executor = AlpacaExecution(
        api_key=settings.alpaca_api_key,
        api_secret=settings.alpaca_api_secret,
        paper=True 
    )

    # 3. Assemble the pipeline
    pipeline = TradingPipeline(
        ingestor=ingestor,
        strategy=strategy,
        transformer=transformer,
        executor=executor,
        symbols=symbols,
        timeframe=timeframe,
        lookback=lookback,
        trade_every=trade_every # Pass it to the pipeline
    )

    # 4. Instantiate the Postgres Listener
    listener = PostgresListener(
        db_url=settings.database_url,
        channel="market_data_ready",
        callback=pipeline.handle_pubsub_event
    )

    logger.info("Trading Pod active. Waiting for ETL Hub broadcasts on 'market_data_ready'...")

    # Start the blocking event loop (Sleeps at 0 CPU until triggered)
    try:
        listener.start()
    except KeyboardInterrupt:
        logger.info("Manual interruption detected. Shutting down.")
        listener.stop()
    except Exception as e:
        logger.error(f"Listener loop crashed: {e}")

if __name__ == "__main__":
    main()