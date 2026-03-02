import os
import sys
import logging
import yaml
from dotenv import load_dotenv
from time import sleep

# Add project root to path
sys.path.append(os.getcwd())

# IMPORTS (Updated to match the new architecture)
from src.core.alerts import send_discord_alert
from src.engine_py.src.strategy.factory import get_strategy
from src.engine_py.src.ingestion.ingestor import StandardIngestor
from src.engine_py.src.transformer.transformer import StandardTransformer
from src.engine_py.src.risk.risk import StandardRiskManager
from src.engine_py.src.execution.alpaca_executor import AlpacaExecutor
from src.engine_py.src.scheduler.scheduler import Scheduler

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TradingEngine")

def load_config():
    path = os.getenv("CONFIG_PATH", "")
    
    logger.info(f"Loading configuration from: {path}")
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {path}. Please check your volume mounts.")
        sys.exit(1)

def run_trading_cycle(config, ingestor, strategy, transformer, risk, executor):
    """
    The core pipeline with unidirectional data flow.
    """
    logger.info("\n--- Starting Trading Cycle ---")
    
    try:
        # INGESTION
        target_assets = config.get('assets', [])
        lookback = config.get('lookback', 50)
        
        market_data = ingestor.get_data(target_assets, lookback)
        state = ingestor.get_live_portfolio_state(market_data)
        
        logger.info(f"Total Portfolio Equity: ${state.total_equity:.2f}")
        
        # STRATEGY
        allocations = strategy.compute_weights(market_data, state)

        
        # TRANSFORMER
        raw_orders = transformer.transform(allocations, state)


        # RISK MANAGEMENT
        safe_orders = risk.validate_orders(raw_orders, state)


        # EXECUTION
        if safe_orders:
            receipts = executor.execute_orders(safe_orders)

            # Send discord message with order details
            webhook_url = config.get("discord_webhook_url")
            if webhook_url and receipts:
                order_summary = "\n".join([
                    f"✅ {r.side.upper()} {r.qty:.4f} {r.symbol} [{r.status}]" 
                    for r in receipts
                ])
                send_discord_alert(webhook_url, "📈 Trades Executed", order_summary, 0x00ff00)
        else:
            logger.info("No valid orders to execute this cycle.")

        logger.info("--- Finishing Trading Cycle ---\n")

    except Exception as e:
        logger.error(f"CRITICAL ERROR in Trading Cycle: {e}", exc_info=True)
        webhook_url = config.get("discord_webhook_url")
        if webhook_url:
            send_discord_alert(webhook_url, "🚨 CRITICAL ERROR", str(e), 0xff0000)


def main():
    logger.info("TIC-Trading-Pod Initialized.")
    for i in range(20):
        logger.info(f"Waiting for {i+1}/20 seconds")
        sleep(1)

    # Load config and env vars ONCE
    load_dotenv()
    config = load_config()
    
    db_url = os.getenv("DATABASE_URL")
    alpaca_key = os.getenv("ALPACA_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET")

    if not db_url or not alpaca_key or not alpaca_secret:
        logger.error("Missing required environment variables. Exiting.")
        sys.exit(1)

    # Initialize the pipeline components once at startup
    ingestor = StandardIngestor(db_url, alpaca_key, alpaca_secret)
    
    # Extract the strategy block, defaulting to an empty dict if missing
    strat_config = config.get('strategy', {})
    
    # Safely extract the name and params
    strat_name = strat_config.get('name', 'ExampleStrategy')
    strat_params = strat_config.get('params', {})
    
    # Initialize the strategy
    strategy = get_strategy(strat_name, strat_params)

    transformer = StandardTransformer()
    risk = StandardRiskManager()
    executor = AlpacaExecutor(alpaca_key, alpaca_secret)

    # Setup Scheduler
    interval = config.get('schedule_minutes', 60)
    scheduler = Scheduler(interval_minutes=interval)

    # Run the loop, passing the initialized components to the trading cycle function
    scheduler.start(lambda: run_trading_cycle(config, ingestor, strategy, transformer, risk, executor))

if __name__ == "__main__":
    main()