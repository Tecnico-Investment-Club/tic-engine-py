import os
import sys
import logging
import yaml
from dotenv import load_dotenv
from src.core.alerts import send_discord_alert
from src.engine_py.src.strategy.factory import get_strategy

# Add project root to path
sys.path.append(os.getcwd())

# IMPORTS
from src.engine_py.src.ingestion.alpaca_ingestion import AlpacaIngestion
from src.engine_py.src.strategy.example_strat import MovingAverageCrossStrategy
from src.engine_py.src.transformer.transformer import Transformer
from src.engine_py.src.risk.risk import RiskManager
from src.engine_py.src.execution.alpaca_execution import AlpacaExecution
from src.engine_py.src.scheduler.scheduler import Scheduler


# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TradingEngine")

def load_config(path="src/engine_py/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_trading_cycle():
    logger.info("\n--- Starting Trading Cycle ---")
    
    try:
        # Load configuration and environment variables
        config = load_config()
        load_dotenv()
        
        # Set up local variables from environment
        db_url = os.getenv("DATABASE_URL")
        alpaca_key = os.getenv("ALPACA_KEY")
        alpaca_secret = os.getenv("ALPACA_SECRET")

        if not all([db_url, alpaca_key, alpaca_secret]):
            logger.error("Missing required environment variables. Please check your .env file.")
            return

        # Instance creation
        ingestor = AlpacaIngestion(db_url, alpaca_key, alpaca_secret)
        strategy = get_strategy(config["strategy_name"], config["strategy"])
        transformer = Transformer()
        risk = RiskManager(max_qty_per_order=config.get('risk', {}).get('max_order_qty', 100))
        executor = AlpacaExecution(alpaca_key, alpaca_secret)


        # INGESTION
        # Get current cash balance for sizing
        cash = ingestor.get_portfolio_cash()

        # Get a list of target assets from config, defaulting to empty list if not provided
        target_assets = config.get('assets', [])

        # Get the lookback period for historical data, defaulting to 50 if not provided
        lookback = config.get('lookback', 50)

        # Fetch historical market data for target assets with specified lookback
        market_data = ingestor.get_historical_data(target_assets, limit=lookback)

        # Get current positions for sizing and risk management
        current_positions = ingestor.get_current_positions() 


        # STRATEGY
        # Generate target allocations based on strategy logic
        allocations = strategy.generate_allocations(market_data)

        # TRANSFORMER
        # Get current prices for all assets in market data to value positions and size orders
        current_prices = {sym: candles[-1].close for sym, candles in market_data.items() if candles}
        
        # Calculate total equity (cash + value of all positions)
        total_equity = cash
        for symbol, qty in current_positions.items():
            if symbol in current_prices:
                # Value each position based on current price and quantity held
                position_value = qty * current_prices[symbol]

                # Add position value to total equity
                total_equity += position_value
                logger.info(f"Valuing {symbol}: {qty} units @ {current_prices[symbol]} = ${position_value:.2f}")
            else:
                logger.warning(f"Owned asset {symbol} not found in price data.")
        
        # Log total portfolio equity for transparency
        logger.info(f"Total Portfolio Equity: ${total_equity:.2f}")

        # Transform target allocations into executable orders
        orders = transformer.translate_to_orders(allocations, total_equity, current_prices, current_positions)

        # RISK
        # filtered_orders = risk.run()
        filtered_orders = risk.filter_orders(orders)

        # EXECUTION
        # executor.run(filtered_orders)
        if filtered_orders:
            # Execute the filtered orders
            executor.execute_orders(filtered_orders)

            # Send discord message with order details
            order_summary = "\n".join([f"✅ {o.side.upper()} {o.qty:.4f} {o.symbol}" for o in filtered_orders])
            send_discord_alert(config.get("discord_webhook_url"), "📈 Trades Executed", order_summary, 0x00ff00)
        else:
            logger.info("No orders to execute this cycle.")

        logger.info("--- Finishing Trading Cycle ---\n")

    except Exception as e:
        logger.error(f"CRITICAL ERROR in Trading Cycle: {e}", exc_info=True)
        send_discord_alert(config.get("discord_webhook_url"), "🚨 CRITICAL ERROR", str(e), 0xff0000)

def main():
    # Log startup message
    logger.info("TIC-Trading-Pod Initialized.")

    # Load configuration
    config = load_config()

    # Get scheduling interval from config, defaulting to 60 minutes
    interval = config.get('schedule_minutes', 60)
    
    # Initialize the scheduler
    scheduler = Scheduler(interval_minutes=interval)

    # Run the trading cycle in the specified interval
    scheduler.start(run_trading_cycle)

if __name__ == "__main__":
    main()