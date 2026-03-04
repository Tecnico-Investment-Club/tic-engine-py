"""
Stateless trading pipeline: Ingest -> Strategy -> Transform -> Risk -> Execute.

All modules are initialised via the function build_modules
The single-cycle execution lives in execute_cycle
"""

import logging
from datetime import datetime

from src.core.alerts import send_discord_alert
from src.core.config import settings
from src.engine_py.src.config import EngineConfig
from src.engine_py.src.execution.alpaca_executor import AlpacaExecutor
from src.engine_py.src.execution.mock_executor import MockExecutor
from src.engine_py.src.ingestion.ingestor import StandardIngestor
from src.engine_py.src.risk.risk import StandardRiskManager
from src.engine_py.src.strategy.factory import get_strategy
from src.engine_py.src.transformer.transformer import StandardTransformer

logger = logging.getLogger("TradingEngine.Pipeline")


# Module factory

def build_modules(config: EngineConfig):
    """
    Instantiate every pipeline module from the validated config.
    Returns a plain dict of ready-to-use objects.
    """
    # INGESTION
    ingestor = StandardIngestor(
        settings.database_url, settings.alpaca_key, settings.alpaca_secret,
    )

    # STRATEGY
    strategy = get_strategy(
        config.strategy.name,
        config.strategy.params,
        class_path=config.strategy.class_path,
    )

    # TRANSFORM
    transformer = StandardTransformer()

    # RISK MANAGEMENT
    risk = StandardRiskManager(
        max_notional_per_order=config.risk.max_notional_per_order,
        max_total_notional_per_symbol=config.risk.max_total_notional_per_symbol,
    )

    # EXECUTION
    if config.execution.type == "mock":
        executor = MockExecutor()
    elif config.execution.type == "alpaca":
        executor = AlpacaExecutor(
            settings.alpaca_key, settings.alpaca_secret, paper=config.execution.paper,
        )
    else:
        raise ValueError(f"Unknown executor type '{config.execution.type}'.")

    # Return all modules in a simple dict for easy unpacking in the main loop
    return {
        "ingestor": ingestor,
        "strategy": strategy,
        "transformer": transformer,
        "risk": risk,
        "executor": executor,
    }


# Single-cycle execution

def execute_cycle(
    *,
    ingestor: StandardIngestor,
    strategy,
    transformer: StandardTransformer,
    risk: StandardRiskManager,
    executor,
    assets: list[str],
    lookback: int,
    timeframe: str,
    webhook: str | None,
    latest_candle_time: datetime,
):
    """Runs the full, strictly unidirectional trading pipeline."""
    logger.info(
        f"\n--- Starting Cycle ---"
        f"\nLatest Candle Time: {latest_candle_time.isoformat()}"
        f"\nAssets: {assets}, Lookback: {lookback}, Timeframe: {timeframe}"    
    )
    try:
        # INGESTION
        market_data = ingestor.get_data(assets, limit=lookback, timeframe=timeframe)
        state = ingestor.get_live_portfolio_state(market_data)
        logger.info(f"Total Portfolio Equity: ${state.total_equity:.2f}")

        # STRATEGY
        allocations = strategy.compute_weights(market_data, state)

        # TRANSFORM
        raw_orders = transformer.transform(allocations, state)

        # RISK MANAGEMENT
        safe_orders = risk.validate_orders(raw_orders, state)

        # EXECUTION
        if safe_orders:
            receipts = executor.execute_orders(safe_orders)
            if webhook and receipts:
                summary = "\n".join(
                    f"\u2705 {r.side.upper()} {r.qty:.4f} {r.symbol} [{r.status}]"
                    for r in receipts
                )
                send_discord_alert(
                    webhook,
                    f"\U0001f4c8 {strategy.__class__.__name__} Trades",
                    summary,
                    0x00FF00,
                )
        else:
            logger.info("No valid orders to execute in this cycle.")

    except Exception as e:
        logger.error(f"Engine cycle failed: {e}")
        raise

