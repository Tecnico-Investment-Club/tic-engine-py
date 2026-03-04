import logging
from typing import List
from datetime import datetime, timezone

from trading_pod.interfaces.IIngestion import IIngestion
from trading_pod.interfaces.IStrategy import IStrategy
from trading_pod.interfaces.ITransformer import ITransformer
from trading_pod.interfaces.IExecution import IExecution
from core.utils import parse_time_interval

logger = logging.getLogger("TRADING.PIPELINE")

class TradingPipeline:
    def __init__(
        self,
        ingestor: IIngestion,
        strategy: IStrategy,
        transformer: ITransformer,
        executor: IExecution,
        symbols: List[str],
        timeframe: str,
        lookback: int,
        trade_every: str = None  # Add the cooldown parameter
    ):
        self.ingestor = ingestor
        self.strategy = strategy
        self.transformer = transformer
        self.executor = executor
        
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback = lookback
        
        # Parse the execution frequency (Defaults to 0 seconds if None)
        self.cooldown_period = parse_time_interval(trade_every) if trade_every else parse_time_interval("0s")
        self.last_execution_time = None

    def handle_pubsub_event(self, payload: dict) -> None:
        """
        Triggered by messaging.py when the ETL Hub broadcasts an update.
        """
        logger.info(f"Pub/Sub Event Received: {payload.get('message', 'Triggered')}")
        
        now = datetime.now(timezone.utc)
        
        # The Cooldown / Throttle Logic
        if self.last_execution_time:
            time_since_last_run = now - self.last_execution_time
            if time_since_last_run < self.cooldown_period:
                logger.info(
                    f"Skipping execution. Cooldown active. "
                    f"(Needs {self.cooldown_period}, elapsed {time_since_last_run})"
                )
                return

        # If we passed the throttle, run the heavy logic
        logger.info("Cooldown passed. Starting Trading Cycle...")
        self._run_cycle()
        
        # Record the successful execution time to start the cooldown timer
        self.last_execution_time = datetime.now(timezone.utc)

    def _run_cycle(self) -> None:
        """The actual Read -> Compute -> Write loop."""
        try:
            # 1. READ
            market_data = self.ingestor.fetch_data(self.symbols, self.timeframe, self.lookback)
            if not market_data.data:
                logger.warning("No market data returned. Aborting cycle.")
                return
                
            portfolio_state = self.ingestor.fetch_portfolio_state(market_data)

            # 2. COMPUTE
            allocations = self.strategy.generate_allocations(market_data)
            orders = self.transformer.generate_orders(allocations, portfolio_state)

            # 3. WRITE
            if not orders:
                logger.info("No actionable orders generated.")
            else:
                receipts = self.executor.execute_orders(orders)
                for r in receipts:
                    logger.info(f" -> {r.side.name} {r.qty} {r.symbol} | Status: {r.status}")

        except Exception as e:
            logger.error(f"Critical failure during trading cycle: {e}", exc_info=True)