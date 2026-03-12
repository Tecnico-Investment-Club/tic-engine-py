import logging
from typing import List
from datetime import datetime, timezone

from trading_pod.interfaces.IIngestion import IIngestion
from trading_pod.interfaces.IStrategy import IStrategy
from trading_pod.interfaces.ITransformer import ITransformer
from trading_pod.interfaces.IExecution import IExecution
from core.utils import parse_time_interval
from alpaca.common.exceptions import APIError

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
        Handle a Pub/Sub event from the ETL Hub.

        Runs the trading cycle only if the cooldown period has passed.
        """
        logger.info(f"Pub/Sub Event Received: {payload.get('message', 'Triggered')}")

        now = datetime.now(timezone.utc)

        # Skip if we're still in the cooldown period
        if self.last_execution_time:
            time_since_last_run = now - self.last_execution_time
            if time_since_last_run < self.cooldown_period:
                logger.info(
                    f"Skipping execution. Cooldown active "
                    f"(Cooldown: {self.cooldown_period}, Elapsed: {time_since_last_run})"
                )
                return

        # Run the trading cycle
        logger.info("Cooldown passed. Running trading cycle...")
        self._run_cycle()

        # Update last execution time, aligned to the cooldown
        cooldown_sec = self.cooldown_period.total_seconds()
        if cooldown_sec > 0:
            epoch = now.timestamp()
            aligned_epoch = epoch - (epoch % cooldown_sec)
            self.last_execution_time = datetime.fromtimestamp(aligned_epoch, tz=timezone.utc)
        else:
            self.last_execution_time = now


    def _run_cycle(self) -> None:
        """The actual Read -> Compute -> Write loop."""
        try:
            # CLEAN ALL PENDING TRADES
            self.executor.cancel_all_open_orders()

            # INGEST
            market_data = self.ingestor.fetch_data(self.symbols, self.timeframe, self.lookback)
            if not market_data.data:
                logger.warning("No market data returned. Aborting cycle.")
                return
                
            # FETCH
            portfolio_state = self.ingestor.fetch_portfolio_state(market_data)

            # COMPUTE
            allocations = self.strategy.generate_allocations(market_data)
            orders = self.transformer.generate_orders(allocations, portfolio_state)

            # EXECUTE
            if not orders:
                logger.info("No actionable orders generated.")
            else:
                receipts = self.executor.execute_orders(orders)
                for r in receipts:
                    logger.info(f" -> {r.side.name} {r.qty} {r.symbol} | Status: {r.status}")

        except Exception as e:
            logger.error(f"Critical failure during trading cycle: {e}", exc_info=True)