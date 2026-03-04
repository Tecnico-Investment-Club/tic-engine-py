"""
Slim entry-point for the Trading Engine pod.
"""

import logging
import os
import sys
import time
from datetime import datetime

from src.engine_py.src.config import load_engine_config
from src.engine_py.src.pipeline import build_modules, execute_cycle

# #region agent log
with open('/home/mv/tic/tic-engine-py/.cursor/debug.log', 'a') as f:
    import json
    f.write(json.dumps({"id": f"log_engine_module_{os.getpid()}", "timestamp": int(__import__('time').time() * 1000), "location": "engine_py/src/main.py:MODULE_LEVEL", "message": "Module-level code executing", "data": {"__name__": __name__, "module": sys.modules.get(__name__), "pid": os.getpid(), "handlers_before": len(logging.root.handlers)}, "runId": "debug1", "hypothesisId": "A,C"}) + '\n')
# #endregion

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# #region agent log
with open('/home/mv/tic/tic-engine-py/.cursor/debug.log', 'a') as f:
    import json
    f.write(json.dumps({"id": f"log_engine_basicconfig_{os.getpid()}", "timestamp": int(__import__('time').time() * 1000), "location": "engine_py/src/main.py:AFTER_BASICCONFIG", "message": "basicConfig called", "data": {"handlers_after": len(logging.root.handlers), "handler_ids": [id(h) for h in logging.root.handlers], "pid": os.getpid()}, "runId": "debug1", "hypothesisId": "B"}) + '\n')
# #endregion

logger = logging.getLogger("TradingEngine")


# Runner

class TradingEngineRunner:
    def __init__(self):
        self.config = load_engine_config()

        strat = self.config.strategy
        self.timeframe = strat.timeframe
        self.trade_every = strat.trade_every or self.timeframe

        self.modules = build_modules(self.config)
        self.poll_interval = self.config.poll_interval_seconds

        self.watermark: datetime | None = None
        self.last_trade_period: datetime | None = None

    # Helpers

    def get_latest_db_timestamp(self) -> datetime | None:
        """Lightweight query: MAX(timestamp) for the first tracked asset."""
        assets = self.config.assets
        if not assets:
            return None

        market_data = self.modules["ingestor"].get_data(
            [assets[0]], limit=1, timeframe=self.timeframe,
        )
        candles = market_data.data.get(assets[0], [])
        return candles[-1].timestamp if candles else None

    # Boundary / freshness logic

    @staticmethod
    def _truncate_to_period(ts: datetime, period: str) -> datetime:
        """Truncate a timestamp to the start of its period."""
        if period == "1d":
            return ts.replace(hour=0, minute=0, second=0, microsecond=0)
        # Handle Nh patterns (1h, 2h, 4h, 6h, 12h …)
        if period.endswith("h"):
            n = int(period[:-1])
            truncated_hour = (ts.hour // n) * n
            return ts.replace(hour=truncated_hour, minute=0, second=0, microsecond=0)
        return ts

    def _crossed_trade_boundary(self, new_ts: datetime) -> bool:
        """True when new_ts falls in a different trade_every period."""
        if self.last_trade_period is None:
            return True
        return (
            self._truncate_to_period(new_ts, self.trade_every)
            != self.last_trade_period
        )

    def _is_fresh(self, ts: datetime) -> bool:
        """True if *ts* falls within the current trade_every period."""
        now = datetime.now(ts.tzinfo)
        return (
            self._truncate_to_period(ts, self.trade_every)
            == self._truncate_to_period(now, self.trade_every)
        )

    def _wait_for_data(self):
        """Poll DB until a candle exists within the current trade period."""
        logger.info("Waiting for fresh market data in DB...")
        while True:
            latest = self.get_latest_db_timestamp()
            if latest and self._is_fresh(latest):
                logger.info(f"Fresh data available (latest: {latest}).")
                return
            logger.info(f"No fresh data yet. Retrying in {self.poll_interval}s...")
            time.sleep(self.poll_interval)

    # Cycle wrapper

    def _run_cycle(self, candle_time: datetime):
        execute_cycle(
            **self.modules,
            assets=self.config.assets,
            lookback=self.config.lookback,
            timeframe=self.timeframe,
            webhook=self.config.discord_webhook_url,
            latest_candle_time=candle_time,
        )

    # Main loop

    def run(self):
        """
        Wait for fresh data (ETL bootstrap).
        Execute first trade immediately.
        Poll for new candles; trade on boundary crossings.
        """
        # Wait for the first candle to arrive
        self._wait_for_data()

        # Set the watermark to the latest DB timestamp
        self.watermark = self.get_latest_db_timestamp()
        logger.info(f"Watermark set to {self.watermark}. Running first trade cycle.")

        # Run the first cycle immediately to avoid waiting for the next boundary
        self._run_cycle(self.watermark)
        self.last_trade_period = self._truncate_to_period(self.watermark, self.trade_every)

        logger.info(
            f"Entering poll loop (interval={self.poll_interval}s, "
            f"trade_every={self.trade_every}). Press Ctrl+C to stop."
        )
        try:
            while True:
                time.sleep(self.poll_interval)
                latest = self.get_latest_db_timestamp()

                if not latest or latest <= self.watermark:
                    continue

                self.watermark = latest

                if self._crossed_trade_boundary(latest):
                    logger.info(
                        f"New {self.trade_every} trade boundary at {latest} "
                        f"(prev trade period: {self.last_trade_period})."
                    )
                    self._run_cycle(latest)
                    self.last_trade_period = self._truncate_to_period(
                        latest, self.trade_every,
                    )
                else:
                    logger.debug(
                        f"New {self.timeframe} candle at {latest}, "
                        f"but still within {self.trade_every} period. Skipping trade."
                    )
        except KeyboardInterrupt:
            logger.info("Engine stopped by user.")


# Entry point

def main():
    # #region agent log
    with open('/home/mv/tic/tic-engine-py/.cursor/debug.log', 'a') as f:
        import json
        f.write(json.dumps({"id": f"log_engine_main_entry_{os.getpid()}", "timestamp": int(__import__('time').time() * 1000), "location": "engine_py/src/main.py:main()", "message": "main() function called", "data": {"pid": os.getpid(), "handlers_count": len(logging.root.handlers)}, "runId": "debug1", "hypothesisId": "A"}) + '\n')
    # #endregion
    
    runner = TradingEngineRunner()
    
    # #region agent log
    with open('/home/mv/tic/tic-engine-py/.cursor/debug.log', 'a') as f:
        import json
        f.write(json.dumps({"id": f"log_engine_before_info_{os.getpid()}", "timestamp": int(__import__('time').time() * 1000), "location": "engine_py/src/main.py:BEFORE_INFO_LOG", "message": "About to call logger.info", "data": {"pid": os.getpid(), "handlers_count": len(logging.root.handlers)}, "runId": "debug1", "hypothesisId": "D"}) + '\n')
    # #endregion
    
    logger.info(
        f"Starting TradingEngineRunner with timeframe={runner.timeframe}, "
        f"trade_every={runner.trade_every}, poll_interval={runner.poll_interval}s."
    )
    
    # #region agent log
    with open('/home/mv/tic/tic-engine-py/.cursor/debug.log', 'a') as f:
        import json
        f.write(json.dumps({"id": f"log_engine_after_info_{os.getpid()}", "timestamp": int(__import__('time').time() * 1000), "location": "engine_py/src/main.py:AFTER_INFO_LOG", "message": "After calling logger.info", "data": {"pid": os.getpid()}, "runId": "debug1", "hypothesisId": "D"}) + '\n')
    # #endregion
    
    runner.run()


if __name__ == "__main__":
    # #region agent log
    with open('/home/mv/tic/tic-engine-py/.cursor/debug.log', 'a') as f:
        import json
        f.write(json.dumps({"id": f"log_engine_name_main_{os.getpid()}", "timestamp": int(__import__('time').time() * 1000), "location": "engine_py/src/main.py:__main__", "message": "__name__ == '__main__' block executing", "data": {"__name__": __name__, "pid": os.getpid()}, "runId": "debug1", "hypothesisId": "A"}) + '\n')
    # #endregion
    main()
