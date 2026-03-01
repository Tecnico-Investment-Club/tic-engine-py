import time
import logging
import schedule
from typing import Callable
from src.engine_py.src.scheduler.base import IScheduler

logger = logging.getLogger(__name__)

class Scheduler(IScheduler):
    """
    Handles the timing and execution loop of the trading pod.
    Abstracts away the 'while True' loop and the 'schedule' library.
    """
    def __init__(self, interval_minutes: int = 60):
        self.interval = interval_minutes

    def start(self, job_func: Callable):
        """
        Starts the blocking scheduler loop.
        
        :param job_func: The function to execute (e.g., run_trading_cycle)
        """
        logger.info(f"Scheduler: Configured to run every {self.interval} minutes.")

        # Run once immediately on startup so we don't wait an hour to see if it works
        logger.info("Scheduler: Triggering immediate startup run...")
        job_func()

        # Schedule future runs
        schedule.every(self.interval).minutes.do(job_func)

        # Enter the Infinite Loop
        logger.info("Scheduler: Entering heartbeat loop. Press Ctrl+C to stop.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(1) # Sleep to prevent CPU overuse
        except KeyboardInterrupt:
            logger.info("Scheduler: Stopping...")