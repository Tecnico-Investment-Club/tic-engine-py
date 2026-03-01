from abc import ABC, abstractmethod
from typing import Callable

class IScheduler(ABC):
    @abstractmethod
    def start(self, job_func: Callable):
        """
        Starts the blocking scheduler loop, which will execute the provided
        job function at the configured interval.
        """
        pass