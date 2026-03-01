from abc import ABC, abstractmethod
from typing import List
from src.core.datatypes import OrderRequest

class IExecution(ABC):
    @abstractmethod
    def execute_orders(self, orders: List[OrderRequest]):
        """Takes a list of valid orders and submits them to the broker."""
        pass