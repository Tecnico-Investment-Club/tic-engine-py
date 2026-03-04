from abc import ABC, abstractmethod
from typing import List
from src.core.types import OrderRequest, TradeReceipt

class IExecutor(ABC):
    @abstractmethod
    def execute_orders(self, orders: List[OrderRequest]) -> List[TradeReceipt]:
        """
        Submits live orders to the broker and returns the execution receipts.
        """
        pass