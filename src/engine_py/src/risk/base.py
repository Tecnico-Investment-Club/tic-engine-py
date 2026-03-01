from abc import ABC, abstractmethod
from typing import List
from src.core.datatypes import OrderRequest

class IRisk(ABC):
    @abstractmethod
    def filter_orders(self, orders: List[OrderRequest]) -> List[OrderRequest]:
        """
        Takes a list of proposed orders and returns a list of orders that
        have been approved by the risk management rules.
        """
        pass