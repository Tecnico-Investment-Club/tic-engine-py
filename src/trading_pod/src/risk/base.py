from abc import ABC, abstractmethod
from typing import List
from src.core.types import OrderRequest, PortfolioState

class IRiskManager(ABC):
    @abstractmethod
    def validate_orders(self, orders: List[OrderRequest], state: PortfolioState) -> List[OrderRequest]:
        """
        Takes raw orders from the Transformer and filters/modifies them 
        to ensure they comply with hard risk limits.
        """
        pass