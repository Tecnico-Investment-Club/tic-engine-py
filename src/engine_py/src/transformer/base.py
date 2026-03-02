from abc import ABC, abstractmethod
from typing import List
from src.core.types import TargetAllocation, PortfolioState, OrderRequest

class ITransformer(ABC):
    @abstractmethod
    def transform(self, allocations: List[TargetAllocation], state: PortfolioState) -> List[OrderRequest]:
        """
        Converts target percentage weights into absolute broker orders (Buy/Sell X units).
        """
        pass