from abc import ABC, abstractmethod
from typing import List
from core.datatypes import TargetAllocation, PortfolioState, OrderRequest

class ITransformer(ABC):
    """
    Contract for risk management and order sizing.
    Translates abstract percentages into concrete share quantities.
    """

    @abstractmethod
    def generate_orders(self, allocations: List[TargetAllocation], state: PortfolioState) -> List[OrderRequest]:
        """
        Calculates the exact buy/sell quantities needed to transition 
        the current PortfolioState into the TargetAllocations.
        """
        pass