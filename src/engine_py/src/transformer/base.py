from abc import ABC, abstractmethod
from typing import List, Dict
from src.core.datatypes import TargetAllocation, OrderRequest

class ITransformer(ABC):
    @abstractmethod
    def translate_to_orders(
        self,
        allocations: List[TargetAllocation],
        total_equity: float,
        current_prices: Dict[str, float],
        current_positions: Dict[str, float]
    ) -> List[OrderRequest]:
        """
        Takes high-level strategy allocations and system state, and returns
        a list of concrete, executable order requests.
        """
        pass