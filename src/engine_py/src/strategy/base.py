from abc import ABC, abstractmethod
from typing import List, Dict
from src.core.datatypes import Candle, TargetAllocation

class IStrategy(ABC):
    """
    The strict contract for any trading strategy in the TIC Engine.
    """
    @abstractmethod
    def generate_allocations(self, market_data: Dict[str, List[Candle]]) -> List[TargetAllocation]:
        """
        Takes historical market data and returns target portfolio weights.
        The sum of weights should be <= 1.0 (100%).
        """
        pass