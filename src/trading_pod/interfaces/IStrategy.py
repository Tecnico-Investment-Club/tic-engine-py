from abc import ABC, abstractmethod
from typing import List
from core.datatypes import MarketData, TargetAllocation

class IStrategy(ABC):
    """
    Contract for all trading algorithms. 
    Outputs target portfolio weights (0.0 to 1.0).
    """

    @abstractmethod
    def generate_allocations(self, data: MarketData) -> List[TargetAllocation]:
        """
        Evaluates the market data and determines target portfolio weights.
        
        :param data: The historical MarketData dictionary from IIngestion
        :return: A list of TargetAllocation objects (e.g., 50% AAPL, 50% Cash)
        """
        pass