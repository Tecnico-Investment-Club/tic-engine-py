from abc import ABC, abstractmethod
from typing import List
from src.core.types import MarketData, PortfolioState, TargetAllocation

class IStrategy(ABC):
    @abstractmethod
    def compute_weights(self, market_data: MarketData, state: PortfolioState) -> List[TargetAllocation]:
        """
        Input: 
            market_data: Historical OHLCV dict for technical analysis.
            state: Current cash, positions, and synced prices.
        Output: 
            List of TargetAllocation (symbol + % weight).
        """
        pass