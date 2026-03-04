from abc import ABC, abstractmethod
from typing import List
from src.core.types import MarketData, PortfolioState

class IIngestion(ABC):
    @abstractmethod
    def get_data(self, symbols: List[str], limit: int, timeframe: str) -> MarketData:
        """
        Query the internal DB for candles and return them
        grouped by symbol for O(1) strategy access.
        """
        pass

    @abstractmethod
    def get_live_portfolio_state(self, market_data: MarketData) -> PortfolioState:
        """Query the Broker for current cash and open positions."""
        pass
