from abc import ABC, abstractmethod
from typing import List
from core.datatypes import MarketData, PortfolioState

class IIngestion(ABC):
    """
    Contract for reading market data from the internal ETL database.
    """
    
    @abstractmethod
    def connect(self) -> None:
        """Establishes connection to the internal database."""
        pass

    @abstractmethod
    def fetch_data(self, symbols: List[str], timeframe: str, limit: int) -> MarketData:
        """
        Retrieves the most recent N candles for the requested assets.
        
        :param symbols: List of ticker strings (e.g., ["AAPL", "BTCUSD"])
        :param timeframe: Interval (e.g., "1h", "1d")
        :param limit: Number of candles to pull from the DB
        :return: A populated MarketData object containing the history
        """
        pass

    @abstractmethod
    def fetch_portfolio_state(self) -> PortfolioState:
        """
        Queries the broker (or a mock/paper state) to get the current cash, 
        open positions, and latest prices.
        """
        pass