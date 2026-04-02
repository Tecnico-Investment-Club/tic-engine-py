from abc import ABC, abstractmethod
from typing import List, Dict
from core.datatypes import Candle 

class IProvider(ABC):
    """
    Contract for all external market data sources.
    Forces all broker APIs to return standardized data structures.
    """
    
    @abstractmethod
    def fetch_candles(self, symbols: List[str], timeframe: str, limit: int = 100) -> Dict[str, List[Candle]]:
        """
        Fetches historical OHLCV data for the given symbols.
        
        :param symbols: List of ticker strings
        :param timeframe: The interval
        :param limit: Number of candles to fetch per asset
        :return: A dictionary mapping each symbol to a list of validated Candle objects
        """
        pass
        
    @abstractmethod
    def get_provider_name(self) -> str:
        """Returns the name of the provider (e.g., 'Alpaca', 'Binance') for logging."""
        pass