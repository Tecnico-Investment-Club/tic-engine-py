from abc import ABC, abstractmethod
from typing import List, Dict
from src.core.datatypes import Candle

class IIngestion(ABC):
    @abstractmethod
    def get_portfolio_cash(self) -> float:
        """Returns the current available cash balance from the broker."""
        pass

    @abstractmethod
    def get_current_positions(self) -> Dict[str, float]:
        """Returns a dict of {'symbol': quantity} for all current holdings."""
        pass

    @abstractmethod
    def get_historical_data(self, symbols: List[str], limit: int) -> Dict[str, List[Candle]]:
        """Returns historical OHLCV data from the internal DB."""
        pass