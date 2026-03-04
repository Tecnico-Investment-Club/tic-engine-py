from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from core.datatypes import Candle

class IDatabase(ABC):
    """
    Contract for internal database operations.
    """
    
    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def save_candles(self, candles: List[Candle], table_name: str) -> None:
        """Saves a batch of standardized candles."""
        pass

    @abstractmethod
    def get_latest_timestamp(self, symbol: str, table_name: str) -> Optional[datetime]:
        """Returns the most recent candle timestamp for an asset."""
        pass