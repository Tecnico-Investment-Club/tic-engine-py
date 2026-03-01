from abc import ABC, abstractmethod
from typing import List, Literal
from src.core.datatypes import Candle

class DataSource(ABC):
    @abstractmethod
    def fetch_candles(self, symbol: str, timeframe: Literal["1h", "1d"], limit: int = 100) -> List[Candle]:
        """
        Fetches historical candles for a specific symbol.
        Must return a list of standard Pydantic Candle objects.
        """
        pass