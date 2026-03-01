from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel

class Candle(BaseModel):
    """
    Standard OHLCV data point.
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class TargetAllocation(BaseModel):
    """
    Strategy output: How much of the portfolio should be in this asset?
    """
    symbol: str
    weight: float  # Expected range: 0.0 to 1.0

class OrderRequest(BaseModel):
    """
    Execution instruction: Buy/Sell specific quantity.
    """
    symbol: str
    qty: float
    side: Literal["buy", "sell"]
    type: Literal["market", "limit"] = "market"
    limit_price: Optional[float] = None