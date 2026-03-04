from datetime import datetime, timedelta, timezone
from typing import List, Literal
import logging
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from src.core.types import Candle
from src.etl_db.src.data_source.base import DataSource

logger = logging.getLogger("ETL.ALPACA")

class AlpacaSource(DataSource):
    def __init__(self, api_key: str, api_secret: str):
        self.client = StockHistoricalDataClient(api_key, api_secret)
    
    def fetch_candles(self, symbol: str, timeframe: Literal["1h", "1d"], limit: int = 500) -> List[Candle]:
        tf = TimeFrame.Hour if timeframe == "1h" else TimeFrame.Day

        buffer_minutes = 16 
        delayed_now = datetime.now(timezone.utc) - timedelta(minutes=buffer_minutes)
        
        days_back = int(limit * 2.5) if timeframe == "1d" else int(limit / 3)
        start_date = delayed_now - timedelta(days=days_back)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol], 
            timeframe=tf, 
            start=start_date,
            end=delayed_now
            # Removed 'limit' from here so it doesn't cut off early
        )

        try:
            bars = self.client.get_stock_bars(request_params)
            
            if not bars or symbol not in bars.data:
                logger.warning(f"No stock data returned for {symbol}")
                return []

            # Grab exactly the most recent `limit` amount of candles
            recent_bars = bars.data[symbol][-limit:]

            return [Candle(
                symbol=symbol,
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume
            ) for bar in recent_bars]
        except Exception as e:
            logger.error(f"Alpaca API Error for {symbol}: {e}")
            return []