import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import List, Literal

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from core.datatypes import Candle
from etl_hub.interfaces.IProvider import IProvider

logger = logging.getLogger("ETL.ALPACA")

class AlpacaProvider(IProvider):
    """
    Alpaca Data Source.
    Fetches public candles (OHLCV).
    """
    def __init__(self, api_key: str, api_secret: str):
        self.client = StockHistoricalDataClient(api_key, api_secret)
    
    def fetch_candles(self, symbol: str, timeframe: Literal["1h", "1d"], limit: int = 500) -> List[Candle]:
        tf = TimeFrame.Hour if timeframe == "1h" else TimeFrame.Day

        buffer_minutes = 16 
        delayed_now = datetime.now(timezone.utc) - timedelta(minutes=buffer_minutes)
        
        # Calculate a wide enough start date to ensure we get enough trading days
        if timeframe == "1d":
            days_back = int(limit * 1.5) + 10  
        else:
            days_back = int((limit / 6.5) * 1.5) + 5 
            
        start_date = delayed_now - timedelta(days=days_back)

        # We want it to fetch ALL candles from the start_date up to right now.
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol, 
            timeframe=tf, 
            start=start_date,
            end=delayed_now
        )

        try:
            bars = self.client.get_stock_bars(request_params)
            
            if not bars or symbol not in bars.data:
                logger.warning(f"No stock data returned for {symbol}")
                return []

            # Map the full dataset to our Pydantic objects
            all_candles = [Candle(
                symbol=symbol,
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume
            ) for bar in bars.data[symbol]]
            
            # Slice exactly the amount we need from the very end
            # the LIMIT most recent ones
            # TODO: CHANGE THE STOCK BROKER SINCE ALPACA-SDK IS TERRIBLE AND DOES NOT RETURN onlyRTH hours.
            return all_candles[-limit:]
            
        except Exception as e:
            logger.error(f"Alpaca API Error for {symbol}: {e}")
            return []
            
    def get_provider_name(self) -> str:
        return "Alpaca"