from datetime import datetime, timedelta
from typing import List, Literal
import logging
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.core.datatypes import Candle
from src.etl_db.data_source.base import DataSource

logger = logging.getLogger(__name__)

class AlpacaSource(DataSource):
    def __init__(self, api_key: str, api_secret: str):
        self.client = StockHistoricalDataClient(api_key, api_secret)

    def fetch_candles(self, symbol: str, timeframe: Literal["1h", "1d"], limit: int = 100) -> List[Candle]:
        # Map internal timeframe to Alpaca format
        tf = TimeFrame.Hour if timeframe == "1h" else TimeFrame.Day  # type: ignore

        # Calculate start date. We go back 150 days to guarantee we find 100 active trading days.
        start_date = datetime.now() - timedelta(days=150)

        # Build the request with the start_date included
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol], 
            timeframe=tf, 
            start=start_date,
            limit=limit
        )

        try:
            # Fetch Data
            bars = self.client.get_stock_bars(request_params)
            
            # Safely cast Alpaca's BarSet object to a standard Python dictionary
            bars_dict = dict(bars)
            
            # THE FIX: Extract the actual 'data' dictionary inside it
            actual_data = bars_dict.get("data", {})

            # DEBUG PRINT: Let's see what we actually got!
            print(f"DEBUG: Keys inside actual_data for {symbol}:", actual_data.keys())
            
            # Check if we got data for this symbol
            if symbol not in actual_data or not actual_data[symbol]:
                logger.warning(f"No data returned for {symbol}")
                return []

            # Convert Alpaca Objects into Internal Pydantic Candles
            candles = []
            for bar in actual_data[symbol]:
                candles.append(Candle(
                    symbol=symbol,
                    timestamp=bar.timestamp,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume
                ))
            
            return candles

        except Exception as e:
            # There was an exception fail-fast and log the error
            logger.error(f"Alpaca API Error for {symbol}: {e}")
            return []