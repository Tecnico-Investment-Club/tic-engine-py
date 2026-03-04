import logging
import requests
from datetime import datetime, timezone
from typing import List, Literal

from src.core.types import Candle
from src.core.utils import normalize_symbol
from src.etl_db.src.data_source.base import DataSource

logger = logging.getLogger("ETL.BINANCE")

class BinanceSource(DataSource):
    """
    Binance Data Source.
    Fetches public klines (OHLCV) data without requiring API keys.
    """
    BASE_URL = "https://api.binance.com/api/v3/klines"

    def fetch_candles(
        self, 
        symbol: str, 
        timeframe: Literal["1h", "1d"], 
        limit: int = 100
    ) -> List[Candle]:
        """
        Fetches the most recent candles from Binance.
        """
        # Map timeframe (Binance uses 1h, 1d)
        tf = "1h" if timeframe == "1h" else "1d"

        # Construct Parameters
        # The intent is to return the "limit" most recent candles.
        # With limit being a number
        params = {
            "symbol": symbol,
            "interval": tf,
            "limit": limit
        }

        try:
            # Fetch Data
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                logger.warning(f"No data returned for {symbol} on Binance")
                return []

            # Normalize exchange symbol to Internal Symbol
            internal_symbol = normalize_symbol(symbol)

            # Convert to Internal Pydantic Candles
            candles = []
            for row in data:
                # Binance row format: [Open time, Open, High, Low, Close, Volume, ...]
                timestamp_ms = row[0]
                dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)

                candles.append(Candle(
                    symbol=internal_symbol,
                    timestamp=dt,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5])
                ))
            
            return candles

        except Exception as e:
            logger.error(f"Binance API Error for {symbol}: {e}")
            return []