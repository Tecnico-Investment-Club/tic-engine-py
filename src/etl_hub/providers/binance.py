import logging
import requests
from datetime import datetime, timezone
from typing import List, Literal

from core.datatypes import Candle
from core.utils import normalize_symbol
from etl_hub.interfaces.IProvider import IProvider

logger = logging.getLogger("ETL.BINANCE")

class BinanceProvider(IProvider):
    """
    Binance Data Source.
    Fetches public klines (OHLCV).
    Does not require API keys.
    """
    BASE_URL = "https://api.binance.com/api/v3/klines"

    def fetch_candles(self, symbol: str, timeframe: Literal["1h", "1d"], limit: int = 100) -> List[Candle]:
        # Map timeframe to match Binance's
        tf = "1h" if timeframe == "1h" else "1d" 

        # Create the request object
        params = {
            "symbol": symbol,
            "interval": tf,
            "limit": limit
        }

        try:
            # Query the broker
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # Safety check
            if not data:
                logger.warning(f"No data returned for {symbol} on Binance")
                return []

            # Normalize exchange symbol to Internal Symbol
            internal_symbol = normalize_symbol(symbol)

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

    def get_provider_name(self) -> str:
        return "Binance"