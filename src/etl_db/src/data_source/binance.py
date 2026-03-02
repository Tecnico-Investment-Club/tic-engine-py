import logging
import requests
from datetime import datetime, timezone
from typing import List, Literal

from src.core.types import Candle
from src.etl_db.src.data_source.base import DataSource

logger = logging.getLogger(__name__)

class BinanceSource(DataSource):
    BASE_URL = "https://api.binance.com/api/v3/klines"

    def fetch_candles(self, symbol: str, timeframe: Literal["1h", "1d"], limit: int = 100) -> List[Candle]:
        
        # Map timeframe to Binance format
        tf = "1h" if timeframe == "1h" else "1d"

        # Build Request
        params = {"symbol": symbol, "interval": tf, "limit": limit}

        try:
            # Fetch Data
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                logger.warning(f"No data returned for {symbol} on Binance")
                return []

            # Convert Binance Arrays into Internal Pydantic Candles
            candles = []
            for row in data:
                # Binance returns a list of lists.
                # [Open time (ms), Open, High, Low, Close, Volume, Close time, ...]
                timestamp_ms = row[0]
                
                # Convert milliseconds to UTC datetime
                dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)

                # Translate symbol into internal logic
                stablecoins = ["USDT", "BUSD", "TUSD", "USDC"]
                internal_symbol = symbol
                for coin in stablecoins:
                    if symbol.endswith(coin):
                        internal_symbol = symbol[:-len(coin)] + "USD"
                        break


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