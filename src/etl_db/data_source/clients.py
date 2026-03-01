"""
API clients for fetching market data from Binance and Alpaca.
Handles rate limiting, retries, and data format conversion.
"""
from datetime import datetime, timedelta
from typing import List, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError

from src.core.datatypes import Candle
from src.core.config import settings

logger = logging.getLogger(__name__)


class BinanceClient:
    """Client for fetching crypto data from Binance public API."""
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    @staticmethod
    def _convert_ticker(ticker: str) -> str:
        """
        Convert ticker format from "BTC/USD" to Binance format "BTCUSDT".
        Assumes USD pairs map to USDT on Binance.
        """
        if "/" in ticker:
            base, quote = ticker.split("/")
            if quote.upper() == "USD":
                return f"{base.upper()}USDT"
            return f"{base.upper()}{quote.upper()}"
        return ticker.upper()
    
    @staticmethod
    def _convert_binance_to_candle(ticker: str, kline_data: List) -> Candle:
        """Convert Binance kline response to Candle Pydantic model."""
        return Candle(
            ticker=ticker,
            timestamp=datetime.fromtimestamp(kline_data[0] / 1000),
            open=float(kline_data[1]),
            high=float(kline_data[2]),
            low=float(kline_data[3]),
            close=float(kline_data[4]),
            volume=float(kline_data[5])
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException))
    )
    def fetch_candles(
        self,
        ticker: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Candle]:
        """
        Fetch candles from Binance API.
        
        Args:
            ticker: Ticker symbol (e.g., "BTC/USD")
            interval: Time interval ("1h" for hourly, "1d" for daily)
            start_time: Start time for historical data
            end_time: End time for historical data
            limit: Maximum number of candles to fetch (max 1000)
        
        Returns:
            List of Candle objects
        """
        binance_symbol = self._convert_ticker(ticker)
        binance_interval = interval  # "1h" or "1d" work directly with Binance
        
        params = {
            "symbol": binance_symbol,
            "interval": binance_interval,
            "limit": min(limit, 1000)
        }
        
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(f"{self.BASE_URL}/klines", params=params)
                response.raise_for_status()
                data = response.json()
                
                candles = [
                    self._convert_binance_to_candle(ticker, kline)
                    for kline in data
                ]
                
                logger.info(f"Fetched {len(candles)} {interval} candles from Binance for {ticker}")
                return candles
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching Binance data for {ticker}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching Binance data for {ticker}: {e}")
            raise


class AlpacaClient:
    """Client for fetching stock data from Alpaca API."""
    
    def __init__(self):
        """Initialize Alpaca clients with API credentials."""
        self.stock_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_api_secret,
            paper=settings.env == "testnet"
        )
        self.crypto_client = CryptoHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_api_secret,
            paper=settings.env == "testnet"
        )
    
    @staticmethod
    def _convert_alpaca_to_candle(ticker: str, bar) -> Candle:
        """Convert Alpaca bar object to Candle Pydantic model."""
        return Candle(
            ticker=ticker,
            timestamp=bar.timestamp,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume)
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(APIError)
    )
    def fetch_candles(
        self,
        ticker: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Candle]:
        """
        Fetch candles from Alpaca API.
        
        Args:
            ticker: Ticker symbol (e.g., "AAPL")
            timeframe: Time frame ("1h" for hourly, "1d" for daily)
            start_time: Start time for historical data
            end_time: End time for historical data
            limit: Maximum number of candles to fetch
        
        Returns:
            List of Candle objects
        """
        # Determine if we should use stock or crypto client
        # For now, assume stocks unless ticker contains "/"
        is_crypto = "/" in ticker
        
        # Convert timeframe string to Alpaca TimeFrame
        if timeframe == "1h":
            alpaca_timeframe = TimeFrame.Hour
        elif timeframe == "1d":
            alpaca_timeframe = TimeFrame.Day
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        try:
            if is_crypto:
                # Use crypto client
                request = CryptoBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=alpaca_timeframe,
                    start=start_time,
                    end=end_time,
                    limit=limit
                )
                bars = self.crypto_client.get_crypto_bars(request)
            else:
                # Use stock client
                request = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=alpaca_timeframe,
                    start=start_time,
                    end=end_time,
                    limit=limit
                )
                bars = self.stock_client.get_stock_bars(request)
            
            # Extract bars for the ticker
            # Alpaca returns a BarSet (dictionary-like) with ticker as key
            ticker_bars = []
            if hasattr(bars, 'get'):
                # Dictionary-like object (BarSet)
                ticker_bars = bars.get(ticker, [])
            elif isinstance(bars, dict):
                ticker_bars = bars.get(ticker, [])
            elif hasattr(bars, '__iter__'):
                # Iterable (list or similar)
                ticker_bars = list(bars)
            
            candles = [
                self._convert_alpaca_to_candle(ticker, bar)
                for bar in ticker_bars
            ]
            
            logger.info(f"Fetched {len(candles)} {timeframe} candles from Alpaca for {ticker}")
            return candles
        except APIError as e:
            logger.error(f"Alpaca API error fetching data for {ticker}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching Alpaca data for {ticker}: {e}")
            raise

