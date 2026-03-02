import logging
import psycopg2
import time
from psycopg2.extras import DictCursor
from typing import List, Dict
from alpaca.trading.client import TradingClient

from src.core.types import Candle, MarketData, PortfolioState # Ensure types are consistent
from .base import IIngestion

logger = logging.getLogger(__name__)

class StandardIngestor(IIngestion):
    def __init__(self, db_url: str, alpaca_key: str, alpaca_secret: str):
        self.db_url = db_url
        self.broker = TradingClient(alpaca_key, alpaca_secret, paper=True)

    

    def get_data(self, symbols: List[str], lookback_candles: int) -> MarketData:
        """
        Fetches the last N candles with a retry loop for DB availability.
        """
        if not symbols:
            return MarketData(data={})

        grouped_data: Dict[str, List[Candle]] = {s: [] for s in symbols}
        
        # RETRY CONFIG
        max_retries = 5
        retry_delay = 3 # seconds
        
        conn = None
        for attempt in range(max_retries):
            try:
                conn = psycopg2.connect(self.db_url)
                # If we reach here, connection successful
                break 
            except psycopg2.OperationalError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Database not ready. Retrying in {retry_delay}s... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    logger.error("Could not connect to database after multiple retries.")
                    raise e

        try:
            with conn:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    query = """
                        SELECT symbol, timestamp, open, high, low, close, volume 
                        FROM (
                            SELECT symbol, timestamp, open, high, low, close, volume,
                                ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY timestamp DESC) as rn
                            FROM candles_1h 
                            WHERE symbol = ANY(%s)
                        ) sub
                        WHERE rn <= %s
                        ORDER BY symbol, timestamp ASC;
                    """
                    cursor.execute(query, (symbols, lookback_candles))
                    
                    for row in cursor.fetchall():
                        symbol = row['symbol']
                        grouped_data[symbol].append(Candle(**row))
            
            logger.info(f"Loaded market data for {len(symbols)} symbols")
            return MarketData(data=grouped_data)

        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            raise e
        finally:
            if conn:
                conn.close()

    def get_live_portfolio_state(self, market_data: MarketData) -> PortfolioState:
        """
        Fetches cash+positions and maps latest prices.
        """
        try:
            # Fetch from Broker
            account = self.broker.get_account()
            broker_positions = self.broker.get_all_positions()

            # Normalize and Map Positions
            positions = {}
            for pos in broker_positions:
                symbol = self._normalize_symbol(pos.symbol)
                positions[symbol] = float(pos.qty)

            logger.info(f"[DEBUG INGESTOR] Normalized broker positions: {positions}")

            # Map the latest close prices from the market data
            prices = {}
            logger.info(f"[DEBUG INGESTOR] Market data symbols received: {list(market_data.data.keys())}")
            
            for symbol, candles in market_data.data.items():
                logger.info(f"[DEBUG INGESTOR] Symbol {symbol} has {len(candles)} candles.")
                if candles:
                    # Explicitly cast to float just in case the DB returned a Decimal type
                    prices[symbol] = float(candles[-1].close) 
                else:
                    logger.warning(f"[DEBUG INGESTOR] No candles found in market_data for {symbol}!")

            logger.info(f"[DEBUG INGESTOR] Final mapped prices dictionary: {prices}")

            # Return the PortfolioState object
            return PortfolioState(
                cash=float(account.cash),
                positions=positions,
                prices=prices
            )
        except Exception as e:
            logger.error(f"Failed to create portfolio state: {e}")
            raise e

    def _normalize_symbol(self, symbol: str) -> str:
        """Internal helper for Alpaca crypto/stock naming quirks."""
        if "/" in symbol:
            return symbol.replace("/", "") # e.g., BTC/USD -> BTCUSD
        return symbol