import logging
import time
from typing import Dict, List

import psycopg2
from psycopg2.extras import DictCursor
from alpaca.trading.client import TradingClient

from core.datatypes import Candle, MarketData, PortfolioState
from trading_pod.interfaces.IIngestion import IIngestion

logger = logging.getLogger("TRADING.INGESTION")

class DataIngestion(IIngestion):
    """
    Concrete ingestion implementation for the Trading Pod.
    Reads market data from Postgres and portfolio state from Alpaca.
    """

    def __init__(self, db_url: str, alpaca_key: str, alpaca_secret: str, max_retries: int = 5):
        self.db_url = db_url
        self.broker = TradingClient(alpaca_key, alpaca_secret, paper=True)
        self.max_retries = max_retries
        self._conn = None

    def connect(self) -> None:
        """Establishes connection to the internal database with retry logic."""
        if self._conn is not None and not self._conn.closed:
            return

        for attempt in range(self.max_retries):
            try:
                self._conn = psycopg2.connect(self.db_url)
                logger.info("Successfully connected to the internal database.")
                return
            except psycopg2.OperationalError as e:
                logger.warning(f"DB connection failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(3)
                
        raise ConnectionError("Could not connect to database after multiple retries.")

    def fetch_data(self, symbols: List[str], timeframe: str, limit: int) -> MarketData:
        """
        Retrieves the most recent N candles for the requested assets.
        """
        self.connect()
        
        if not symbols or limit <= 0:
            return MarketData(data={})

        table_name = f"candles_{timeframe}"
        grouped_data: Dict[str, List[Candle]] = {s: [] for s in symbols}

        try:
            with self._conn.cursor(cursor_factory=DictCursor) as cursor:
                # Efficiently grab exactly 'limit' rows per symbol using Postgres window functions
                query = f"""
                    SELECT symbol, timestamp, open, high, low, close, volume
                    FROM (
                        SELECT symbol, timestamp, open, high, low, close, volume,
                               ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY timestamp DESC) as rn
                        FROM {table_name}
                        WHERE symbol = ANY(%s)
                    ) sub
                    WHERE rn <= %s
                    ORDER BY symbol, timestamp ASC;
                """
                cursor.execute(query, (symbols, limit))

                for row in cursor.fetchall():
                    symbol = row["symbol"]
                    if symbol in grouped_data:
                        grouped_data[symbol].append(Candle(
                            symbol=symbol,
                            timestamp=row["timestamp"],
                            open=row["open"],
                            high=row["high"],
                            low=row["low"],
                            close=row["close"],
                            volume=row["volume"]
                        ))

            logger.info(f"Loaded {limit} candles for {len(symbols)} symbols from {table_name}.")
            return MarketData(data=grouped_data)

        except Exception as e:
            self._conn.rollback()
            logger.error(f"Failed to fetch market data: {e}")
            raise

    def fetch_portfolio_state(self, market_data: MarketData) -> PortfolioState:
        try:
            account = self.broker.get_account()
            broker_positions = self.broker.get_all_positions()

            positions = {pos.symbol: float(pos.qty) for pos in broker_positions}

            prices = {}
            for symbol in market_data.symbols():
                candles = market_data.data.get(symbol)
                if candles:
                    prices[symbol] = float(candles[-1].close)

            logger.info(f"Portfolio Synced: ${account.equity} Total Equity | Buying Power: ${account.cash}")

            return PortfolioState(
                cash=float(account.cash), 
                positions=positions,
                prices=prices
            )
        except Exception as e:
            logger.error(f"Failed to sync portfolio state: {e}")
            raise