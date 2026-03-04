import logging
import time
from typing import Dict, List

import psycopg2
from alpaca.trading.client import TradingClient
from psycopg2.extras import DictCursor

from src.core.types import Candle, MarketData, PortfolioState
from src.core.utils import normalize_symbol, timeframe_to_table
from .base import IIngestion

logger = logging.getLogger(__name__)

# Retry settings for initial DB connection
_MAX_CONNECT_RETRIES = 5
_CONNECT_RETRY_DELAY = 3  # seconds


class StandardIngestor(IIngestion):
    """
    Concrete ingestion implementation used by the trading engine.

    Responsibilities:
    - Read-only access to the internal candles tables for a given timeframe.
    - Query the broker for live account state and positions.

    Maintains a persistent DB connection and reconnects transparently on failure.
    """

    def __init__(self, db_url: str, alpaca_key: str, alpaca_secret: str):
        self.db_url = db_url
        self.broker = TradingClient(alpaca_key, alpaca_secret, paper=True)
        self._conn = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_connection(self):
        """Return the persistent connection, reconnecting if necessary."""
        if self._conn is not None and not self._conn.closed:
            return self._conn

        for attempt in range(_MAX_CONNECT_RETRIES):
            try:
                self._conn = psycopg2.connect(self.db_url)
                return self._conn
            except psycopg2.OperationalError:
                if attempt < _MAX_CONNECT_RETRIES - 1:
                    logger.warning(
                        f"Database not ready. Retrying in {_CONNECT_RETRY_DELAY}s... "
                        f"({attempt + 1}/{_MAX_CONNECT_RETRIES})"
                    )
                    time.sleep(_CONNECT_RETRY_DELAY)
                else:
                    logger.error("Could not connect to database after multiple retries.")
                    raise

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_data(self, symbols: List[str], limit: int, timeframe: str) -> MarketData:
        """
        Fetches the last ``limit`` candles for each requested symbol from the
        correct timeframe table.  Uses a persistent connection and reconnects
        transparently on failure.
        """
        if not symbols or limit <= 0:
            return MarketData()

        table_name = timeframe_to_table(timeframe)
        grouped_data: Dict[str, List[Candle]] = {s: [] for s in symbols}

        conn = self._get_connection()

        try:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                query = f"""
                    SELECT symbol, timestamp, open, high, low, close, volume
                    FROM (
                        SELECT symbol,
                               timestamp,
                               open,
                               high,
                               low,
                               close,
                               volume,
                               ROW_NUMBER() OVER(
                                   PARTITION BY symbol
                                   ORDER BY timestamp DESC
                               ) as rn
                        FROM {table_name}
                        WHERE symbol = ANY(%s)
                    ) sub
                    WHERE rn <= %s
                    ORDER BY symbol, timestamp ASC;
                """
                cursor.execute(query, (symbols, limit))

                for row in cursor.fetchall():
                    symbol = row["symbol"]
                    grouped_data[symbol].append(Candle(**row))

            logger.info(
                f"Loaded market data for {len(symbols)} symbols from {table_name} "
                f"(limit={limit})."
            )
            return MarketData(data=grouped_data)

        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            # Connection dropped — reset and let the next call reconnect
            logger.warning(f"DB connection lost during query, will reconnect: {e}")
            self._conn = None
            raise
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            raise

    def get_live_portfolio_state(self, market_data: MarketData) -> PortfolioState:
        """
        Fetches cash + positions from the broker and maps latest prices
        from market data.
        """
        try:
            account = self.broker.get_account()
            broker_positions = self.broker.get_all_positions()

            positions: Dict[str, float] = {}
            for pos in broker_positions:
                symbol = normalize_symbol(pos.symbol)
                positions[symbol] = float(pos.qty)

            logger.debug(f"Normalized broker positions: {positions}")

            prices = {}
            for symbol, candles in market_data.data.items():
                logger.debug(f"Symbol {symbol} has {len(candles)} candles.")
                if candles:
                    prices[symbol] = float(candles[-1].close)
                else:
                    logger.warning(f"No candles found in market_data for {symbol}!")

            logger.debug(f"Final mapped prices dictionary: {prices}")

            return PortfolioState(
                cash=float(account.cash),
                positions=positions,
                prices=prices,
            )
        except Exception as e:
            logger.error(f"Failed to create portfolio state: {e}")
            raise
