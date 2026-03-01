import logging
import psycopg2
from psycopg2.extras import DictCursor
from typing import List, Dict
from alpaca.trading.client import TradingClient

from src.core.datatypes import Candle
from src.engine_py.src.ingestion.base import IIngestion

logger = logging.getLogger(__name__)

class AlpacaIngestion(IIngestion):
    def __init__(self, db_url: str, alpaca_key: str, alpaca_secret: str):
        self.db_url = db_url
        self.broker = TradingClient(alpaca_key, alpaca_secret, paper=True)

    def get_portfolio_cash(self) -> float:
        try:
            account = self.broker.get_account()
            cash_balance = float(account.cash) 
            logger.info(f"Ingestion: Live Broker Cash Balance: ${cash_balance:.2f}")
            return cash_balance
        except Exception as e:
            logger.error(f"Ingestion: Failed to fetch broker cash balance: {e}")
            raise e

    def get_current_positions(self) -> Dict[str, float]:
        # Initialize an empty dictionary to store positions
        positions = {}
        try:
            # Get all positions from the broker
            broker_positions = self.broker.get_all_positions()

            # For each position, extract the symbol and quantity, and store it in the positions dictionary
            for pos in broker_positions:
                # Normalize cryptocurrency symbols to alpaca format
                symbol = pos.symbol
                if "/" in symbol:
                    symbol = symbol.replace("/", "").replace("USD", "USDT")

                elif symbol.endswith("USD") and len(symbol) > 3:
                    symbol = symbol.replace("USD", "USDT")
                
                positions[symbol] = float(pos.qty)
            logger.info(f"Ingestion: Found {len(positions)} existing positions.")
        except Exception as e:
            # Log the error and re-raise it
            logger.error(f"Ingestion: Failed to fetch broker positions: {e}")
            raise e
        return positions

    def get_historical_data(self, symbols: List[str], limit: int = 100) -> Dict[str, List[Candle]]:
        market_data: Dict[str, List[Candle]] = {}
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    for symbol in symbols:
                        query = """
                            SELECT symbol, timestamp, open, high, low, close, volume 
                            FROM candles_1h WHERE symbol = %s 
                            ORDER BY timestamp DESC LIMIT %s;
                        """
                        cursor.execute(query, (symbol, limit))
                        rows = cursor.fetchall()
                        rows.reverse()
                        market_data[symbol] = [Candle(**row) for row in rows]
                        logger.info(f"Loaded {len(market_data[symbol])} historical candles for {symbol}")
        except Exception as e:
            logger.error(f"Failed to fetch market data from internal DB: {e}")
            raise e
        return market_data