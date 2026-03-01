import logging
from typing import List
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from src.core.datatypes import OrderRequest
from src.engine_py.src.execution.base import IExecution

logger = logging.getLogger(__name__)

class AlpacaExecution(IExecution):
    """
    Execution.
    The only module allowed to send write commands (Orders) to the Broker.
    """
    def __init__(self, alpaca_key: str, alpaca_secret: str):
        self.client = TradingClient(alpaca_key, alpaca_secret, paper=True)

    def execute_orders(self, orders: List[OrderRequest]):
        """
        Iterates through the list of valid orders and submits them one by one.
        """
        if not orders:
            logger.info("Execution: No orders to submit.")
            return

        logger.info(f"Execution: Submitting {len(orders)} orders to Alpaca...")

        for order in orders:
            try:
                # Map internal "buy"/"sell" string to Alpaca Enum
                side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL

                # Alpaca uses "/USD" format for crypto pairs, so we need to convert
                if order.symbol.endswith("USDT"):
                    order.symbol = order.symbol.replace("USDT", "/USD")
                
                
                # Construct the Alpaca Request Object
                req = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=TimeInForce.GTC
                )

                # Send the order to the client
                submitted_order = self.client.submit_order(order_data=req)
                
                logger.info(
                    f"Execution: SUCCESS Order submitted for {order.symbol}. "
                    f"Execution: ID: {submitted_order.id} | Status: {submitted_order.status}"
                )

            except Exception as e:
                logger.error(f"Execution: FAILED to submit order for {order.symbol}: {e}")
                continue