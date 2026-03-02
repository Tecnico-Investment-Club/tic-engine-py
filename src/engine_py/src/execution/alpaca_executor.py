import logging
from typing import List

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce

from src.core.types import OrderRequest, TradeReceipt, OrderSide, OrderType
from .base import IExecutor

logger = logging.getLogger(__name__)

class AlpacaExecutor(IExecutor):
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        # Initialize the Alpaca Trading Client
        self.broker = TradingClient(api_key, api_secret, paper=paper)
        logger.info(f"AlpacaExecutor initialized. Paper mode: {paper}")

    def execute_orders(self, orders: List[OrderRequest]) -> List[TradeReceipt]:
        # Initialize an empty list to hold trade receipts for all executed orders
        receipts = []

        # Sanity check: If there are no orders, skip execution and return empty receipts list
        if not orders:
            logger.info("Executor: No orders to execute in this cycle.")
            return receipts

        # Loop through each order and attempt to execute it
        for order in orders:
            try:
                # Map the internal OrderSide to Alpaca's format
                alpaca_side = (AlpacaOrderSide.BUY if order.side == OrderSide.BUY 
                    else AlpacaOrderSide.SELL
                )

                # Prepare the symbol for the order
                exec_symbol = order.symbol
                if exec_symbol.endswith("USD") and len(exec_symbol) > 3:
                    exec_symbol = exec_symbol.replace("USD", "/USD")

                # Build the Alpaca Request Object based on Order Type
                if order.type == OrderType.MARKET:
                    req = MarketOrderRequest(
                        symbol=exec_symbol,
                        qty=order.qty,
                        side=alpaca_side,
                        time_in_force=TimeInForce.IOC
                    )
                elif order.type == OrderType.LIMIT:
                    # Not yet supported in this implementation
                    logger.warning(f"Executor: LIMIT orders are not yet supported. Skipping order for {order.symbol}.")
                else:
                    logger.error(f"Executor: Unsupported order type {order.type} for {order.symbol}.")
                    continue

                # Fire the order to Alpaca
                logger.info(f"Executor: Submitting {order.type.upper()} {order.side.upper()} order for {order.qty:.4f} {order.symbol}...")
                response = self.broker.submit_order(req)


                # Create the TradeReceipt from the broker's response
                receipt = TradeReceipt(
                    broker_order_id=str(response.id),
                    symbol=response.symbol,
                    side=order.side,
                    qty=float(response.qty),
                    status=str(response.status),
                    filled_price=float(response.filled_avg_price) if response.filled_avg_price else None
                )
                
                receipts.append(receipt)
                logger.info(f"Executor: Order {receipt.broker_order_id} submitted successfully. Status: {receipt.status}")

            except Exception as e:
                logger.error(f"Executor: Failed to submit order for {order.symbol}. Error: {e}")

        return receipts