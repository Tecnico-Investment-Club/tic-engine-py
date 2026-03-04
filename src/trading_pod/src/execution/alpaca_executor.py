import logging
from typing import List

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce

from src.core.types import OrderRequest, TradeReceipt, OrderSide, OrderType
from .base import IExecutor

logger = logging.getLogger("EXECUTION.ALPACA")

class AlpacaExecutor(IExecutor):
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.broker = TradingClient(api_key, api_secret, paper=paper)
        logger.info(f"AlpacaExecutor initialized. Paper mode: {paper}")

    def execute_orders(self, orders: List[OrderRequest]) -> List[TradeReceipt]:
        receipts = []

        # Sanity check
        if not orders:
            logger.info("Executor: No orders to execute in this cycle.")
            return receipts

        # For every order in the list
        for order in orders:
            try:
                # Map the internal OrderSide to Alpaca's format
                alpaca_side = (
                    AlpacaOrderSide.BUY if order.side == OrderSide.BUY 
                    else AlpacaOrderSide.SELL
                )

                # Prepare the symbol and TimeInForce
                symbol = order.symbol
                is_crypto = False
                
                # Handle the crypto pair
                if symbol.endswith("USDT"):
                    symbol = symbol.replace("USDT", "/USD")
                    is_crypto = True
                elif symbol.endswith("USD") and len(symbol) > 3:
                    symbol = symbol.replace("USD", "/USD")
                    is_crypto = True
                
                # Crypto must be GTC. Fractional Stocks must be DAY.
                tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY

                # Build the Alpaca Request Object
                if order.type == OrderType.MARKET:
                    req = MarketOrderRequest(
                        symbol=symbol,
                        qty=order.qty,
                        side=alpaca_side,
                        time_in_force=tif
                    )
                elif order.type == OrderType.LIMIT:
                    logger.warning(f"Executor: LIMIT orders are not yet supported. Skipping {symbol}.")
                    continue
                else:
                    logger.error(f"Executor: Unsupported order type {order.type} for {symbol}.")
                    continue

                # Fire the order to Alpaca
                logger.info(
                    f"Executor: Submitting MARKET {alpaca_side.name} order "
                    f"for {order.qty:.4f} {symbol} [TIF: {tif.name}]..."
                )
                
                response = self.broker.submit_order(req)

                # Create the TradeReceipt
                receipt = TradeReceipt(
                    broker_order_id=str(response.id),
                    symbol=str(response.symbol),
                    side=order.side,
                    qty=float(response.qty) if response.qty else 0.0,
                    status=str(response.status).replace("OrderStatus.", ""),
                    filled_price=float(response.filled_avg_price) if response.filled_avg_price else None
                )
                
                receipts.append(receipt)
                logger.info(f"Executor: Order {receipt.broker_order_id} submitted successfully. Status: {receipt.status}")

            except Exception as e:
                logger.error(f"Executor: Failed to submit order for {order.symbol}. Error: {e}")

        return receipts