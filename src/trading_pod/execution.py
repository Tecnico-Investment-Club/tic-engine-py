import logging
from typing import List

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
from alpaca.common.exceptions import APIError

from core.datatypes import OrderRequest, TradeReceipt, OrderSide, OrderType
from trading_pod.interfaces.IExecution import IExecution

logger = logging.getLogger("TRADING.EXECUTION")

class AlpacaExecution(IExecution):
    """
    Concrete execution implementation for Alpaca.
    Translates engine OrderRequests into live or paper API calls.
    """
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.broker = TradingClient(api_key, api_secret, paper=paper)
        logger.info(f"Alpaca Execution initialized. Paper mode: {paper}")
    
    def cancel_all_open_orders(self) -> None:
        """Cancels all pending orders to free up buying power."""
        try:
            logger.info("Sweeping pending orders...")
            cancel_statuses = self.broker.cancel_orders()
            
            # Alpaca returns a list of dictionaries with cancelation status
            if cancel_statuses:
                logger.info(f"Successfully sent cancel requests for {len(cancel_statuses)} orders.")
            else:
                logger.info("No open orders to cancel.")
                
        except APIError as e:
            logger.error(f"Failed to cancel open orders. Alpaca API Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error canceling orders: {e}")

    def execute_orders(self, orders: List[OrderRequest]) -> List[TradeReceipt]:
        receipts = []


        if not orders:
            logger.info("No orders to execute in this cycle.")
            return receipts

        for order in orders:
            try:
                # Map the internal OrderSide to Alpaca's format
                alpaca_side = (
                    AlpacaOrderSide.BUY if order.side == OrderSide.BUY 
                    else AlpacaOrderSide.SELL
                )

                # Format symbol for Alpaca (e.g., BTCUSD -> BTC/USD)
                symbol = order.symbol
                is_crypto = False
                
                if symbol.endswith("USDT"):
                    symbol = symbol.replace("USDT", "/USD")
                    is_crypto = True
                elif symbol.endswith("USD") and len(symbol) > 3:
                    symbol = symbol.replace("USD", "/USD")
                    is_crypto = True
                
                # Crypto usually requires GTC. Stocks usually use DAY.
                tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY

                # Currently only Market Orders are supported
                if order.type != OrderType.MARKET:
                    logger.warning(f"Unsupported order type {order.type} for {symbol}. Skipping.")
                    continue

                # Build the Alpaca Request Object
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=order.qty,
                    side=alpaca_side,
                    time_in_force=tif
                )

                logger.info(
                    f"Submitting MARKET {alpaca_side.name} order "
                    f"for {order.qty:.4f} {symbol} [TIF: {tif.name}]..."
                )
                
                # Fire the order to Alpaca
                response = self.broker.submit_order(req)

                # Create the standardized TradeReceipt
                receipt = TradeReceipt(
                    broker_order_id=str(response.id),
                    symbol=str(response.symbol),
                    side=order.side,
                    qty=float(response.qty) if response.qty else 0.0,
                    status=str(response.status).replace("OrderStatus.", ""),
                    filled_price=float(response.filled_avg_price) if response.filled_avg_price else None
                )
                
                receipts.append(receipt)
                logger.info(f"Order {receipt.broker_order_id} submitted successfully. Status: {receipt.status}")

            except Exception as e:
                logger.error(f"Failed to submit order for {order.symbol}. Error: {e}")

        return receipts