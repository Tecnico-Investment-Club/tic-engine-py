import logging
from typing import List

from src.core.types import OrderRequest, TradeReceipt, OrderSide
from .base import IExecutor

logger = logging.getLogger("EXECUTION.MOCK")


class MockExecutor(IExecutor):
    """
    A paper/mock executor that does not send any orders to a real broker.
    Useful for dry-runs, backtests, or CI smoke tests.
    """

    def execute_orders(self, orders: List[OrderRequest]) -> List[TradeReceipt]:
        receipts: List[TradeReceipt] = []

        if not orders:
            logger.info("MockExecutor: No orders to execute in this cycle.")
            return receipts

        logger.info(f"MockExecutor: Simulating execution of {len(orders)} orders.")

        for idx, order in enumerate(orders, start=1):
            side = "BUY" if order.side == OrderSide.BUY else "SELL"
            logger.info(
                f"MockExecutor: {side} {order.qty:.4f} {order.symbol} "
                f"(type={order.type})"
            )

            receipts.append(
                TradeReceipt(
                    broker_order_id=f"mock-{idx}",
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    status="MOCKED",
                    filled_price=None,
                )
            )

        return receipts


