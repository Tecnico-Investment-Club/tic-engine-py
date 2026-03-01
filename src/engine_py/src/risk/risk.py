import logging
from typing import List

from .base import IRisk # Import the interface
from src.core.datatypes import OrderRequest

logger = logging.getLogger(__name__)

# Rename class and inherit from IRisk
class RiskManager(IRisk):
    """
    A simple placeholder risk manager that performs basic sanity checks.
    """
    def __init__(self, max_qty_per_order: float = 1000.0):
        self.max_qty_per_order = max_qty_per_order

    def filter_orders(self, orders: List[OrderRequest]) -> List[OrderRequest]:
        approved_orders = []
        logger.info("--- Risk Manager reviewing orders ---")
        for order in orders:
            if order.qty > self.max_qty_per_order:
                logger.warning(
                    f"RISK ALERT: {order.side.upper()} {order.qty} {order.symbol} exceeds "
                    f"max limit of {self.max_qty_per_order}. Truncating order to max limit."
                )
                order.qty = self.max_qty_per_order
            logger.info(f"Risk Approved: {order.side.upper()} {order.qty:.4f} shares of {order.symbol}")
            approved_orders.append(order)
        return approved_orders