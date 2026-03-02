import logging
from typing import List
from src.core.types import OrderRequest, PortfolioState

from .base import IRiskManager

logger = logging.getLogger(__name__)

class StandardRiskManager(IRiskManager):
    """
    A passthrough risk manager for testing/development.
    It does not alter, cap, or reject any orders; it only logs them.
    """
    def validate_orders(self, orders: List[OrderRequest], state: PortfolioState) -> List[OrderRequest]:
        logger.info(f"RiskManager (Standard): Received {len(orders)} orders from the Transformer.")
        
        for order in orders:
            logger.info(
                f"RiskManager (Standard) PASSING THROUGH: "
                f"{order.side.upper()} {order.qty:.4f} shares of {order.symbol} "
                f"[{order.type.upper()}]"
            )
            
        # Return the exact same list, completely unmodified
        return orders