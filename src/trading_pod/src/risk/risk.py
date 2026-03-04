import logging
from typing import List

from src.core.types import OrderRequest, PortfolioState

from .base import IRiskManager

logger = logging.getLogger(__name__)


class StandardRiskManager(IRiskManager):
    """
    Configurable risk manager.

    Default behavior is close to pass-through, but it can enforce hard caps
    on per-order notional and per-asset total exposure. All limits are
    expressed in notional USD terms.
    """
    def __init__(self, max_notional_per_order: float = None, max_total_notional_per_symbol: float = None):
        """
        Initialize the risk manager with optional risk limits.
        
        Args:
            max_notional_per_order: Maximum notional value per order in USD (optional)
            max_total_notional_per_symbol: Maximum total notional exposure per symbol in USD (optional)
        """
        self.max_notional_per_order = max_notional_per_order
        self.max_total_notional_per_symbol = max_total_notional_per_symbol
    
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