import logging
from typing import List
from src.core.types import TargetAllocation, PortfolioState, OrderRequest, OrderSide

from .base import ITransformer

logger = logging.getLogger(__name__)

class StandardTransformer(ITransformer):
    def __init__(self, min_order_value: float = 10.0, buffer_pct: float = 0.05):
        # Prevent micro-transactions
        self.min_order_value = min_order_value
        # Leave a percentage of equity in cash to handle market order slippage limits
        self.buffer_pct = buffer_pct

    def transform(self, allocations: List[TargetAllocation], state: PortfolioState) -> List[OrderRequest]:
        orders = []
        total_equity = state.total_equity
        # Deduct the buffer to get the actual equity we are allowed to allocate
        tradable_equity = total_equity * (1.0 - self.buffer_pct)
        
        logger.debug(f"Total Equity: {total_equity:.2f} | Tradable Equity: {tradable_equity:.2f}")
        logger.debug(f"State Prices received: {state.prices}")

        # Loop through each target allocation
        for alloc in allocations:
            symbol = alloc.symbol
            target_weight = alloc.weight
            
            logger.debug(f"Evaluating {symbol} (Target Weight: {target_weight})")

            # Get the latest price
            price = state.prices.get(symbol)

            # If there is no price data for this symbol, skip it
            if not price or price <= 0:
                logger.warning(f"Transformer: No valid price for {symbol}. Price value seen: {price}. Skipping.")
                continue

            # Calculate the Delta: target value - current value
            # CRITICAL: Use tradable_equity here instead of total_equity
            target_value = tradable_equity * target_weight
            current_qty = state.positions.get(symbol, 0.0)
            current_value = current_qty * price

            delta_value = target_value - current_value
            delta_qty = delta_value / price

            logger.debug(f"{symbol} | Target Val: {target_value:.2f} | Current Val: {current_value:.2f} | Delta Val: {delta_value:.2f}")

            # Filter out small orders below the minimum order value threshold
            if abs(delta_value) < self.min_order_value:
                logger.debug(f"{symbol} delta ({delta_value:.2f}) < min_order_value ({self.min_order_value}). Skipping.")
                continue

            # Define the order side based on whether we need to buy or sell
            side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
            order_qty = abs(delta_qty)

            if side == OrderSide.SELL:
                # Ensure we don't try to sell more than we have
                order_qty = min(order_qty, abs(current_qty))

            orders.append(OrderRequest(
                    symbol=symbol,
                    qty=order_qty,
                    side=side
                    ))

        # IMPORTANT: Sort orders so SELLs happen before BUYs.
        # This frees up cash first during rebalancing!
        orders.sort(key=lambda x: 0 if x.side == OrderSide.SELL else 1)

        logger.info(f"Transformer: Converted {len(allocations)} weights into {len(orders)} order requests.")
        return orders
