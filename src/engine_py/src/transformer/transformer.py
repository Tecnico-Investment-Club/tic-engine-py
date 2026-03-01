import logging
from typing import List, Dict

from src.core.datatypes import TargetAllocation, OrderRequest
from src.engine_py.src.transformer.base import ITransformer

logger = logging.getLogger(__name__)

class Transformer(ITransformer):
    """
    Transformer.
    Converts percentage weights into absolute share quantities, diffs them
    against current holdings, and outputs exact Order Requests.
    """
    def translate_to_orders(self, allocations: List[TargetAllocation], total_equity: float, current_prices: Dict[str, float], current_positions: Dict[str, float]) -> List[OrderRequest]:
        
        orders = []
        target_shares = {}

        # Calculate how many shares we SHOULD have
        for alloc in allocations:
            if alloc.symbol not in current_prices:
                logger.warning(f"No current price for {alloc.symbol}. Cannot calculate shares.")
                continue
            
            # Get the dollar amount we want to allocate to this asset
            target_dollar_amount = total_equity * alloc.weight

            # Convert dollar amount to shares
            shares = target_dollar_amount / current_prices[alloc.symbol]

            # Store the target shares for this symbol
            target_shares[alloc.symbol] = shares

        # Diff against what we CURRENTLY have
        # We check both targets AND existing positions
        # Ensuring we consider all symbols that are either in the target or currently held
        all_symbols = set(target_shares.keys()).union(set(current_positions.keys()))

        # For loop through all symbols to determine if we need to buy/sell/hold
        for symbol in all_symbols:
            target_qty = target_shares.get(symbol, 0.0)
            current_qty = current_positions.get(symbol, 0.0)
            
            # The delta is how many shares we need to buy (+) or sell (-)
            delta = target_qty - current_qty
            
            # If the difference is worth less than $1.00, ignore it.
            # This prevents from making irrelevant micro-trades
            dollar_diff = abs(delta * current_prices.get(symbol, 0))
            if dollar_diff < 1.0:
                continue

            side = "buy" if delta > 0 else "sell"
            
            orders.append(OrderRequest(
                symbol=symbol,
                qty=abs(delta),
                side=side,
                type="market"
            ))
            
            logger.info(
                f"Transformer: {symbol} | Target: {target_qty:.4f} shares | "
                f"Current: {current_qty:.4f} shares -> ORDER: {side.upper()} {abs(delta):.4f}"
            )

        return orders