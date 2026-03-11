import logging
from typing import List

from core.datatypes import MarketData, TargetAllocation
from trading_pod.interfaces.IStrategy import IStrategy

logger = logging.getLogger("TRADING.STRATEGY")

class MomentumStrat(IStrategy):
    """
    A simple Momentum strategy.
    Buys assets whose price increased over the last N candles.
    Sells assets whose price decreased over the last N candles.
    """
    def __init__(self, lookback_window: int = 10, min_gain: float = 0.02):
        self.lookback_window = lookback_window
        self.min_gain = min_gain  # Minimum relative gain over the lookback period to trigger a buy
        
        logger.info(
            f"Initialized MomentumStrat | Lookback: {self.lookback_window}, "
            f"Min Gain: {self.min_gain:.2%}"
        )

    def generate_allocations(self, data: MarketData) -> List[TargetAllocation]:
        buy_list = []
        
        for symbol in data.symbols():
            candles = data.data.get(symbol, [])
            
            if len(candles) < self.lookback_window:
                logger.debug(f"[{symbol}] Not enough data. Need {self.lookback_window}, got {len(candles)}.")
                continue
            
            start_price = candles[-self.lookback_window].close
            current_price = candles[-1].close
            gain = (current_price - start_price) / start_price
            
            if gain >= self.min_gain:
                logger.debug(f"[{symbol}] Gain {gain:.2%} >= {self.min_gain:.2%}. Adding to BUY list.")
                buy_list.append(symbol)
            else:
                logger.debug(f"[{symbol}] Gain {gain:.2%} < {self.min_gain:.2%}. Excluded (SELL/HOLD).")
        
        num_buys = len(buy_list)
        weight_per_asset = 1.0 / num_buys if num_buys > 0 else 0.0
        
        allocations = [
            TargetAllocation(symbol=symbol, weight=weight_per_asset if symbol in buy_list else 0.0)
            for symbol in data.symbols()
        ]
        
        logger.info(f"Momentum: {num_buys} assets meet momentum criteria. Weight per asset: {weight_per_asset:.4f}")
        return allocations
