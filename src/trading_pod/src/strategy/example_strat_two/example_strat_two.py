import logging
from typing import List
from src.core.types import MarketData, PortfolioState, TargetAllocation
from ..base import IStrategy

logger = logging.getLogger(__name__)

class ExampleStratTwo(IStrategy):
    """
    An RSI Mean Reversion Strategy.
    Buys assets when their RSI drops below the oversold threshold.
    """
    def __init__(self, rsi_window: int = 14, oversold_threshold: float = 30.0, max_weight: float = 0.50):
        self.rsi_window = rsi_window
        self.oversold_threshold = oversold_threshold
        self.max_weight = max_weight
        logger.info(f"Initialized RSI Strat | Window: {self.rsi_window}, Oversold < {self.oversold_threshold}")

    def compute_weights(self, market_data: MarketData, state: PortfolioState) -> List[TargetAllocation]:
        bullish_assets = []
        
        for symbol in market_data.symbols():
            candles = market_data.data.get(symbol, [])
            
            # We need window + 1 candles to calculate the price differences
            if len(candles) <= self.rsi_window:
                continue
            
            # 1. Calculate RSI Math
            gains = []
            losses = []
            
            subset = candles[-(self.rsi_window + 1):]
            for i in range(1, len(subset)):
                change = subset[i].close - subset[i-1].close
                if change > 0:
                    gains.append(change)
                    losses.append(0.0)
                else:
                    gains.append(0.0)
                    losses.append(abs(change))
                    
            avg_gain = sum(gains) / self.rsi_window
            avg_loss = sum(losses) / self.rsi_window
            
            if avg_loss == 0:
                rsi = 100.0  # Pure up trend
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
                
            logger.debug(f"[{symbol}] Calculated RSI: {rsi:.2f}")
            
            # 2. Check Signal
            if rsi < self.oversold_threshold:
                bullish_assets.append(symbol)

        # 3. Distribute Weights safely
        allocations = []
        num_bullish = len(bullish_assets)
        
        # Cap the weight so we don't dump 100% of our portfolio into 1 asset
        weight_per_asset = min(self.max_weight, 1.0 / num_bullish) if num_bullish > 0 else 0.0

        for symbol in market_data.symbols():
            if symbol in bullish_assets:
                allocations.append(TargetAllocation(symbol=symbol, weight=weight_per_asset))
            else:
                allocations.append(TargetAllocation(symbol=symbol, weight=0.0))
                
        logger.info(f"Strategy Two: {num_bullish} oversold signals found.")
        return allocations