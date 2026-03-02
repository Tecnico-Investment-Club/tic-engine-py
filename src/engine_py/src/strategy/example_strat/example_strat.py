import logging
from typing import List
from src.core.types import MarketData, PortfolioState, TargetAllocation
from ..base import IStrategy

logger = logging.getLogger(__name__)

class ExampleStrat(IStrategy):
    """
    A simple SMA crossover strategy.
    Identifies bullish assets (Price > SMA) and allocates equity equally among them.
    """
    def __init__(self, sma_window: int = 20):
        self.sma_window = sma_window
        logger.info(f"Initialized ExampleStrat | SMA Window: {self.sma_window}")

    def compute_weights(self, market_data: MarketData, state: PortfolioState) -> List[TargetAllocation]:
        bullish_assets = []
        
        # Identify all bullish assets
        for symbol in market_data.symbols():
            candles = market_data.data.get(symbol, [])
            
            if len(candles) < self.sma_window:
                logger.debug(f"[{symbol}] Not enough data. Need {self.sma_window}, got {len(candles)}.")
                continue
            
            current_price = candles[-1].close
            sma = sum(c.close for c in candles[-self.sma_window:]) / self.sma_window
            
            if current_price > sma:
                bullish_assets.append(symbol)

        # Calculate equal weight distribution
        num_bullish = len(bullish_assets)
        weight_per_asset = 1.0 / num_bullish if num_bullish > 0 else 0.0
        
        allocations = []
        
        # Assign weights to ALL tracked symbols
        for symbol in market_data.symbols():
            if symbol in bullish_assets:
                allocations.append(TargetAllocation(symbol=symbol, weight=weight_per_asset))
            else:
                # Set 0.0 for non-bullish assets so the Transformer knows to sell them
                allocations.append(TargetAllocation(symbol=symbol, weight=0.0))
                
        logger.info(f"Strategy: {num_bullish} bullish signals. Weight per asset: {weight_per_asset:.4f}")
        
        return allocations