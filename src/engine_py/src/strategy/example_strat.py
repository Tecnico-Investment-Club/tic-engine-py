import pandas as pd
from typing import List, Dict
import logging

from src.core.datatypes import Candle, TargetAllocation
from src.engine_py.src.strategy.base import IStrategy

logger = logging.getLogger(__name__)

class MovingAverageCrossStrategy(IStrategy):
    def __init__(self, short_window: int = 9, long_window: int = 21):
        self.short_window = short_window
        self.long_window = long_window

    def generate_allocations(self, market_data: Dict[str, List[Candle]]) -> List[TargetAllocation]:
        allocations = []
        bullish_assets = []

        # Analyze each asset
        for symbol, candles in market_data.items():
            if len(candles) < self.long_window:
                logger.warning(f"Not enough data for {symbol}. Skipping.")
                continue

            # Convert to Pandas DataFrame for easy math
            df = pd.DataFrame([c.model_dump() for c in candles])
            
            # Calculate Moving Averages
            df['SMA_short'] = df['close'].rolling(window=self.short_window).mean()
            df['SMA_long'] = df['close'].rolling(window=self.long_window).mean()

            # Get the most recent values
            latest_short = df['SMA_short'].iloc[-1]
            latest_long = df['SMA_long'].iloc[-1]

            # Strategy Logic: Golden Cross (Short MA > Long MA)
            if latest_short > latest_long:
                logger.info(f"{symbol} is BULLISH (Short MA: {latest_short:.2f} > Long MA: {latest_long:.2f})")
                bullish_assets.append(symbol)
            else:
                logger.info(f"{symbol} is BEARISH (Short MA: {latest_short:.2f} < Long MA: {latest_long:.2f})")

        # Determine Weights: Equally distribute among bullish assets
        if not bullish_assets:
            logger.info("No bullish assets found. Target allocations: 0% everywhere (Cash).")
            return [] # Empty list means "sell everything"

        total_alocation_cap = 0.90 # Keep 10% cash buffer
        weight_per_asset = total_alocation_cap / len(bullish_assets)
        
        for symbol in bullish_assets:
            allocations.append(TargetAllocation(symbol=symbol, weight=weight_per_asset))

        return allocations