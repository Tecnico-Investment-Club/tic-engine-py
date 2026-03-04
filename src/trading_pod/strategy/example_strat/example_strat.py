import logging
from typing import List

from core.datatypes import MarketData, TargetAllocation
from trading_pod.interfaces.IStrategy import IStrategy

logger = logging.getLogger("TRADING.STRATEGY")

class PingPongStrat(IStrategy):
    """
    A simple Mean Reversion (Ping-Pong) strategy.
    Buys when the asset price drops below X% of its SMA.
    Sells when the asset price rises above Y% of its SMA.
    """
    def __init__(self, sma_window: int = 20, buy_threshold: float = 10, sell_threshold: float = 20):
        self.sma_window = sma_window
        
        # X: Buy if price is under 98% of the moving average
        self.buy_threshold = buy_threshold 
        
        # Y: Sell if price is over 102% of the moving average
        self.sell_threshold = sell_threshold 
        
        logger.info(
            f"Initialized PingPongStrat | SMA: {self.sma_window}, "
            f"Buy < {self.buy_threshold}, Sell > {self.sell_threshold}"
        )

    def generate_allocations(self, data: MarketData) -> List[TargetAllocation]:
        buy_list = []
        
        # 1. Identify which assets meet our "Under X" criteria
        for symbol in data.symbols():
            candles = data.data.get(symbol, [])
            
            if len(candles) < self.sma_window:
                logger.debug(f"[{symbol}] Not enough data. Need {self.sma_window}, got {len(candles)}.")
                continue
            
            current_price = candles[-1].close
            sma = sum(c.close for c in candles[-self.sma_window:]) / self.sma_window
            
            # Calculate the ratio of price to SMA
            ratio = current_price / sma
            
            if ratio < self.buy_threshold:
                logger.debug(f"[{symbol}] {ratio:.3f} is < {self.buy_threshold}. Adding to BUY list.")
                buy_list.append(symbol)
            elif ratio > self.sell_threshold:
                logger.debug(f"[{symbol}] {ratio:.3f} is > {self.sell_threshold}. Excluded (SELL).")
            else:
                logger.debug(f"[{symbol}] {ratio:.3f} is in the middle. Excluded (SELL/HOLD).")

        # Calculate equal weight distribution for the assets we want to hold
        num_buys = len(buy_list)
        weight_per_asset = 1.0 / num_buys if num_buys > 0 else 0.0
        
        allocations = []
        
        # 3. Assign the target weights
        for symbol in data.symbols():
            if symbol in buy_list:
                allocations.append(TargetAllocation(symbol=symbol, weight=weight_per_asset))
            else:
                # Giving an asset 0.0 weight tells the Transformer to liquidate any current holdings
                allocations.append(TargetAllocation(symbol=symbol, weight=0.0))
                
        logger.info(f"PingPong: {num_buys} assets under buy threshold. Weight per asset: {weight_per_asset:.4f}")
        
        return allocations