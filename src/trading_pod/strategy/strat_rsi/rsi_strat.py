import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from core.datatypes import MarketData, TargetAllocation
from trading_pod.interfaces.IStrategy import IStrategy

logger = logging.getLogger(__name__)


class RSIStrategy(IStrategy):
    """
    RSI Strategy
    """

    def __init__(
        self,
        period: int = 14,
        overbought: int = 70,
        oversold: int = 30,
        lookback_window: int = 50,
        trend_period: int = 200,
    ):
        self.period = int(period)
        self.overbought = overbought
        self.oversold = oversold
        self.trend_period = trend_period

        self.lookback_window = max(
            self.trend_period, 2 * self.period, int(lookback_window)
        )

        logger.info(
            "Initialized RSIStrategy | period=%s | overbought=%s | oversold=%s | lookback_window=%s",
            self.period,
            self.overbought,
            self.oversold,
            self.lookback_window,
        )

    def generate_allocations(self, data: MarketData) -> List[TargetAllocation]:
        symbols = data.symbols()
        if not symbols:
            return []

        prices = self.build_prices_dataframe(data)
        if prices.empty:
            logger.warning("[RSI] Empty price data. Falling back to zero allocations.")
            return [TargetAllocation(symbol=symbol, weight=0.0) for symbol in symbols]

        prepared = self.prepare_prices(prices)

        raw_weights = {}

        for symbol in symbols:
            try:
                if symbol not in prepared.columns:
                    raw_weights[symbol] = 0.0
                    continue

                series = prepared[symbol]
                if len(series) <= self.period:
                    raw_weights[symbol] = 0.0
                    continue

                # SMA Calculation

                sma = series.rolling(window=self.trend_period).mean()

                # RSI Calculation
                delta = series.diff()
                gain = (
                    (delta.where(delta > 0, 0))
                    .ewm(alpha=1 / self.period, adjust=False)
                    .mean()
                )
                loss = (
                    (-delta.where(delta < 0, 0))
                    .ewm(alpha=1 / self.period, adjust=False)
                    .mean()
                )

                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]

                # Maps RSI linearly
                #
                if series[-1] > sma.iloc[-1]:
                    weight = 1.0 - (current_rsi - self.oversold) / (
                        self.overbought - self.oversold
                    )
                    weight = float(np.clip(weight, 0.0, 1.0))

                    raw_weights[symbol] = weight
                    logger.debug(
                        "[RSI] %s: RSI=%.2f, Raw Weight=%.4f",
                        symbol,
                        current_rsi,
                        weight,
                    )
                else:
                    raw_weights[symbol] = 0.0

            except Exception as e:
                logger.error("[RSI] Error calculating RSI for %s: %s", symbol, e)
                raw_weights[symbol] = 0.0

        # Prevents total portfolio weight from exceeding 100%
        total_raw_weight = sum(raw_weights.values())

        allocations = []
        for symbol in symbols:
            final_weight = 0.0
            if total_raw_weight > 0:
                final_weight = raw_weights[symbol] / total_raw_weight

            allocations.append(TargetAllocation(symbol=symbol, weight=final_weight))

        active = sum(1 for w in raw_weights.values() if w > 0)
        logger.info("[RSI] %s/%s symbols active.", active, len(symbols))
        return allocations

    def build_prices_dataframe(self, data: MarketData) -> pd.DataFrame:
        """Standardized conversion from MarketData to DataFrame."""
        rows_by_symbol: Dict[str, Dict[pd.Timestamp, float]] = {}

        for symbol in data.symbols():
            candles = data.data.get(symbol, [])
            rows_by_symbol[symbol] = {
                pd.Timestamp(candle.timestamp).tz_localize(None): float(candle.close)
                for candle in candles
            }

        if not rows_by_symbol:
            return pd.DataFrame()

        return pd.DataFrame(rows_by_symbol).sort_index()

    def prepare_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            return prices

        cleaned = prices.iloc[-self.lookback_window :]
        cleaned = cleaned.dropna(axis=1, how="all")
        if cleaned.empty:
            return cleaned

        min_valid_assets = max(2, int(cleaned.shape[1] * 0.5))
        cleaned = cleaned.dropna(axis=0, thresh=min_valid_assets)
        cleaned = cleaned.ffill().bfill()
        cleaned = cleaned.dropna(axis=1, how="any")
        return cleaned
