from typing import Dict, List
import logging
import pandas as pd

from core.datatypes import MarketData, TargetAllocation
from trading_pod.interfaces.IStrategy import IStrategy
from trading_pod.strategy.po_hrp.hrp.functions import _get_weights

logger = logging.getLogger(__name__)

class POHRPStrat(IStrategy):
    """Hierarchical Risk Parity strategy integrated with the current trading engine."""

    def __init__(self, n_clusters: int = 4, lookback_window: int = 186):
        self.n_clusters = max(2, int(n_clusters))
        self.lookback_window = max(30, int(lookback_window))

        logger.info(
            "Initialized POHRPStrat | n_clusters=%s | lookback_window=%s",
            self.n_clusters,
            self.lookback_window,
        )

    def generate_allocations(self, data: MarketData) -> List[TargetAllocation]:
        symbols = data.symbols()
        if not symbols:
            return []

        prices = self._build_prices_dataframe(data)
        if prices.empty:
            logger.warning("[HRP] Empty price dataframe. Falling back to zero allocations.")
            return [TargetAllocation(symbol=symbol, weight=0.0) for symbol in symbols]

        prepared = self._prepare_prices(prices)
        if prepared.empty or prepared.shape[1] < 2:
            logger.warning("[HRP] Insufficient cleaned data for HRP. Using equal-weight fallback.")
            return self._equal_weight_allocations(symbols)

        try:
            clusters = min(self.n_clusters, prepared.shape[1])
            weights = _get_weights(prepared, clusters)
        except Exception as error:
            logger.error("[HRP] Failed to compute HRP weights: %s", error, exc_info=True)
            return self._equal_weight_allocations(symbols)

        positive_weights = {
            symbol: float(weight)
            for symbol, weight in weights.items()
            if float(weight) > 0.0
        }
        total_weight = sum(positive_weights.values())

        if total_weight <= 0.0:
            logger.warning("[HRP] All computed weights are zero. Using equal-weight fallback.")
            return self._equal_weight_allocations(symbols)

        allocations = []
        for symbol in symbols:
            normalized = positive_weights.get(symbol, 0.0) / total_weight
            allocations.append(TargetAllocation(symbol=symbol, weight=normalized))

        logger.info(
            "[HRP] Generated allocations for %s symbols (%s active).",
            len(allocations),
            len(positive_weights),
        )
        return allocations

    def _build_prices_dataframe(self, data: MarketData) -> pd.DataFrame:
        rows_by_symbol: Dict[str, Dict[pd.Timestamp, float]] = {}

        for symbol in data.symbols():
            candles = data.data.get(symbol, [])
            rows_by_symbol[symbol] = {
                pd.Timestamp(candle.timestamp).tz_localize(None): float(candle.close)
                for candle in candles
            }

        if not rows_by_symbol:
            return pd.DataFrame()

        prices = pd.DataFrame(rows_by_symbol).sort_index()
        return prices

    def _prepare_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            return prices

        recent = prices.iloc[-self.lookback_window :]
        recent = recent.dropna(axis=1, how="all")
        if recent.empty:
            return recent

        min_valid_assets = max(2, int(recent.shape[1] * 0.5))
        recent = recent.dropna(axis=0, thresh=min_valid_assets)
        recent = recent.ffill().bfill()
        recent = recent.dropna(axis=1, how="any")
        return recent

    def _equal_weight_allocations(self, symbols: List[str]) -> List[TargetAllocation]:
        if not symbols:
            return []

        weight = 1.0 / len(symbols)
        return [TargetAllocation(symbol=symbol, weight=weight) for symbol in symbols]