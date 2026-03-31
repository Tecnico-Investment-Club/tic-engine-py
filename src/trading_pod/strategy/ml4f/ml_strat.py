import pandas as pd
import numpy as np
from typing import List
from collections import deque
from pickle import load
import os
import sys

from core.datatypes import MarketData, TargetAllocation
from trading_pod.interfaces.IStrategy import IStrategy

# ML Team Feature Functions
from .Chapter_2 import cusum_filter_live
from .Chapter_5 import fracDiff_FFD_live, fracDiff_FFD_initial
from .features_live import (
    compute_macd_initial, update_macd, compute_macd_normalized_initial, update_macd_normalized,
    compute_atr_log_14_initial, update_atr_log_14, compute_entropy_100_initial, update_entropy_100,
    aroon_init, aroon_update, mad_normalized, mad_normalized_update, volume_weighted_ma_ratio,
    volume_weighted_ma_ratio_update, normalized_volume_index, normalized_volume_index_update,
    short_long_volume_ratio_indicator, short_long_volume_ratio_indicator_update,
    ppo_init, ppo_update, adx_init, adx_update 
)

class MLBtcStrategy(IStrategy):
    """
    Event-driven implementation of the ML Team's BTC Hourly CUSUM Strategy.
    Preserves all indicator states across ticks/bars without needing to reload CSVs.
    """
    def __init__(self, symbol: str = "BTCUSD"):
        self.symbol = symbol
        self.is_bootstrapped = False
        self.last_timestamp = None
        
        # 1. Load Models (Robust Pathing)
        current_dir = os.path.dirname(__file__)
        if current_dir not in sys.path:
            sys.path.append(current_dir)
            
        model_dir = os.path.join(current_dir, 'models')
        print(f"[DEBUG] Loading models from: {model_dir}")
        self.model_side = load(open(os.path.join(model_dir, 'best_primary_model_all_features.pkl'), 'rb'))
        self.model_meta = load(open(os.path.join(model_dir, 'best_meta_model_all_features.pkl'), 'rb'))
        
        # 2. Strategy Tracking States
        self.active_bets = pd.DataFrame(columns=['side', 'size', 'probs', 'trgt', 't1', 'Adj Close'])
        self.active_bets.index.name = "date"
        self.sPos, self.sNeg = 0, 0
        
        # 3. Data History States
        self.combined_closes = None
        self.prev_closes = None
        self.fracdiff_w = None
        
        # 4. Indicator Persistent States
        self.ema_fast = self.ema_slow = self.signal_line = None
        self.atr_window = deque(maxlen=14)
        self.atr_window_macd = deque(maxlen=101)
        self.entropy_window = deque(maxlen=100)
        
        self.short_alpha = 2.0 / (10 + 1)
        self.long_alpha = 2.0 / (100 + 1)
        self.last_short_sum = self.last_long_sum = 0.0
        
        # Native ML script feature states
        self.state_aroon = self.state_mad = self.state_vol_ma = None
        self.state_norm_vol = self.state_vol_ratio = self.state_ppo = self.state_adx = None

    def _get_daily_vol(self, close_series, span0=168):
        """Calculates dynamic target thresholds using integer indexing to avoid timezone bugs."""
        # 1. Find the integer indices for the price 1 day ago
        # We use searchsorted to find the position, then subtract 1 to get the prior bar
        idx_1d_ago = close_series.index.searchsorted(close_series.index - pd.Timedelta(days=1))
        
        # 2. Filter out indices that are out of bounds (0)
        # We only want indices where a '1 day ago' actually exists in our 300-candle window
        valid_indices = idx_1d_ago[idx_1d_ago > 0] - 1
        
        # 3. Slice the series to match the length of the found indices
        # This ensures we are comparing 'Today's Price' vs 'Yesterday's Price'
        current_prices = close_series.iloc[-len(valid_indices):]
        past_prices = close_series.iloc[valid_indices]
        
        # 4. Calculate returns and exponential volatility
        # Use .values to ignore index labels and avoid the KeyError entirely
        returns = (current_prices.values / past_prices.values) - 1
        
        return pd.Series(returns, index=current_prices.index).ewm(span=span0).std()
    
    def _convert_to_df(self, data: MarketData) -> pd.DataFrame:
        """Translates engine MarketData into the ML Team's expected DataFrame format."""
        # Use .get() to safely handle cases where the symbol is missing
        list_candles = data.data.get(self.symbol, [])

        if not list_candles:
            print(f"[DEBUG] No data found for symbol: {self.symbol}")
            return pd.DataFrame() # Fixed: Capital 'F'

        # Use list_candles (matching the variable above) 
        # Use .model_dump() for modern Pydantic compatibility
        df = pd.DataFrame([c.model_dump() for c in list_candles])

        # Rename to match the ML team's exact requirements in live_final.py
        df.rename(columns={
            'timestamp': 'date', 
            'close': 'Adj Close', 
            'open': 'Open', 
            'high': 'High', 
            'low': 'Low', 
            'volume': 'Volume'
        }, inplace=True)

        # Ensure datetime indexing for time-series operations (FracDiff, CUSUM)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Sort index to ensure chronological order for rolling indicators
        df.sort_index(inplace=True)
        
        return df

    def _bootstrap(self, df: pd.DataFrame):
        """Warms up the indicators mimicking the `main()` initialization from live_final.py"""
        print(f"[DEBUG] Bootstrapping strategy with {len(df)} historical rows...")
        self.combined_closes = df['Adj Close'].copy()
        
        # FracDiff
        self.prev_closes = df[['Adj Close']][-250:].copy()
        _, self.fracdiff_w = fracDiff_FFD_initial(df[['Adj Close']][-238:], d=0.45, thres=1e-4)

        # MACD
        _, e_fast, e_slow, s_line = compute_macd_initial(df['Adj Close'])
        self.ema_fast, self.ema_slow, self.signal_line = float(e_fast.iloc[-1]), float(e_slow.iloc[-1]), float(s_line.iloc[-1])

        # ATR / Entropy
        for idx in df.tail(14).index:
            self.atr_window.append((df.at[idx,'Open'], df.at[idx,'High'], df.at[idx,'Low'], df.at[idx,'Adj Close']))
        for c in df['Adj Close'].tail(100):
            self.entropy_window.append(c)

        # MACD Normalized EMAs
        closes = df['Adj Close'].values
        self.last_short_sum = self.last_long_sum = closes[0]
        for price in closes[1:]:
            self.last_short_sum = self.short_alpha * price + (1 - self.short_alpha) * self.last_short_sum
            self.last_long_sum  = self.long_alpha  * price + (1 - self.long_alpha)  * self.last_long_sum
            
        for i in range(len(df) - 101, len(df)):
            self.atr_window_macd.append((df['High'].iloc[i], df['Low'].iloc[i], df['Adj Close'].iloc[i]))

        # State Initialization
        _, self.state_aroon = aroon_init(df['High'].values, df['Low'].values, lookback=100)
        _, self.state_mad = mad_normalized(df['Adj Close'].values, df['High'].values, df['Low'].values, 100, 10, 10)
        _, self.state_vol_ma = volume_weighted_ma_ratio(df['Adj Close'].values, df['Volume'].values, lookback=20)
        _, self.state_norm_vol = normalized_volume_index(df['Adj Close'].values, df['Volume'].values)
        _, self.state_vol_ratio = short_long_volume_ratio_indicator(df['Volume'].values)
        _, self.state_ppo = ppo_init(df['Adj Close'].values, 10, 100, 10)
        _, self.state_adx = adx_init(df['High'].values, df['Low'].values, df['Adj Close'].values, 14)

        self.last_timestamp = df.index.max()
        self.is_bootstrapped = True
        print(f"[DEBUG] Bootstrap complete. Last timestamp recorded: {self.last_timestamp}")

    def _process_new_data(self, new_data: pd.DataFrame):
        """Mirrors the `while True:` loop logic from live_final.py for sequential new bars."""
        print(f"[DEBUG] Processing {len(new_data)} new candle(s)...")

        # Use combined_closes to avoid NaN if new_data is only 1 row (Engine adjustment)
        h = 2 * self.combined_closes.diff().std()
        print(f"[DEBUG] Current CUSUM Threshold (h): {h:.4f}")

        cusum_events, self.sPos, self.sNeg = cusum_filter_live(
            new_data.reset_index(), 'Adj Close', h, 
            sPos_prev=self.sPos, sNeg_prev=self.sNeg
        )
        
        if not cusum_events.empty:
            cusum_events.set_index('date', inplace=True)
            print(f"[DEBUG] >>> {len(cusum_events)} CUSUM Event(s) Triggered! <<<")
        else:
            print("[DEBUG] No CUSUM events triggered this cycle.")

        close_new = new_data[['Adj Close']]
        new_fracdiff_series = fracDiff_FFD_live(close_new, self.fracdiff_w, self.prev_closes)
        
        self.combined_closes = pd.concat([self.combined_closes, close_new['Adj Close']])
        self.prev_closes = pd.concat([self.prev_closes, close_new]).groupby(level=0).last().iloc[-250:]
        trgt_series = self._get_daily_vol(self.combined_closes)

        feature_rows_for_prediction = []

        # Iterate sequentially to maintain accurate indicator states across multiple incoming bars
        for timestamp, row in new_data.iterrows():
            new_open, new_high, new_low = row['Open'], row['High'], row['Low']
            new_close, new_vol = row['Adj Close'], row['Volume']

            macd_update_df, self.ema_fast, self.ema_slow, self.signal_line = update_macd(
                pd.Series([new_close]), self.ema_fast, self.ema_slow, self.signal_line
            )
            current_macd = macd_update_df['macd'].iloc[-1]

            atr_log_val, self.atr_window = update_atr_log_14(new_open, new_high, new_low, new_close, self.atr_window)
            macd_norm_val, self.last_short_sum, self.last_long_sum, self.atr_window_macd = update_macd_normalized(
                new_close, new_high, new_low, self.last_short_sum, self.last_long_sum, 
                self.atr_window_macd, self.long_alpha, self.short_alpha
            )
            entropy_val, self.entropy_window = update_entropy_100(new_close, self.entropy_window)
            
            aroon_val, self.state_aroon = aroon_update(new_high, new_low, self.state_aroon, mode="diff", lookback=100)
            mad_val, self.state_mad = mad_normalized_update(new_close, new_high, new_low, self.state_mad)
            vol_ma_val, self.state_vol_ma = volume_weighted_ma_ratio_update(new_close, new_vol, self.state_vol_ma)
            norm_vol_val, self.state_norm_vol = normalized_volume_index_update(new_close, new_vol, self.state_norm_vol)
            vol_ratio_val, self.state_vol_ratio = short_long_volume_ratio_indicator_update(new_vol, self.state_vol_ratio)
            ppo_val, self.state_ppo = ppo_update(new_close, self.state_ppo)
            adx_val, self.state_adx = adx_update(new_high, new_low, new_close, self.state_adx)

            # Reindexing Equivalent: Only append features to predict if CUSUM triggered on THIS timestamp
            if not cusum_events.empty and timestamp in cusum_events.index:
                feature_dict = {
                    'Adj Close': new_fracdiff_series.loc[timestamp, 'Adj Close'] if timestamp in new_fracdiff_series.index else 0,
                    'trgt': trgt_series.loc[timestamp] if timestamp in trgt_series.index else 0,
                    'mad': mad_val,
                    'ppo': ppo_val,
                    'adx': adx_val,
                    'aroon': aroon_val,
                    'normalized_volume': norm_vol_val,
                    'volume_ratio': vol_ratio_val,
                    'vol_ma': vol_ma_val,
                    'macd_initial': current_macd,
                    'macd_normalized': macd_norm_val,
                    'atr_log_14': atr_log_val,
                    'entropy_100': entropy_val
                }
                feature_rows_for_prediction.append(pd.Series(feature_dict, name=timestamp))

        # Model Predictions
        if feature_rows_for_prediction:
            features_df = pd.DataFrame(feature_rows_for_prediction).dropna()
            print(f"[DEBUG] Generated feature matrix shape: {features_df.shape}")

            if not features_df.empty:
                side = self.model_side.predict(features_df)
                size = self.model_meta.predict(features_df)
                probs = self.model_meta.predict_proba(features_df)[:, 1]

                print(f"[DEBUG] Raw Model Output -> Side: {side.tolist()}, Size: {size.tolist()}, Probs: {probs.tolist()}")

                predictions = pd.DataFrame({'side': side, 'size': size, 'probs': probs}, index=features_df.index)
                
                # Filter out 0 bets
                predictions = predictions[(predictions['side'] != 0) & (predictions['size'] != 0)]
                print(f"[DEBUG] Actionable predictions after zero-filter: {len(predictions)}")

                if not predictions.empty:
                    predictions['Adj Close'] = cusum_events.loc[predictions.index, 'Adj Close']
                    predictions['trgt'] = features_df['trgt']
                    predictions['t1'] = predictions.index + pd.Timedelta(days=2) # ML hardcoded 2-day expiry
                    self.active_bets = pd.concat([self.active_bets, predictions])
                    print(f"[DEBUG] Added {len(predictions)} active bet(s) to state. Total active: {len(self.active_bets)}")

    def _clear_expired_bets(self):
        """Mirrors the `clear_active_bets` function from live_final.py"""
        starting_bets = len(self.active_bets)
        
        # 1. Drop time-expired
        self.active_bets = self.active_bets[self.active_bets['t1'] > pd.to_datetime('now')]
        to_remove = []

        # 2. Drop hit stop/targets
        for idx, row in self.active_bets.iterrows():
            entry_time, entry_price, target = idx, row['Adj Close'], row['trgt']
            if pd.isna(target):
                continue

            try:
                post_entry = self.combined_closes.loc[entry_time:]
            except KeyError:
                continue

            if not post_entry.empty:
                if (post_entry >= entry_price + target).any() or (post_entry <= entry_price - target).any():
                    to_remove.append(idx)

        if to_remove:
            self.active_bets.drop(index=to_remove, inplace=True)
            
        ending_bets = len(self.active_bets)
        if starting_bets != ending_bets:
            print(f"[DEBUG] Cleared {starting_bets - ending_bets} expired/triggered bet(s).")

    def generate_allocations(self, data: MarketData) -> List[TargetAllocation]:
        """Main Strategy Execution Pipeline."""

        df = self._convert_to_df(data)
        
        print(f"[DEBUG] generate_allocations triggered. Extracted {len(df)} total rows from MarketData.")
        
        if df.empty:
            return []

        if not self.is_bootstrapped:
            if len(df) < 250:
                print(f"[DEBUG] Awaiting more data to bootstrap. Have {len(df)}, need 250.")
                return []
            self._bootstrap(df)
            return [TargetAllocation(symbol=self.symbol, weight=0.0)]

        # Filter strictly for new, unprocessed candles
        new_data = df[df.index > self.last_timestamp]
        
        if not new_data.empty:
            self._process_new_data(new_data)
            self.last_timestamp = new_data.index.max()
        else:
            print(f"[DEBUG] No new candles past {self.last_timestamp}. Standing by.")

        # Position Sizing matching ML team's exact math
        self._clear_expired_bets()
        
        new_bet_size = 0.0
        if not self.active_bets.empty:
            probs_sum = self.active_bets['probs'].sum()
            if probs_sum > 0:
                weighted_sizes = self.active_bets['probs'] * self.active_bets['size']
                new_bet_size = weighted_sizes.sum() / probs_sum
                print(f"[DEBUG] Calculated probability-weighted bet size: {new_bet_size:.4f}")

        new_bet_size = max(0.0, new_bet_size)
        bet_size = (new_bet_size // 0.1) * 0.1
        
        print(f"[DEBUG] Final Target Allocation Weight: {bet_size} (Discretized from {new_bet_size:.4f})")

        return [TargetAllocation(symbol=self.symbol, weight=float(bet_size))]