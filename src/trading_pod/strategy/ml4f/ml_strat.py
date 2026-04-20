import os
import sys
import logging
from pickle import load
from collections import deque
from typing import List

import numpy as np
import pandas as pd

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

logger = logging.getLogger("TRADING.ML")

class MLBtcStrategy(IStrategy):
    """
    Event-driven implementation of the ML Team's BTC Hourly Strategy.
    Bounded memory footprint, safe index alignment, and strict feature ordering.
    """
    def __init__(self, symbol: str = "BTCUSD"):
        self.symbol = symbol
        self.is_bootstrapped = False
        self.last_timestamp = None
        self.max_history = 2000  # Prevent memory leaks in live engines

        # --- 1. Load ML Models ---
        current_dir = os.path.dirname(__file__)
        if current_dir not in sys.path:
            sys.path.append(current_dir)

        model_dir = os.path.join(current_dir, 'models')
        
        with open(os.path.join(model_dir, 'best_primary_model_all_features.pkl'), 'rb') as f1, \
             open(os.path.join(model_dir, 'best_meta_model_all_features.pkl'), 'rb') as f2:
            self.model_side = load(f1)
            self.model_meta = load(f2)

        # --- 2. Trade Tracking ---
        self.active_bets = pd.DataFrame(columns=['side', 'size', 'probs', 'trgt', 't1', 'Adj Close'])
        self.active_bets.index.name = "date"
        
        # --- 3. Persistent Data States ---
        self.sPos = 0
        self.sNeg = 0
        self.combined_closes = pd.Series(dtype=float)
        self.prev_closes = pd.DataFrame()
        self.fracdiff_w = np.array([])
        self.trgt_series = pd.Series(dtype=float)

        # --- 4. Indicator States ---
        self.ema_fast = self.ema_slow = self.signal_line = None
        self.short_alpha = 2.0 / (10 + 1)
        self.long_alpha = 2.0 / (100 + 1)
        self.last_short_sum = self.last_long_sum = 0.0

        # Sliding Windows
        self.atr_window = deque(maxlen=14)
        self.atr_window_macd = deque(maxlen=101)
        self.entropy_window = deque(maxlen=100)

        # Complex Feature Dictionaries
        self.state_aroon = None
        self.state_mad = None
        self.state_vol_ma = None
        self.state_norm_vol = None
        self.state_vol_ratio = None
        self.state_ppo = None
        self.state_adx = None
        
        # Latest calculated feature values cache
        self.current_features = {}

    # =========================================================================
    # CORE INTERFACE
    # =========================================================================

    def generate_allocations(self, data: MarketData) -> List[TargetAllocation]:
        df = self._convert_to_df(data)
        if df.empty: return []

        if not self.is_bootstrapped:
            if len(df) < 250:
                logger.warning(f"Awaiting data for bootstrap. Have {len(df)}, need 250.")
                return []
            self._bootstrap(df)
            return [TargetAllocation(symbol=self.symbol, weight=0.0)]

        # Get only the newly arrived data
        new_data = df[df.index > self.last_timestamp]
        
        if not new_data.empty:
            for timestamp, row in new_data.iterrows():
                self._process_single_candle(timestamp, row)
            self.last_timestamp = new_data.index.max()

        self._clear_expired_bets()
        
        # Calculate size based on active meta-model predictions
        bet_size = 0.0
        if not self.active_bets.empty:
            probs_sum = self.active_bets['probs'].sum()
            if probs_sum > 0:
                weighted_sizes = self.active_bets['probs'] * self.active_bets['size']
                bet_size = weighted_sizes.sum() / probs_sum

        # Ensure long-only constraints (0.0 to 1.0) and discretize
        bet_size = max(0.0, min(1.0, bet_size))
        final_weight = round((bet_size // 0.1) * 0.1, 2)

        return [TargetAllocation(symbol=self.symbol, weight=final_weight)]

    # =========================================================================
    # ENGINE LOGIC
    # =========================================================================

    def _process_single_candle(self, timestamp: pd.Timestamp, row: pd.Series):
        """Processes exactly one candle, updates states, and checks for triggers."""
        close_price = float(row['Adj Close'])
        vol = float(row['Volume'])
        high = float(row['High'])
        low = float(row['Low'])
        open_p = float(row['Open'])
        
        # 1. Update historical buffers safely
        self.combined_closes.loc[timestamp] = close_price
        self.combined_closes = self.combined_closes.iloc[-self.max_history:]
        
        close_df = pd.DataFrame({'Adj Close': [close_price]}, index=[timestamp])
        self.prev_closes = pd.concat([self.prev_closes, close_df]).iloc[-250:]
        
        # 2. Update all indicators incrementally
        self._update_all_indicators(timestamp, open_p, high, low, close_price, vol)
        
        # 3. Target Volatility
        self.trgt_series = self._get_daily_vol(self.combined_closes)
        current_trgt = self.trgt_series.iloc[-1] if not self.trgt_series.empty else 0.0
        
        # 4. Check CUSUM (FIXED: h uses historical std, not single-bar std)
        h = 2 * self.combined_closes.diff().std()
        if pd.isna(h) or h == 0: 
            return

        # CUSUM requires previous state and the current diff
        cusum_input = self.combined_closes.iloc[-2:].to_frame('Adj Close')
        cusum_input.index.name = 'date'
        
        cusum_events, self.sPos, self.sNeg = cusum_filter_live(
            cusum_input.reset_index(), 'Adj Close', h,
            sPos_prev=self.sPos, sNeg_prev=self.sNeg
        )

        # 5. ML Inference on Event Trigger
        if not cusum_events.empty:
            logger.info(f"[CUSUM EVENT] Triggered at {timestamp}. Firing ML models.")
            
            # Run Live FracDiff just in time
            new_fracdiff = fracDiff_FFD_live(close_df, self.fracdiff_w, self.prev_closes)
            frac_val = new_fracdiff['Adj Close'].iloc[-1] if not new_fracdiff.empty else close_price

            # Build rigid feature vector
            feature_vector = {
                'Adj Close': frac_val,
                'trgt': current_trgt,
                'mad': self.current_features['mad'],
                'ppo': self.current_features['ppo'],
                'adx': self.current_features['adx'],
                'aroon': self.current_features['aroon'],
                'normalized_volume': self.current_features['normalized_volume'],
                'volume_ratio': self.current_features['volume_ratio'],
                'vol_ma': self.current_features['vol_ma'],
                'macd_initial': self.current_features['macd_initial'],
                'macd_normalized': self.current_features['macd_normalized'],
                'atr_log_14': self.current_features['atr_log_14'],
                'entropy_100': self.current_features['entropy_100']
            }

            features_df = pd.DataFrame([feature_vector], index=[timestamp])
            
            # Predict
            side = self.model_side.predict(features_df)[0]
            size = self.model_meta.predict(features_df)[0]
            prob = self.model_meta.predict_proba(features_df)[0, 1]
            
            logger.info(f"[MODEL-RAW] Predict Side: {side} | Meta Size: {size} | Prob: {prob:.2f}")
            
            if side != 0 and size != 0:
                logger.info(f"Trade Signal Generated: Side={side}, Size={size}, Prob={prob:.2f}")
                
                new_bet = pd.DataFrame([{
                    'side': side,
                    'size': side * size,
                    'probs': prob,
                    'trgt': current_trgt,
                    't1': timestamp + pd.Timedelta(days=2),
                    'Adj Close': close_price
                }], index=[timestamp])
                
                if self.active_bets.empty:
                    self.active_bets = new_bet
                else:
                    self.active_bets = pd.concat([self.active_bets, new_bet])

    def _update_all_indicators(self, ts, o, h, l, c, v):
        """Advances state machines for all math indicators for a single bar."""
        macd_df, self.ema_fast, self.ema_slow, self.signal_line = update_macd(
            pd.Series([c], index=[ts]), self.ema_fast, self.ema_slow, self.signal_line
        )
        
        atr_log_val, self.atr_window = update_atr_log_14(o, h, l, c, self.atr_window)
        
        macd_norm_val, self.last_short_sum, self.last_long_sum, self.atr_window_macd = update_macd_normalized(
            c, h, l, self.last_short_sum, self.last_long_sum,
            self.atr_window_macd, self.long_alpha, self.short_alpha
        )
        
        entropy_val, self.entropy_window = update_entropy_100(c, self.entropy_window)
        aroon_val, self.state_aroon = aroon_update(h, l, self.state_aroon, mode="diff", lookback=100)
        mad_val, self.state_mad = mad_normalized_update(c, h, l, self.state_mad)
        vol_ma_val, self.state_vol_ma = volume_weighted_ma_ratio_update(c, v, self.state_vol_ma)
        norm_vol_val, self.state_norm_vol = normalized_volume_index_update(c, v, self.state_norm_vol)
        vol_ratio_val, self.state_vol_ratio = short_long_volume_ratio_indicator_update(v, self.state_vol_ratio)
        ppo_val, self.state_ppo = ppo_update(c, self.state_ppo)
        adx_val, self.state_adx = adx_update(h, l, c, self.state_adx)

        # Cache values
        self.current_features = {
            'mad': mad_val,
            'ppo': ppo_val,
            'adx': adx_val,
            'aroon': aroon_val,
            'normalized_volume': norm_vol_val,
            'volume_ratio': vol_ratio_val,
            'vol_ma': vol_ma_val,
            'macd_initial': macd_df['macd'].iloc[-1],
            'macd_normalized': macd_norm_val,
            'atr_log_14': atr_log_val,
            'entropy_100': entropy_val
        }

    # =========================================================================
    # UTILITIES & LIFECYCLE
    # =========================================================================

    def _clear_expired_bets(self):
        if self.active_bets.empty: return
        
        now_ts = pd.Timestamp.now(tz='UTC')
        self.active_bets = self.active_bets[self.active_bets['t1'] > now_ts]
        
        to_remove = []
        for idx, row in self.active_bets.iterrows():
            entry_price, target = row['Adj Close'], row['trgt']
            if pd.isna(target): continue
            
            try:
                post_entry = self.combined_closes.loc[idx:]
            except KeyError:
                continue
                
            if not post_entry.empty:
                if (post_entry >= entry_price + target).any() or (post_entry <= entry_price - target).any():
                    to_remove.append(idx)
                    
        if to_remove:
            self.active_bets.drop(index=to_remove, inplace=True)

    def _get_daily_vol(self, close_series: pd.Series, span0: int = 168) -> pd.Series:
        if len(close_series) < 2: return pd.Series(dtype=float)
        idx_1d_ago = close_series.index.searchsorted(close_series.index - pd.Timedelta(days=1))
        valid_indices = idx_1d_ago[idx_1d_ago > 0] - 1
        if len(valid_indices) == 0: return pd.Series(dtype=float)
        
        current_prices = close_series.iloc[-len(valid_indices):]
        past_prices = close_series.iloc[valid_indices]
        returns = (current_prices.values / past_prices.values) - 1
        return pd.Series(returns, index=current_prices.index).ewm(span=span0).std()

    def _convert_to_df(self, data: MarketData) -> pd.DataFrame:
        list_candles = data.data.get(self.symbol, [])
        if not list_candles: return pd.DataFrame()
        
        df = pd.DataFrame([c.model_dump() for c in list_candles])
        df.rename(columns={
            'timestamp': 'date', 'close': 'Adj Close', 'open': 'Open',
            'high': 'High', 'low': 'Low', 'volume': 'Volume'
        }, inplace=True)
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df

    def _bootstrap(self, df: pd.DataFrame):
        """Initializes all math states from historical payload."""
        logger.info(f"[BOOTSTRAP] Warming up with {len(df)} rows.")
        
        self.combined_closes = df['Adj Close'].copy().iloc[-self.max_history:]
        self.prev_closes = df[['Adj Close']][-250:].copy()
        _, self.fracdiff_w = fracDiff_FFD_initial(df[['Adj Close']][-238:], d=0.45, thres=1e-4)

        _, e_fast, e_slow, s_line = compute_macd_initial(df['Adj Close'])
        self.ema_fast, self.ema_slow, self.signal_line = float(e_fast.iloc[-1]), float(e_slow.iloc[-1]), float(s_line.iloc[-1])

        for idx in df.tail(14).index:
            self.atr_window.append((df.at[idx,'Open'], df.at[idx,'High'], df.at[idx,'Low'], df.at[idx,'Adj Close']))
        for c in df['Adj Close'].tail(100):
            self.entropy_window.append(c)

        closes = df['Adj Close'].values
        self.last_short_sum = self.last_long_sum = closes[0]
        for price in closes[1:]:
            self.last_short_sum = self.short_alpha * price + (1 - self.short_alpha) * self.last_short_sum
            self.last_long_sum  = self.long_alpha  * price + (1 - self.long_alpha)  * self.last_long_sum

        for i in range(len(df) - 101, len(df)):
            self.atr_window_macd.append((df['High'].iloc[i], df['Low'].iloc[i], df['Adj Close'].iloc[i]))

        _, self.state_aroon = aroon_init(df['High'].values, df['Low'].values, lookback=100)
        _, self.state_mad = mad_normalized(df['Adj Close'].values, df['High'].values, df['Low'].values, 100, 10, 10)
        _, self.state_vol_ma = volume_weighted_ma_ratio(df['Adj Close'].values, df['Volume'].values, lookback=20)
        _, self.state_norm_vol = normalized_volume_index(df['Adj Close'].values, df['Volume'].values)
        _, self.state_vol_ratio = short_long_volume_ratio_indicator(df['Volume'].values)
        _, self.state_ppo = ppo_init(df['Adj Close'].values, 10, 100, 10)
        _, self.state_adx = adx_init(df['High'].values, df['Low'].values, df['Adj Close'].values, 14)

        self.last_timestamp = df.index.max()
        self.is_bootstrapped = True
