import pandas as pd
from typing import List
from collections import deque
from pickle import load
import os
import sys

from core.datatypes import MarketData, TargetAllocation
from trading_pod.interfaces.IStrategy import IStrategy

# Import the ML team's functions here...
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
    def __init__(self, symbol: str = "BTC/USD"):
        self.symbol = symbol
        
        # Bootstrap Flag
        self.is_bootstrapped = False
        self.last_timestamp = None
        
        # --- BULLETPROOF DOCKER PATH ---
        current_dir = os.path.dirname(__file__)
        model_dir = os.path.join(current_dir, 'models')
        
        # ---> ADD THESE TWO LINES HERE <---
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        self.model_side = load(open(os.path.join(model_dir, 'best_primary_model_all_features.pkl'), 'rb'))
        self.model_meta = load(open(os.path.join(model_dir, 'best_meta_model_all_features.pkl'), 'rb'))
        
        # Persistent State Variables
        self.active_bets = pd.DataFrame(columns=['side', 'size', 'probs', 'trgt', 't1', 'Adj Close'])
        self.active_bets.index.name = "date"
        
        # CUSUM states
        self.sPos, self.sNeg = 0, 0
        
        # Feature states
        # populated during bootstrap
        self.combined_closes = None
        self.prev_closes = None
        self.fracdiff_w = None
        self.ema_fast = None
        self.ema_slow = None
        self.signal_line = None
        
        # Deques for rolling windows
        self.atr_window = deque(maxlen=14)
        self.atr_window_macd = deque(maxlen=101)
        self.entropy_window = deque(maxlen=100)
        
        # Lists to hold the history of features for reindexing
        self.macd_norm_list, self.atr_log_list, self.entropy_list = [], [], []
        
        # EMA states for macd_normalized
        self.short_alpha = 2.0 / (10 + 1)
        self.long_alpha = 2.0 / (100 + 1)
        self.last_short_sum = 0.0
        self.last_long_sum = 0.0
        
        # Native states for features (need to be initialized to None)
        self.state_aroon = None
        self.state_mad = None
        self.state_vol_ma = None
        self.state_norm_vol = None
        self.state_vol_ratio = None
        self.state_ppo = None
        self.state_adx = None

    def _get_daily_vol(self, close_series, span0=168):
        """Calculates dynamic target thresholds."""
        df0 = close_series.index.searchsorted(close_series.index - pd.Timedelta(days=1))
        df0 = df0[df0 > 0] - 1
        df0 = pd.Series(close_series.index[df0], index=close_series.index[-df0.shape[0]:])
        time_vals = pd.to_datetime(df0.values)
        df0 = close_series.loc[df0.index] / close_series.loc[time_vals].values - 1
        return df0.ewm(span=span0).std()

    def _convert_to_df(self, data: MarketData) -> pd.DataFrame:
        """Converts the engine's MarketData to the DataFrame format the ML script expects."""
        candles = data.data.get(self.symbol, [])
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame([c.dict() for c in candles])
        df.rename(columns={'timestamp': 'date', 'close': 'Adj Close', 'open': 'Open', 
                           'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def _bootstrap(self, df: pd.DataFrame):
        """Warms up the indicators using the initial lookback data."""
        print(f"Bootstrapping models with {len(df)} rows...")
        self.combined_closes = df['Adj Close'].copy()
        
        # 1. FracDiff
        optimal_d = 0.45
        fracdiff_series, self.fracdiff_w = fracDiff_FFD_initial(df[['Adj Close']][-238:], d=optimal_d, thres=1e-4)
        self.prev_closes = df[['Adj Close']][-250:].copy()

        # 2. MACD Standard
        macd_df, ema_fast_series, ema_slow_series, signal_line_series = compute_macd_initial(df['Adj Close'])
        self.ema_fast  = float(ema_fast_series.iloc[-1])
        self.ema_slow  = float(ema_slow_series.iloc[-1])
        self.signal_line = float(signal_line_series.iloc[-1])

        # 3. ATR Log 14
        atr_log_initial = compute_atr_log_14_initial(df['Open'], df['High'], df['Low'], df['Adj Close'])
        for idx in df.tail(14).index:
            self.atr_window.append((df.at[idx,'Open'], df.at[idx,'High'], df.at[idx,'Low'], df.at[idx,'Adj Close']))

        # 4. MACD Normalized
        macd_norm_df = compute_macd_normalized_initial(df['Adj Close'], df['Open'], df['High'], df['Low'])
        closes = df['Adj Close'].values
        self.last_short_sum = closes[0]
        self.last_long_sum = closes[0]
        for price in closes[1:]:
            self.last_short_sum = self.short_alpha * price + (1 - self.short_alpha) * self.last_short_sum
            self.last_long_sum  = self.long_alpha  * price + (1 - self.long_alpha)  * self.last_long_sum
            
        for i in range(len(df) - 101, len(df)):
            self.atr_window_macd.append((df['High'].iloc[i], df['Low'].iloc[i], df['Adj Close'].iloc[i]))

        # 5. Entropy 100
        entropy_initial = compute_entropy_100_initial(df['Adj Close'])
        for c in df['Adj Close'].tail(100):
            self.entropy_window.append(c)

        # 6. Initialize Feature Lists
        self.macd_norm_list = list(macd_norm_df['macd_normalized'].values)
        self.atr_log_list   = list(atr_log_initial.values)
        self.entropy_list   = list(entropy_initial.values)

        # 7. Native States (Aroon, MAD, Vol, PPO, ADX)
        aroon_values, self.state_aroon = aroon_init(df['High'].values, df['Low'].values, lookback=100)
        mad_values, self.state_mad = mad_normalized(df['Adj Close'].values, df['High'].values, df['Low'].values, long_length=100, short_length=10, lag=10)
        vol_ma_values, self.state_vol_ma = volume_weighted_ma_ratio(df['Adj Close'].values, df['Volume'].values, lookback=20)
        norm_vol_values, self.state_norm_vol = normalized_volume_index(df['Adj Close'].values, df['Volume'].values)
        vol_ratio_values, self.state_vol_ratio = short_long_volume_ratio_indicator(df['Volume'].values)
        ppo_values, self.state_ppo = ppo_init(df['Adj Close'].values, short_length=10, long_length=100, n_to_smooth=10)
        adx_values, self.state_adx = adx_init(df['High'].values, df['Low'].values, df['Adj Close'].values, lookback=14)

        # 8. List copies
        self.aroon_list = aroon_values[:]
        self.mad_list = mad_values[:]
        self.vol_ma_list = vol_ma_values[:]
        self.norm_vol_list = norm_vol_values[:]
        self.vol_ratio_list = vol_ratio_values[:]
        self.ppo_list = ppo_values[:]
        self.adx_list = adx_values[:]

        self.last_timestamp = df.index.max()
        self.is_bootstrapped = True
        print("Bootstrap complete.")

    def _process_new_data(self, new_data: pd.DataFrame):
        # Calculate threshold based on the window
        h = 2 * self.combined_closes.diff().std()
        
        # LOGGING: See if we are even looking at data
        # print(f"Processing {len(new_data)} new rows. Current CUSUM Threshold (h): {h}")

        cusum_events, self.sPos, self.sNeg = cusum_filter_live(
            new_data.reset_index(), 'Adj Close', h, 
            sPos_prev=self.sPos, sNeg_prev=self.sNeg
        )

        if not cusum_events.empty:
            print(f"!!! CUSUM TRIGGERED at {cusum_events.index} !!!")
            # ... rest of your logic
            cusum_events.set_index('date', inplace=True)
            
            # Optional: Log events to your internal metrics/discord instead of CSV
            # print(f"Detected {len(cusum_events)} CUSUM events.")

        # 2. Update FracDiff (Requires combining with previous history)
        # We process fracdiff in batch as it requires the historical window
        close_new = new_data[['Adj Close']]
        new_fracdiff_series = fracDiff_FFD_live(close_new, self.fracdiff_w, self.prev_closes)
        
        # Update rolling history for the next cycle
        self.combined_closes = pd.concat([self.combined_closes, close_new['Adj Close']])
        self.prev_closes = pd.concat([self.prev_closes, close_new]).groupby(level=0).last().iloc[-250:]

        # 3. Target (Volatility) for the new bars
        trgt_series = self._get_daily_vol(self.combined_closes)

        # 4. Sequentially update stateful indicators to prevent the "Skipped Bar" bug
        feature_rows_for_prediction = []

        for timestamp, row in new_data.iterrows():
            new_open, new_high, new_low = row['Open'], row['High'], row['Low']
            new_close, new_vol = row['Adj Close'], row['Volume']

            # -- MACD Standard --
            # Note: We wrap the single close in a Series to match their function signature
            macd_update_df, self.ema_fast, self.ema_slow, self.signal_line = update_macd(
                pd.Series([new_close]), self.ema_fast, self.ema_slow, self.signal_line
            )
            current_macd = macd_update_df['macd'].iloc[-1]

            # -- Custom Stateful Updates --
            atr_log_val, self.atr_window = update_atr_log_14(
                new_open, new_high, new_low, new_close, self.atr_window
            )
            
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

            # 5. Check if this specific timestamp triggered a CUSUM event
            if not cusum_events.empty and timestamp in cusum_events.index:
                # Build the exact feature vector the model expects
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
                
                # Assign the timestamp as the name/index for the dataframe later
                feature_series = pd.Series(feature_dict, name=timestamp)
                feature_rows_for_prediction.append(feature_series)

        # 6. Execute ML Predictions
        if feature_rows_for_prediction:
            # Convert list of Series to a DataFrame
            features_df = pd.DataFrame(feature_rows_for_prediction)
            features_df.dropna(inplace=True)

            if not features_df.empty:
                # Primary Model (Side)
                side = self.model_side.predict(features_df)
                
                # Meta Model (Size & Probability)
                size = self.model_meta.predict(features_df)
                probs = self.model_meta.predict_proba(features_df)[:, 1]

                predictions = pd.DataFrame({
                    'side': side,
                    'size': size,
                    'probs': probs
                }, index=features_df.index)

                # Filter out zero-predictions
                predictions = predictions[(predictions['side'] != 0) & (predictions['size'] != 0)]

                if not predictions.empty:
                    # Map the raw Close price and target for stop-loss / take-profit tracking
                    predictions['Adj Close'] = cusum_events.loc[predictions.index, 'Adj Close']
                    predictions['trgt'] = features_df['trgt']
                    
                    # Hardcoded from ML script: bets expire in 2 days
                    predictions['t1'] = predictions.index + pd.Timedelta(days=2)

                    # Append to our state tracker
                    self.active_bets = pd.concat([self.active_bets, predictions])
                    # print(f"Generated {len(predictions)} new trade allocations.")

    def _clear_expired_bets(self):
        """Removes bets that hit their time limit, stop loss, or take profit."""
        # 1. Drop time-expired bets (t1 < now)
        self.active_bets = self.active_bets[self.active_bets['t1'] > pd.to_datetime('now')]

        to_remove = []

        # 2. Drop bets that hit Stop Loss (SL) or Take Profit (TP)
        for idx, row in self.active_bets.iterrows():
            entry_time = idx
            entry_price = row['Adj Close']
            target = row['trgt']

            if pd.isna(target):
                continue

            # Look at all prices that happened *after* the bet was placed
            try:
                post_entry = self.combined_closes.loc[entry_time:]
            except KeyError:
                continue

            if post_entry.empty:
                continue

            # Check if the price ever touched the upper or lower boundary
            if (post_entry >= entry_price + target).any() or (post_entry <= entry_price - target).any():
                to_remove.append(idx)

        # Remove the triggered bets from the active dataframe
        if to_remove:
            self.active_bets.drop(index=to_remove, inplace=True)

    def generate_allocations(self, data: MarketData) -> List[TargetAllocation]:
        """The main entry point called by the trading engine every cycle."""
        df = self._convert_to_df(data)
        if df.empty:
            return []

        # 1. Check Bootstrap
        if not self.is_bootstrapped:
            # Ensure the ETL passed enough data for the lookback
            if len(df) < 250:
                print("Not enough data to bootstrap. Waiting for more history.")
                return []
            self._bootstrap(df)
            return [TargetAllocation(symbol=self.symbol, weight=0.0)]

        # 2. Filter for strictly new data
        new_data = df[df.index > self.last_timestamp]
        
        if not new_data.empty:
            self._process_new_data(new_data)
            self.last_timestamp = new_data.index.max()

        # 3. Cleanup and Position Sizing
        self._clear_expired_bets()
        
        new_bet_size = 0.0
        if not self.active_bets.empty:
            probs_sum = self.active_bets['probs'].sum()
            if probs_sum > 0:
                weighted_sizes = self.active_bets['probs'] * self.active_bets['size']
                new_bet_size = weighted_sizes.sum() / probs_sum

        # No short bets
        new_bet_size = max(0.0, new_bet_size)
        
        # Discretize to 0.1 units
        bet_size = (new_bet_size // 0.1) * 0.1

        return [TargetAllocation(symbol=self.symbol, weight=float(bet_size))]