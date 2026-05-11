from collections import deque
import logging
from pickle import load
from typing import List
import pandas as pd
import numpy as np
import time
import os
from datetime import timedelta
from Chapter_2 import cusum_filter_live
from Chapter_5 import fracDiff_FFD_live, fracDiff_FFD_initial
from core.datatypes import MarketData, TargetAllocation
from features_live import (
    compute_macd_initial, 
    update_macd,
    compute_macd_normalized_initial,
    update_macd_normalized,
    compute_atr_log_14_initial,
    update_atr_log_14,
    compute_entropy_100_initial,
    update_entropy_100,
    aroon_init,
    aroon_update,
    mad_normalized,
    mad_normalized_update,
    volume_weighted_ma_ratio,
    volume_weighted_ma_ratio_update,
    normalized_volume_index,
    normalized_volume_index_update,
    short_long_volume_ratio_indicator,
    short_long_volume_ratio_indicator_update,
    ppo_init,
    ppo_update,
    adx_init,
    adx_update 
)
from trading_pod.interfaces.IStrategy import IStrategy
from trading_pod.strategy.ml4f.ml4f import clear_active_bets 



logger = logging.getLogger("TRADING.ML4F")


class ML4FStrategy(IStrategy):
    """Machine Learning for Finance strategy integrated with the current trading engine."""

    def __init__(self, lookback_window: int = 250):
        self.lookback_window = lookback_window

        self.warmup = True

        logger.info("[ML4F] Loading models...")
        self.model_side = load(open('models/best_first_model_final.pkl', 'rb'))
        self.model_meta = load(open('models/best_meta_model_final.pkl', 'rb'))

        self.last_timestamp = None # Probably wont need it since the engine sends a candle directly

        self.sPos = 0
        self.sNeg = 0

        # NOTE: All the following variables can be obtained from the time bars exclusively

        self.combined_closes = None # The same as above

        # These will be initialize in the warmup with compute_macd_initial(x)
        self.macd_df = None

        # These will be based on the last values of the series above, and will be updated with update_macd(x)
        self.ema_fast = 0
        self.ema_slow = 0
        self.signal_line = 0

        # This is the optimal d for the fracdiff, but it can be optimized further in the future
        self.w = None
        self.prev_closes = None

        # Create and index active bets dataframe to keep track of open positions and their details
        self.active_bets = pd.DataFrame(columns=['side', 'size', 'probs', 'trgt', 't1', 'Adj Close'])
        self.active_bets.index.name = 'date'

       
        self.atr_window = deque(maxlen=14) # Append values to it during an initial warmup

        self.short_alpha = 0
        self.long_alpha = 0
        self.last_short_sum = 0
        self.last_long_sum = 0
        self.atr_window_macd = deque(maxlen=14) # This is the same as the atr_window but for the macd normalized
        self.entropy_window = deque(maxlen=100) # This is the window for the entropy calculation, it will be updated with the update_entropy_100(x) function

        # All these derive from the initial calculations above, and will be updated with their respective update functions
        self.macd_norm_list = None
        self.atr_log_list = None
        self.entropy_list = None

        # Relted to the aroon indicator
        self.aroon_list = None
        self.state_aroon = None

        # Related to the mad normalized
        self.mad_list = None
        self.state_mad = None

        # Related to the volume weighted ma ratio
        self.vol_ma_list = None
        self.state_vol_ma = None

        # Related to the normalized volume index
        self.norm_vol_list = None
        self.state_norm_vol = None

        # Related to the volume ratio
        self.vol_ratio_list = None
        self.state_vol_ratio = None

        # Related to the ppo
        self.ppo_list = None
        self.state_ppo = None

        # Related to the adx
        self.adx_list = None
        self.state_adx = None


    def generate_allocations(self, data: MarketData) -> List[TargetAllocation]:
        # NOTE: The first "allocation" will be the warmup, therefore there wont be any actual allocations, just the processing of the incoming data and the update of the features. After the warmup, the strategy will start generating actual allocations based on the model predictions and the features.
        symbols = data.symbols()
        if not symbols:
            logger.warning("[ML4F] No symbols provided in market data.")
            return []
            
        symbol = symbols[0]
        
        # Build the OHLCV dataframe for the symbol and reset the index to have 'date' as a column
        built_df = self._build_ohlcv_dataframe(data, symbol).reset_index()
        
        if built_df.empty:
            return []
        
        if self.warmup:
            if len(built_df) < self.lookback_window:
                logger.warning(f"[ML4F] Waiting for more data. Have {len(built_df)}, need {self.lookback_window}")
                return []
                
            self._warmup(built_df)
            self.warmup = False
            logger.info("[ML4F] Warmup completed. Starting live trading.")
            return []

        logger.info(f"Active bets before clearing: {len(self.active_bets)}")
        self.active_bets = clear_active_bets(self.active_bets, self.combined_closes)
        
        # TODO: Update features with the new data


        # POSITION SIZING
        
        if not self.active_bets.empty:
            probs_sum = self.active_bets['probs'].sum()
            if probs_sum > 0:
                weighted_sizes = self.active_bets['probs'] * self.active_bets['size']
                new_bet_size = weighted_sizes.sum() / probs_sum
            else:
                new_bet_size = 0
        else:
            new_bet_size = 0

        # No short bets allowed based on meta-labeling assumption
        if new_bet_size < 0:
            new_bet_size = 0

        # Discretization (0.1 units)
        bet_size = (new_bet_size // 0.1) * 0.1
        logger.info(f"Bet size: {bet_size}, Active bets: {len(self.active_bets)}")

        return [TargetAllocation(symbol=symbol, weight=bet_size)]



    def _build_ohlcv_dataframe(self, data: MarketData, symbol: str) -> pd.DataFrame:
        candles = data.data.get(symbol, [])
        if not candles:
            return pd.DataFrame()

        records = [{
            'date': pd.Timestamp(c.timestamp).tz_localize(None),
            'Open': float(c.open),
            'High': float(c.high),
            'Low': float(c.low),
            'Adj Close': float(c.close),
            'Volume': float(c.volume)
        } for c in candles]

        df = pd.DataFrame(records)
        df.set_index('date', inplace=True)
        return df

    def _warmup(self, time_bars: pd.DataFrame):
        
        self.last_timestamp = time_bars['date'].max()

        close_initial_df = time_bars[['date','Adj Close']].set_index('date')
        self.combined_closes = close_initial_df.copy()

        # MACD initial
        self.macd_df, ema_fast_series, ema_slow_series, signal_line_series = \
            compute_macd_initial(close_initial_df['Adj Close'])

        self.ema_fast  = float(ema_fast_series.iloc[-1])
        self.ema_slow  = float(ema_slow_series.iloc[-1])
        self.signal_line = float(signal_line_series.iloc[-1])

        # FracDiff initialization
        optimal_d = 0.45
        fracdiff_series, self.w = fracDiff_FFD_initial(close_initial_df[-238:], d=optimal_d, thres=1e-4)
        self.prev_closes = close_initial_df[-250:].copy()

        # Active bets initialization
        self.active_bets = pd.DataFrame(columns=['side', 'size', 'probs', 'trgt', 't1', 'Adj Close'])
        self.active_bets.index.name = "date"


        # ATR_LOG_14 initialization
        atr_log_initial = compute_atr_log_14_initial(
            time_bars['Open'], time_bars['High'],
            time_bars['Low'], time_bars['Adj Close']
        )
        self.atr_window = deque(maxlen=14)
        for idx in time_bars.tail(14).index:
            self.atr_window.append((
                time_bars.at[idx,'Open'],
                time_bars.at[idx,'High'],
                time_bars.at[idx,'Low'],
                time_bars.at[idx,'Adj Close']
            ))

        # MACD_NORMALIZED initial
        macd_norm_df = compute_macd_normalized_initial(
            close_initial_df['Adj Close'],
            time_bars['Open'],
            time_bars['High'],
            time_bars['Low']
        )
        macd_norm_series = macd_norm_df['macd_normalized']
        closes = close_initial_df['Adj Close'].values

        # EMA states for macd_normalized
        self.short_alpha = 2.0 / (10 + 1)
        self.long_alpha = 2.0 / (100 + 1)

        self.last_short_sum = closes[0]
        self.last_long_sum = closes[0]
        for price in closes[1:]:
            self.last_short_sum = self.short_alpha * price + (1 - self.short_alpha) * self.last_short_sum
            self.last_long_sum  = self.long_alpha  * price + (1 - self.long_alpha)  * self.last_long_sum

        # ATR window for macd_norm
        self.atr_window_macd = deque(maxlen=101)
        for i in range(len(time_bars) - 101, len(time_bars)):
            h = time_bars['High'].iloc[i]
            l = time_bars['Low'].iloc[i]
            c = time_bars['Adj Close'].iloc[i]
            self.atr_window_macd.append((h,l,c))

        # ENTROPY_100 initial
        entropy_initial = compute_entropy_100_initial(close_initial_df['Adj Close'])
        self.entropy_window = deque(maxlen=100)
        for c in close_initial_df['Adj Close'].tail(100):
            self.entropy_window.append(c)


        self.macd_norm_list = list(macd_norm_series.values)
        self.atr_log_list   = list(atr_log_initial.values)
        self.entropy_list   = list(entropy_initial.values)


        # Aroon
        self.aroon_list, self.state_aroon = aroon_init(
            time_bars['High'].values,
            time_bars['Low'].values,
            lookback=100
        )

        # MAD
        self.mad_list, self.state_mad = mad_normalized(
            time_bars['Adj Close'].values,
            time_bars['High'].values,
            time_bars['Low'].values,
            long_length=100,
            short_length=10,
            lag=10
        )

        # VOLUME WEIGHTED MA
        self.vol_ma_list, self.state_vol_ma = volume_weighted_ma_ratio(
            time_bars['Adj Close'].values,
            time_bars['Volume'].values,
            lookback=20
        )

        # NORMALIZED VOLUME INDEX
        self.norm_vol_list, self.state_norm_vol = normalized_volume_index(
            time_bars['Adj Close'].values,
            time_bars['Volume'].values
        )

        # VOLUME RATIO
        self.vol_ratio_list, self.state_vol_ratio = short_long_volume_ratio_indicator(
            time_bars['Volume'].values
        )

        # PPO
        self.ppo_list, self.state_ppo = ppo_init(
            time_bars['Adj Close'].values,
            short_length=10,
            long_length=100,
            n_to_smooth=10
        )

        # ADX
        self.adx_list, self.state_adx = adx_init(
            time_bars['High'].values,
            time_bars['Low'].values,
            time_bars['Adj Close'].values,
            lookback=14
        )
