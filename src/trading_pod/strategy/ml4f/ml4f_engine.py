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



logger = logging.getLogger("TRADING.ML4F")


class ML4FStrategy(IStrategy):
    """Machine Learning for Finance strategy integrated with the current trading engine."""

    def __init__(self, lookback_window: int = 200):
        self.lookback_window = lookback_window

        self.warmup = True

        logger.info("[ML4F] Loading models...")
        self.model_side = load(open('models/best_first_model_final.pkl', 'rb'))
        self.model_meta = load(open('models/best_meta_model_final.pkl', 'rb'))

        self.time_bars = None # Might only need them for the warmup phase, but keeping as an instance variable for now
        self.last_timestamp = None # Probably wont need it since the engine sends a candle directly

        self.sPos = 0
        self.sNeg = 0

        # NOTE: All the following variables can be obtained from the time bars exclusively

        self.close_initial_df = None # Dependent on the time bars, perhaps only needed in warmup
        self.combined_closes = None # The same as above

        # These will be initialize in the warmup with compute_macd_initial(x)
        self.macd_df = None
        self.ema_fast_series = None
        self.ema_slow_series = None
        self.signal_line_series = None

        # These will be based on the last values of the series above, and will be updated with update_macd(x)
        self.ema_fast = 0
        self.ema_slow = 0
        self.signal_line = 0

        # This is the optimal d for the fracdiff, but it can be optimized further in the future
        self.optimal_d = 0.45
        self.fracdiff_series = None
        self.w = None
        self.previous_closes = None

        # Create and index active bets dataframe to keep track of open positions and their details
        self.active_bets = pd.DataFrame(columns=['side', 'size', 'probs', 'trgt', 't1', 'Adj Close'])
        self.active_bets.index.name = 'date'

       
        self.atr_log_initial = None # This list will come from compute_atr_log_14_initial(x)
        self.atr_window = deque(maxlen=14) # Append values to it during an initial warmup

        self.macd_norm_df = None # Calculated with compute_macd_normalized_initial(x)
        self.macd_norm_series = None
        self.closes = None

        self.short_alpha = 0
        self.long_alpha = 0
        self.last_short_sum = 0
        self.last_long_sum = 0
        self.atr_window_macd = deque(maxlen=14) # This is the same as the atr_window but for the macd normalized
        self.entropy_initial = None
        self.entropy_window = deque(maxlen=100) # This is the window for the entropy calculation, it will be updated with the update_entropy_100(x) function

        # All these derive from the initial calculations above, and will be updated with their respective update functions
        self.macd_norm_list = None
        self.atr_log_list = None
        self.entropy_list = None

        # Relted to the aroon indicator
        self.aroon_values = None
        self.state_aroon = None

        # Related to the mad normalized
        self.mad_values = None
        self.state_mad = None

        # Related to the volume weighted ma ratio
        self.vol_ma_values = None
        self.state_vol_ma = None

        # Related to the normalized volume index
        self.norm_vol_values = None
        self.state_norm_vol = None

        # Related to the volume ratio
        self.vol_ratio_values = None
        self.state_vol_ratio = None

        # Related to the ppo
        self.ppo_values = None
        self.state_ppo = None

        # Related to the adx
        self.adx_values = None
        self.state_adx = None


    def generate_allocations(self, data: MarketData) -> List[TargetAllocation]:
        # NOTE: The first "allocation" will be the warmup, therefore there wont be any actual allocations, just the processing of the incoming data and the update of the features. After the warmup, the strategy will start generating actual allocations based on the model predictions and the features.
        
        symbols = data.symbols()
        if not symbols:
            logger.warning("[ML4F] No symbols provided in market data.")
            return []
        
    
    def _warmup(self, data: MarketData):
        pass