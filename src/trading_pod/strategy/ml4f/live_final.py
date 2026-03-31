# live_all_features.py

from collections import deque
from pickle import load
import pandas as pd
import numpy as np
import time
import os
from datetime import timedelta
from .chapter_2 import cusum_filter_live
from .chapter_5 import fracDiff_FFD_live, fracDiff_FFD_initial
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

CSV_PATH = "data/btc_hourly_jan18_may25.csv"
NEW_CSV = "data/clean.csv"
CUSUM_EVENTS_PATH = "cusum_events.csv"


def preprocess_csv():
    df = pd.read_csv(CSV_PATH)
    df.rename(columns={'Open time': 'date', 'Close': 'Adj Close'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])

    # Keep OHLCV for all features
    df = df[['date','Open','High','Low','Adj Close','Volume']]
    df.to_csv(NEW_CSV, index=False)
    return df


def process_new_data(last_timestamp):
    df = pd.read_csv(NEW_CSV)
    df['date'] = pd.to_datetime(df['date'])
    if last_timestamp is None:
        return df
    return df[df['date'] > last_timestamp]


def getDailyVol(close, span0=168):  # span of 1 week in hour bars
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0] - 1
    df0 = pd.Series(close.index[df0], index=close.index[-df0.shape[0]:])
    time_vals = pd.to_datetime(df0.values)
    df0 = close.loc[df0.index] / close.loc[time_vals].values - 1
    return df0.ewm(span=span0).std()


def clear_active_bets(active_bets, combined_closes):
    #  Drop expired bets
    active_bets = active_bets[active_bets['t1'] > pd.to_datetime('now')]

    to_remove = deque()

    for idx, row in active_bets.iterrows():
        entry_time = idx
        t1 = row['t1']
        entry_price = row['Adj Close']
        target = row['trgt']

        if pd.isna(target):
            continue

        # Slice price series from entry time to t1
        post_entry = combined_closes.loc[entry_time:t1]['Adj Close']
        if post_entry.empty:
            continue

        # Check if stop-loss or take-profit was hit
        if (post_entry >= entry_price + target).any() or (post_entry <= entry_price - target).any():
            to_remove.append(idx)

    # Drop bets that hit target
    return active_bets.drop(index=to_remove)


def main():

    print("Loading models...")
    model_side = load(open('models/best_primary_model_all_features.pkl', 'rb'))
    model_meta = load(open('models/best_meta_model_all_features.pkl', 'rb'))

    # Load initial data
    time_bars = preprocess_csv()
    last_timestamp = time_bars['date'].max()

    print("Initial rows:", time_bars.shape)
    sPos, sNeg = 0, 0

    close_initial_df = time_bars[['date','Adj Close']].set_index('date')
    combined_closes = close_initial_df.copy()

    

    # MACD initial
    macd_df, ema_fast_series, ema_slow_series, signal_line_series = \
        compute_macd_initial(close_initial_df['Adj Close'])

    ema_fast  = float(ema_fast_series.iloc[-1])
    ema_slow  = float(ema_slow_series.iloc[-1])
    signal_line = float(signal_line_series.iloc[-1])

    # FracDiff initialization
    optimal_d = 0.45
    fracdiff_series, w = fracDiff_FFD_initial(close_initial_df[-238:], d=optimal_d, thres=1e-4)
    prev_closes = close_initial_df[-250:].copy()

    # Active bets initialization
    active_bets = pd.DataFrame(columns=['side', 'size', 'probs', 'trgt', 't1', 'Adj Close'])
    active_bets.index.name = "date"


    # ATR_LOG_14 initialization
    atr_log_initial = compute_atr_log_14_initial(
        time_bars['Open'], time_bars['High'],
        time_bars['Low'], time_bars['Adj Close']
    )
    atr_window = deque(maxlen=14)
    for idx in time_bars.tail(14).index:
        atr_window.append((
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
    short_alpha = 2.0 / (10 + 1)
    long_alpha = 2.0 / (100 + 1)

    last_short_sum = closes[0]
    last_long_sum = closes[0]
    for price in closes[1:]:
        last_short_sum = short_alpha * price + (1 - short_alpha) * last_short_sum
        last_long_sum  = long_alpha  * price + (1 - long_alpha)  * last_long_sum

    # ATR window for macd_norm
    atr_window_macd = deque(maxlen=101)
    for i in range(len(time_bars) - 101, len(time_bars)):
        h = time_bars['High'].iloc[i]
        l = time_bars['Low'].iloc[i]
        c = time_bars['Adj Close'].iloc[i]
        atr_window_macd.append((h,l,c))

    # ENTROPY_100 initial
    entropy_initial = compute_entropy_100_initial(close_initial_df['Adj Close'])
    entropy_window = deque(maxlen=100)
    for c in close_initial_df['Adj Close'].tail(100):
        entropy_window.append(c)


    macd_norm_list = list(macd_norm_series.values)
    atr_log_list   = list(atr_log_initial.values)
    entropy_list   = list(entropy_initial.values)


    # Aroon
    aroon_values, state_aroon = aroon_init(
        time_bars['High'].values,
        time_bars['Low'].values,
        lookback=100
    )

    # MAD
    mad_values, state_mad = mad_normalized(
        time_bars['Adj Close'].values,
        time_bars['High'].values,
        time_bars['Low'].values,
        long_length=100,
        short_length=10,
        lag=10
    )

    # VOLUME WEIGHTED MA
    vol_ma_values, state_vol_ma = volume_weighted_ma_ratio(
        time_bars['Adj Close'].values,
        time_bars['Volume'].values,
        lookback=20
    )

    # NORMALIZED VOLUME INDEX
    norm_vol_values, state_norm_vol = normalized_volume_index(
        time_bars['Adj Close'].values,
        time_bars['Volume'].values
    )

    # VOLUME RATIO
    vol_ratio_values, state_vol_ratio = short_long_volume_ratio_indicator(
        time_bars['Volume'].values
    )

    # PPO
    ppo_values, state_ppo = ppo_init(
        time_bars['Adj Close'].values,
        short_length=10,
        long_length=100,
        n_to_smooth=10
    )

    # ADX
    adx_values, state_adx = adx_init(
        time_bars['High'].values,
        time_bars['Low'].values,
        time_bars['Adj Close'].values,
        lookback=14
    )
    

    aroon_list       = aroon_values[:]        # list copy
    mad_list         = mad_values[:]
    vol_ma_list      = vol_ma_values[:]
    norm_vol_list    = norm_vol_values[:]
    vol_ratio_list   = vol_ratio_values[:]
    ppo_list         = ppo_values[:]
    adx_list         = adx_values[:]

    
    # Live loop
    
    while True:
        print("\nChecking for new data...")
        new_data = process_new_data(last_timestamp)

        if not new_data.empty:

            # CUSUM threshold
            h = 2 * new_data['Adj Close'].diff().std()

            # CUSUM events
            cusum_events, sPos, sNeg = cusum_filter_live(
                new_data, 'Adj Close', h,
                sPos_prev=sPos, sNeg_prev=sNeg
            )

            if not cusum_events.empty:
                # Save CUSUM events
                cusum_events.to_csv(
                    CUSUM_EVENTS_PATH,
                    mode='a',
                    header=not pd.io.common.file_exists(CUSUM_EVENTS_PATH),
                    index=True
                )

            # Update last timestamp
            last_timestamp = new_data['date'].max()
            close_new = new_data[['date','Adj Close']].set_index('date')

            
            # FRACDIFF LIVE UPDATE
            
            new_fracdiff_series = fracDiff_FFD_live(
                close_new, w, prev_closes
            )

            # Append new closes to history
            combined_closes = pd.concat([prev_closes, close_new]).groupby(level=0).last()
            prev_closes = combined_closes.iloc[-250:]

            
            # UPDATE FEATURES
            
            macd_update_df, ema_fast, ema_slow, signal_line = update_macd(
                close_new['Adj Close'], ema_fast, ema_slow, signal_line
            )

            macd_df = macd_update_df
            macd_df.index = close_new.index  # ensure correct timestamp alignment

            trgt = getDailyVol(combined_closes['Adj Close'])
            

            row = new_data.iloc[-1]
            new_open  = row['Open']
            new_high  = row['High']
            new_low   = row['Low']
            new_close = row['Adj Close']
            new_vol   = row['Volume']

            atr_log_value, atr_window = update_atr_log_14(
                new_open, new_high, new_low, new_close, atr_window
            )

            macd_norm_value, last_short_sum, last_long_sum, atr_window_macd = update_macd_normalized(
                new_close=new_close,
                new_high=new_high,
                new_low=new_low,
                last_short_sum=last_short_sum,
                last_long_sum=last_long_sum,
                atr_window=atr_window_macd,
                long_alpha=long_alpha,
                short_alpha=short_alpha
            )

            entropy_value, entropy_window = update_entropy_100(
                new_close, entropy_window
            )

            atr_log_list.append(atr_log_value)
            macd_norm_list.append(macd_norm_value)
            entropy_list.append(entropy_value)
            

            aroon_live, state_aroon = aroon_update(
                new_high, new_low, state_aroon,
                mode="diff", lookback=100
            )
            aroon_list.append(aroon_live)


            mad_live, state_mad = mad_normalized_update(
                new_close, new_high, new_low, state_mad
            )
            mad_list.append(mad_live)

            vol_ma_live, state_vol_ma = volume_weighted_ma_ratio_update(
                new_close, new_vol, state_vol_ma
            )
            vol_ma_list.append(vol_ma_live)

            norm_vol_live, state_norm_vol = normalized_volume_index_update(
                new_close, new_vol, state_norm_vol
            )
            norm_vol_list.append(norm_vol_live)

            vol_ratio_live, state_vol_ratio = short_long_volume_ratio_indicator_update(
                new_vol, state_vol_ratio
            )
            vol_ratio_list.append(vol_ratio_live)

            ppo_live, state_ppo = ppo_update(
                new_close, state_ppo
            )
            ppo_list.append(ppo_live)

            adx_live, state_adx = adx_update(
                new_high, new_low, new_close, state_adx
            )
            adx_list.append(adx_live)

            
            # REINDEX FEATURES TO CUSUM EVENTS
            
            if not cusum_events.empty:

                # Number of events
                n = len(cusum_events.index)

                fracdiff_reindexed = new_fracdiff_series['Adj Close'].reindex(cusum_events.index)
                trgt_reindexed     = trgt.reindex(cusum_events.index)
                macd_initial_reindexed = macd_df['macd'].reindex(cusum_events.index)

                macd_norm_reindexed = pd.Series(macd_norm_list[-n:], index=cusum_events.index)
                atr_log_reindexed   = pd.Series(atr_log_list[-n:], index=cusum_events.index)
                entropy_reindexed   = pd.Series(entropy_list[-n:], index=cusum_events.index)

                aroon_reindexed     = pd.Series(aroon_list[-n:],     index=cusum_events.index)
                mad_reindexed       = pd.Series(mad_list[-n:],       index=cusum_events.index)
                vol_ma_reindexed    = pd.Series(vol_ma_list[-n:],    index=cusum_events.index)
                norm_vol_reindexed  = pd.Series(norm_vol_list[-n:],  index=cusum_events.index)
                vol_ratio_reindexed = pd.Series(vol_ratio_list[-n:], index=cusum_events.index)
                ppo_reindexed       = pd.Series(ppo_list[-n:],       index=cusum_events.index)
                adx_reindexed       = pd.Series(adx_list[-n:],       index=cusum_events.index)
                

                features = pd.DataFrame({
                    'Adj Close': fracdiff_reindexed,
                    'trgt': trgt_reindexed,
                    'mad': mad_reindexed,
                    'ppo': ppo_reindexed,
                    'adx': adx_reindexed,
                    'aroon': aroon_reindexed,
                    'normalized_volume': norm_vol_reindexed,
                    'volume_ratio': vol_ratio_reindexed,
                    'vol_ma': vol_ma_reindexed,
                    'macd_initial': macd_initial_reindexed,
                    'macd_normalized': macd_norm_reindexed,
                    'atr_log_14': atr_log_reindexed,
                    'entropy_100': entropy_reindexed
                })


                # Remove rows with NaN
                features = features.dropna()
                if features.empty:
                    print("No valid rows for prediction.")
                    continue


                # PREDICTIONS
                print("Making predictions...")

                side = model_side.predict(features)
                size = model_meta.predict(features)
                probs = model_meta.predict_proba(features)[:, 1]

                predictions = pd.DataFrame({
                    'side': side,
                    'size': size,
                    'probs': probs
                }, index=features.index)

                # Keep only trades where both models predict non-zero
                predictions = predictions[(predictions['side'] != 0) & (predictions['size'] != 0)]

                if not predictions.empty:
                    # Add necessary fields
                    predictions['Adj Close'] = cusum_events.loc[predictions.index, 'Adj Close']
                    predictions['trgt'] = trgt.reindex(predictions.index)
                    predictions['t1'] = predictions.index + pd.Timedelta(days=2)

                    # Append to active bets
                    active_bets = pd.concat([active_bets, predictions])
                    print(f"Added {len(predictions)} new predictions.")

        else:
            print("No new data found.")

        
        # CLEAR EXPIRED BETS

        print(f"Active bets before clearing: {len(active_bets)}")
        active_bets = clear_active_bets(active_bets, combined_closes)

        
        # POSITION SIZING
        
        if not active_bets.empty:
            probs_sum = active_bets['probs'].sum()
            if probs_sum > 0:
                weighted_sizes = active_bets['probs'] * active_bets['size']
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
        print(f"Bet size: {bet_size}, Active bets: {len(active_bets)}")

        
        # SLEEP AND LOOP AGAIN
        print("Going to sleep...")
        time.sleep(5)



# RUN
if __name__ == "__main__":
    main()


