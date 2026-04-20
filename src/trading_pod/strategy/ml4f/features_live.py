import numpy as np
import pandas as pd
import math
from collections import deque
from .chapter2 import entropy          
from .chapter4 import atr_                 


###############################################################
#  HELPER: Normal CDF (in case your original uses math.erf)
###############################################################
def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


###############################################################
#  ========== 1) MACD NORMALIZED (INITIAL VERSION) ==========
###############################################################

def compute_macd_normalized_initial(close_series, open_series, high_series, low_series,
                                    short_length=10, long_length=100, n_to_smooth=1):
    """
    Computes the FULL historical macd_normalized using your original method.
    Used in the initialization phase of live.py.
    """

    closes = close_series.values
    highs = high_series.values
    lows = low_series.values
    opens = open_series.values

    n = len(close_series)
    output = np.zeros(n)

    # EMA smoothing factors
    long_alpha  = 2.0 / (long_length + 1)
    short_alpha = 2.0 / (short_length + 1)

    long_sum  = closes[0]
    short_sum = closes[0]

    output[0] = 0.0

    # compute
    for i in range(1, n):

        long_sum  = long_alpha  * closes[i] + (1.0 - long_alpha)  * long_sum
        short_sum = short_alpha * closes[i] + (1.0 - short_alpha) * short_sum

        diff_centers = 0.5 * (long_length - 1) - 0.5 * (short_length - 1)
        denom = math.sqrt(abs(diff_centers))

        # ATR normalizer
        k = long_length + n_to_smooth
        if k > i:
            k = i

        # compute ATR over last k bars
        tr_list = []
        for j in range(i-k+1, i+1):
            tr = max(
                highs[j] / lows[j],
                highs[j] / closes[j-1],
                closes[j-1] / lows[j]
            )
            tr_list.append(math.log(tr))
        atr_norm = np.mean(tr_list)

        denom *= atr_norm + 1e-15

        macd_raw = (short_sum - long_sum) / denom
        macd_scaled = 100.0 * normal_cdf(macd_raw) - 50.0

        output[i] = macd_scaled

    # Optional: smoothing
    if n_to_smooth > 1:
        alpha = 2.0 / (n_to_smooth + 1)
        smoothed = output[0]
        for i in range(1, n):
            smoothed = alpha * output[i] + (1 - alpha) * smoothed
            output[i] -= smoothed

    # return as a dataframe
    return pd.DataFrame({'macd_normalized': output}, index=close_series.index)


###############################################################
#  ========== 2) MACD NORMALIZED (LIVE VERSION) ==========
###############################################################

def update_macd_normalized(
    new_close, new_high, new_low,
    last_short_sum, last_long_sum,
    atr_window, long_alpha, short_alpha,
    long_length=100, short_length=10
):

    # Update EMAs
    long_sum_new  = long_alpha  * new_close + (1-long_alpha)  * last_long_sum
    short_sum_new = short_alpha * new_close + (1-short_alpha) * last_short_sum

    # Update ATR window (length 101)
    atr_window.append((new_high, new_low, new_close))
    if len(atr_window) > long_length+1:
        atr_window.popleft()

    # Compute ATR
    trs = []
    prev_close = atr_window[0][2]
    for (h,l,c) in list(atr_window)[1:]:
        tr = max(h - l, abs(h-prev_close), abs(l-prev_close))
        trs.append(tr)
        prev_close = c

    atr = np.mean(trs)

    # Compute macd_normalized
    diff = 0.5*(long_length-1) - 0.5*(short_length-1)
    denom = np.sqrt(abs(diff)) * atr + 1e-15
    macd_raw = (short_sum_new - long_sum_new) / denom
    macd_val = 100 * normal_cdf(macd_raw) - 50

    return macd_val, short_sum_new, long_sum_new, atr_window



###############################################################
#  ========== 3) ATR LOG 14 (INITIAL VERSION) ==========
###############################################################

def compute_atr_log_14_initial(open_series, high_series, low_series, close_series):
    """
    Computes full ATR_LOG_14 using your ATR function.
    """
    n = len(close_series)
    atr_vals = np.full(n, np.nan)

    for i in range(14, n):

        atr_vals[i] = atr_(
            1,
            i,
            14,
            open_series.values,
            high_series.values,
            low_series.values,
            close_series.values
        )


    return pd.Series(atr_vals, index=close_series.index, name="atr_log_14")


###############################################################
#  ========== 4) ATR LOG 14 (LIVE VERSION) ==========
###############################################################

def update_atr_log_14(new_open, new_high, new_low, new_close, atr_window):
    """
    ATR live version (log). Keeps last 14 bars.
    """

    atr_window.append((new_open, new_high, new_low, new_close))

    if len(atr_window) > 14:
        atr_window.popleft()

    if len(atr_window) < 14:
        return np.nan, atr_window

    trs = []
    for i in range(1, len(atr_window)):
        _, h, l, c = atr_window[i]
        prev_c = atr_window[i-1][3]
        tr = max(h/l, h/prev_c, prev_c/l)
        trs.append(math.log(tr))

    return np.mean(trs), atr_window



###############################################################
#  ========== 5) ENTROPY 100 (INITIAL VERSION) ==========
###############################################################

def compute_entropy_100_initial(close_series):
    return close_series.rolling(100).apply(entropy)


###############################################################
#  ========== 6) ENTROPY 100 (LIVE VERSION) ==========
###############################################################

def update_entropy_100(new_close, entropy_window):
    """
    Live entropy calculator using a deque of length 100.
    """

    entropy_window.append(new_close)

    if len(entropy_window) < 100:
        return np.nan, entropy_window

    arr = np.array(entropy_window)
    return entropy(arr), entropy_window


# Arron Diff

def aroon_init(high, low, lookback=100, mode = "diff"):

    n = len(high)

    if len(low) != n:
        raise ValueError("high and low must have the same length")
    if lookback <= 0:
        raise ValueError("lookback must be > 0")
    if n == 0:
        return [], 0

    front_bad = lookback
    output = [0.0] * n

    # Set first value to neutral
    output[0] = 50.0 if mode in ("up", "down") else 0.0       # up/down is 0-100, so neutral = 50
                                                              # for diff, neutral = 0 (up=down) 

    for icase in range(1, n):
        imax = imin = None
        xmax = xmin = None

        # Find highest high in window (current bar NOT included in lookback count)
        if mode in ("up", "diff"):
            # initialize the highest high as the current high
            imax = icase
            xmax = high[icase]
            for i in range(icase - 1, icase - lookback - 1, -1):        # -1 at the end means it decreases over the range
                if i < 0:
                    break
                if high[i] > xmax:
                    xmax = high[i]
                    imax = i

        # Find lowest low in window
        if mode in ("down", "diff"):
            # initialize the lowest low as the current low
            imin = icase
            xmin = low[icase]
            for i in range(icase - 1, icase - lookback - 1, -1):
                if i < 0:
                    break
                if low[i] < xmin:
                    xmin = low[i]
                    imin = i

        # Compute requested output
        if mode == "up":
            output[icase] = 100.0 * (lookback - (icase - imax)) / lookback
            # BarsSinceCurrentHigh = current bar id - highest high id 
        elif mode == "down":
            output[icase] = 100.0 * (lookback - (icase - imin)) / lookback
            # BarsSinceCurrentLow = current bar id - lowest low id 
        else:  # mode == "diff"
            max_val = 100.0 * (lookback - (icase - imax)) / lookback
            min_val = 100.0 * (lookback - (icase - imin)) / lookback
            output[icase] = max_val - min_val
    

    state = {
        "icase": len(high) - 1,
        "high_buf": list(high[-(lookback + 1):]),
        "low_buf": list(low[-(lookback + 1):]),
        "xmax": xmax,
        "imax": imax,
        "xmin": xmin,
        "imin": imin,
    }


    return output, state

def find_max(high, lookback, icase):
    start = max(0, icase - lookback)
    xmax = high[icase]
    imax = icase
    for i in range(icase - 1, start - 1, -1):
        if high[i] > xmax:
            xmax = high[i]
            imax = i
    return xmax, imax

def find_min(low, lookback, icase):
    start = max(0, icase - lookback)
    xmin = low[icase]
    imin = icase
    for i in range(icase - 1, start - 1, -1):
        if low[i] < xmin:
            xmin = low[i]
            imin = i
    return xmin, imin

def aroon_update(high_new, low_new, state, mode="diff", lookback=100):
    """
    Online implementation of Aroon 
    Rescans ONLY when the stored extremum expires.
    """

    icase = state["icase"] + 1

    # --- update rolling buffers ---
    state["high_buf"].append(high_new)
    state["low_buf"].append(low_new)

    # Ensure we only analyse tha last 'lookback' samples
    if len(state["high_buf"]) > lookback + 1:
        state["high_buf"].pop(0)
        state["low_buf"].pop(0)

    # window start index
    win_start = icase - (len(state["high_buf"]) - 1)

    xmax = state["xmax"]
    imax = state["imax"]
    xmin = state["xmin"]
    imin = state["imin"]

    # -------- MAX logic --------
    if mode in ("up", "diff"):
        if imax < win_start:
            # expired → rescan window
            xmax = state["high_buf"][0]
            imax = win_start
            for i, v in enumerate(state["high_buf"]):
                if v > xmax:
                    xmax = v
                    imax = win_start + i
        elif high_new >= xmax:
            xmax = high_new
            imax = icase

    # -------- MIN logic --------
    if mode in ("down", "diff"):
        if imin < win_start:
            xmin = state["low_buf"][0]
            imin = win_start
            for i, v in enumerate(state["low_buf"]):
                if v < xmin:
                    xmin = v
                    imin = win_start + i
        elif low_new <= xmin:
            xmin = low_new
            imin = icase

    # -------- compute indicator --------
    if mode == "up":
        val = 100.0 * (lookback - (icase - imax)) / lookback
    elif mode == "down":
        val = 100.0 * (lookback - (icase - imin)) / lookback
    else:
        max_val = 100.0 * (lookback - (icase - imax)) / lookback
        min_val = 100.0 * (lookback - (icase - imin)) / lookback
        val = max_val - min_val

    # -------- store state --------
    state["icase"] = icase
    state["xmax"] = xmax
    state["imax"] = imax
    state["xmin"] = xmin
    state["imin"] = imin

    return val, state


# MAD

def normal_cdf(x):
    """
    Standard normal CDF
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def atr(icase, length, high_prices, low_prices, close_prices):
    """
    Compute a simple ATR over `length` bars ending at `icase` (the most recent bar).
    Assumes icase >= length and we have close[icase - length] for TR.
    """
    start = icase - length + 1
    tr_sum = 0.0
    tr_buf = []

    for i in range(start, icase + 1):
        prev_close = close_prices[i - 1]
        high = high_prices[i]
        low = low_prices[i]

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_buf.append(tr)
        tr_sum += tr
        if len(tr_buf) > length:
            tr_sum -= tr_buf.pop(0)

    return tr_sum / float(length), tr_sum, tr_buf

def mad_normalized(close, high, low, long_length=100, short_length=10, lag=10):
    """
    Close, high, low : price series (lists or arrays of same length)
    long_length  : length of long moving average
    short_length : length of short moving average
    lag          : lag applied to the long-term average

    Returns:
        output : list of oscillator values (undefined portion left as 0.0)
    """
    n = len(close)
    output = [0.0] * n

    # ATR will need one extra case for prior close
    front_bad = long_length + lag

    # We start computing at index = front_bad all the way to the most recent bar (n−1)

    for icase in range(front_bad, n):
        long_sum = 0.0
        short_sum = 0.0

        # ----- Long-term moving average (lagged) -----
        for k in range(icase - long_length + 1, icase + 1):
            long_sum += close[k - lag]
        long_quo = long_sum/float(long_length)

        # ----- Short-term moving average (current) -----
        for k in range(icase - short_length + 1, icase + 1):
            short_sum += close[k]
        short_quo = short_sum/float(short_length)

        # ----- Distance between centers of long and short windows -----
        diff = 0.5 * (long_length - 1.0) + lag
        diff -= 0.5 * (short_length - 1.0)

        # ----- Denominator uses sqrt of center distance and ATR -----
        diff_sqrt = math.sqrt(abs(diff))
        atr_val, tr_sum, tr_buf = atr(icase, long_length + lag, high, low, close)
        denom = diff_sqrt*atr_val  
        denom += 1.0e-60  # avoid division by zero

        # ----- Raw oscillator -----
        val = (short_quo - long_quo) / denom

        # ----- Map via normal CDF to roughly [-50, +50] -----
        output[icase] = 100.0 * normal_cdf(1.5 * val) - 50.0


    close_buf_len = max(short_length, long_length + lag + 1)
    state = {
        "icase": n - 1,
        "front_bad": front_bad,
        "long_length": long_length,
        "short_length": short_length,
        "lag": lag,
        "atr_len": front_bad,
        "diff_sqrt": diff_sqrt + 1.0e-60,
        "sum_short": short_sum,
        "sum_long": long_sum,
        "tr_sum": tr_sum,
        "tr_buf": tr_buf[:],  # last atr_len TRs
        "close_buf": list(close[-close_buf_len:]),
        "last_close": close[-1] if n else None,
    }

    return output, state

def mad_normalized_update(close_new, high_new, low_new, state):
    """
    Online update for mad_normalized that matches
    """
    state["icase"] += 1
    i = state["icase"]

    long_length = state["long_length"]
    short_length = state["short_length"]
    lag = state["lag"]
    atr_len = state["atr_len"]
    diff_sqrt = state["diff_sqrt"]
    front_bad = state["front_bad"]

    # ---- update close buffer (bounded) ----
    close_buf = state["close_buf"]
    close_buf.append(close_new)

    close_buf_len = max(short_length, long_length + lag + 1)
    if len(close_buf) > close_buf_len:
        close_buf.pop(0)

    # helper to fetch close[j] from the tail buffer (j is global index)
    # buffer holds the last len(close_buf) closes ending at index i
    def get_close(j):
        return close_buf[j - i - 1] if j - i - 1 != 0 else close_buf[-1]
        # (we'll avoid calling this for j==i; included for completeness)

    # Easier/safer direct mapping:
    # close_buf[-1] == close[i], close_buf[-2] == close[i-1], etc.
    def close_at(j):
        return close_buf[j - i - 1]  # negative index

    # ---- short MA rolling sum ----
    state["sum_short"] += close_new
    if i - short_length >= 0:
        state["sum_short"] -= close_at(i - short_length)

    # ---- long MA rolling sum on lagged closes ----
    add_idx = i - lag
    rem_idx = i - lag - long_length
    if add_idx >= 0:
        # when add_idx == i, that's close_new, which is close_buf[-1]
        if add_idx == i:
            state["sum_long"] += close_new
        else:
            state["sum_long"] += close_at(add_idx)
    if rem_idx >= 0:
        state["sum_long"] -= close_at(rem_idx)

    # ---- ATR(TR) update exactly like your atr() ----
    prev_close = state["last_close"]
    tr = max(high_new - low_new, abs(high_new - prev_close), abs(low_new - prev_close))

    tr_buf = state["tr_buf"]
    tr_buf.append(tr)
    state["tr_sum"] += tr
    if len(tr_buf) > atr_len:
        state["tr_sum"] -= tr_buf.pop(0)

    state["last_close"] = close_new

    # ---- warmup ----
    if i < front_bad:
        return 0.0, state

    long_ma = state["sum_long"] / float(long_length)
    short_ma = state["sum_short"] / float(short_length)
    atr_val = state["tr_sum"] / float(atr_len)

    denom = diff_sqrt * atr_val + 1.0e-60
    val = (short_ma - long_ma) / denom
    return 100.0 * normal_cdf(1.5 * val) - 50.0, state


# Volume Weighted MA

def volume_weighted_ma_ratio(close, volume, lookback=20):
    """
    Batch VWR + returns state for online updates.
    Output matches your original function.
    """
    close  = np.asarray(close, float)
    volume = np.asarray(volume, float)
    n      = len(close)

    output = [0.0] * n


    # Find first bar with volume > 0
    first_volume = 0
    for i in range(n):
        if volume[i] > 0:
            first_volume = i
            break

    front_bad = (lookback - 1) + first_volume
    front_bad = min(front_bad, n)  # avoid loops if n is tiny

    # init rolling window state by streaming through history once
    q_close = deque()
    q_vol   = deque()
    numer = 0.0
    denom = 0.0
    sum_v = 0.0

    for i in range(n):
        c = float(close[i])
        v = float(volume[i])

        q_close.append(c)
        q_vol.append(v)

        numer += v * c
        denom += c
        sum_v += v

        if len(q_close) > lookback:
            c_old = q_close.popleft()
            v_old = q_vol.popleft()
            numer -= v_old * c_old
            denom -= c_old
            sum_v -= v_old

        if i < front_bad:
            output[i] = 0.0
        else:
            if sum_v > 0.0:
                value = lookback * numer / (sum_v * denom)
                value = 1000.0 * math.log(value) / math.sqrt(lookback)
                value = 100.0 * normal_cdf(value) - 50.0
                output[i] = value
            else:
                output[i] = 0.0

    state = {
        "icase": n - 1,
        "lookback": int(lookback),
        "first_volume": int(first_volume),
        "front_bad": int((lookback - 1) + first_volume),
        "q_close": q_close,
        "q_vol": q_vol,
        "numer": numer,
        "denom": denom,
        "sum_v": sum_v,
    }
    return output, state

def volume_weighted_ma_ratio_update(close_new, volume_new, state):
    """
    Online update for VWR.
    Returns the new indicator value for this bar (0.0 during warmup).
    """
    state["icase"] += 1
    i = state["icase"]

    lookback = state["lookback"]

    c = float(close_new)
    v = float(volume_new)

    # first_volume/front_bad logic (match batch behavior)
    if state["first_volume"] is None:
        # not used in this version; see alternative below
        pass

    if v > 0.0 and i < state["first_volume"]:
        # only possible if you initialized state without history; normally won't happen
        state["first_volume"] = i
        state["front_bad"] = (lookback - 1) + i

    # rolling window update
    q_close = state["q_close"]
    q_vol   = state["q_vol"]

    q_close.append(c)
    q_vol.append(v)

    state["numer"] += v * c
    state["denom"] += c
    state["sum_v"] += v

    if len(q_close) > lookback:
        c_old = q_close.popleft()
        v_old = q_vol.popleft()
        state["numer"] -= v_old * c_old
        state["denom"] -= c_old
        state["sum_v"] -= v_old

    # warmup handling
    if i < state["front_bad"]:
        return 0.0, state

    if state["sum_v"] <= 0.0:
        return 0.0, state

    value = lookback * state["numer"] / (state["sum_v"] * state["denom"])
    value = 1000.0 * math.log(value) / math.sqrt(lookback)

    return 100.0 * normal_cdf(value) - 50.0, state

def _variance(i, vol_length, close):
    if vol_length <= 1:
        return 0.0
    start = i - vol_length + 1
    if start < 1:
        return 0.0

    rets = [math.log(close[t] / close[t - 1]) for t in range(start, i + 1)]
    mean = sum(rets) / vol_length
    var = sum((r - mean) ** 2 for r in rets) / vol_length
    return var


# Normalized Volume

def normalized_volume_index(close, volume, lookback=20, var_num="PVI"):
    # var_num can be "PVI" or "NVI"
    close = list(close)
    volume = list(volume)

    n = len(close)
    if len(volume) != n:
        raise ValueError("close and volume must have the same length")
    if lookback <= 0:
        raise ValueError("lookback must be > 0")
    if n == 0:
        return [], None

    # volatility_length = min(2*lookback, 250)
    vol_len = 2 * lookback
    if vol_len > 250:
        vol_len = 250

    # skip early bars with unknown/zero volume
    first_volume = None
    for i in range(n):
        if volume[i] > 0.0:
            first_volume = i
            break
    if first_volume is None:
        out = [0.0] * n
        # state that will keep returning 0 unless you decide to “start” later
        state = {
            "icase": n - 1,
            "lookback": int(lookback),
            "vol_len": int(vol_len),
            "front_bad": 10**18,
            "want_pvi": (var_num.upper() == "PVI"),
            "ret_buf": deque(),
            "sum_ret": 0.0,
            "sumsq_ret": 0.0,
            "cond_buf": deque(),
            "sum_cond": 0.0,
            "last_close": close[-1],
            "last_vol": volume[-1],
        }
        return out, state

    front_bad = vol_len + first_volume
    out = [0.0] * n
    if front_bad >= n:
        # still return a valid state for later online continuation
        # (we’ll build it by streaming through history anyway)
        pass

    want_pvi = (var_num.upper() == "PVI")

    # Rolling windows:
    # variance over last vol_len returns
    ret_buf = deque()
    sum_ret = 0.0
    sumsq_ret = 0.0

    # conditional sum over last lookback returns (return counted only if condition true)
    cond_buf = deque()   # (r, flag)
    sum_cond = 0.0

    # stream through history; returns exist from i=1 onwards
    last_close = close[0]
    last_vol = volume[0]

    # out[0] stays 0.0 by construction
    for icase in range(1, n):
        c = close[icase]
        v = volume[icase]

        r = math.log(c / last_close)
        flag = (v > last_vol) if want_pvi else (v < last_vol)

        # update variance window
        ret_buf.append(r)
        sum_ret += r
        sumsq_ret += r * r
        if len(ret_buf) > vol_len:
            ro = ret_buf.popleft()
            sum_ret -= ro
            sumsq_ret -= ro * ro

        # update conditional window
        cond_buf.append((r, flag))
        if flag:
            sum_cond += r
        if len(cond_buf) > lookback:
            ro, fo = cond_buf.popleft()
            if fo:
                sum_cond -= ro

        # compute out[icase] if warm
        if icase >= front_bad and len(ret_buf) == vol_len:
            s = sum_cond / math.sqrt(float(lookback))

            denom = math.sqrt(_variance(icase, vol_len, close))
            if denom > 0.0:
                s /= math.sqrt(denom)
                out[icase] = 100.0 * normal_cdf(0.5 * s) - 50.0
            else:
                out[icase] = 0.0
        else:
            out[icase] = 0.0

        last_close = c
        last_vol = v

    state = {
        "icase": n - 1,
        "lookback": int(lookback),
        "vol_len": int(vol_len),
        "front_bad": int(front_bad),
        "want_pvi": bool(want_pvi),

        "ret_buf": ret_buf,
        "sum_ret": float(sum_ret),
        "sumsq_ret": float(sumsq_ret),

        "cond_buf": cond_buf,
        "sum_cond": float(sum_cond),

        "last_close": float(close[-1]),
        "last_vol": float(volume[-1]),
    }

    return out, state

def normalized_volume_index_update(close_new, volume_new, state):
    state["icase"] += 1
    icase = state["icase"]

    lookback = state["lookback"]
    vol_len = state["vol_len"]
    front_bad = state["front_bad"]
    want_pvi = state["want_pvi"]

    last_close = state["last_close"]
    last_vol = state["last_vol"]

    c = float(close_new)
    v = float(volume_new)

    r = math.log(c / last_close)
    flag = (v > last_vol) if want_pvi else (v < last_vol)

    # variance window
    ret_buf = state["ret_buf"]
    ret_buf.append(r)
    state["sum_ret"] += r
    state["sumsq_ret"] += r * r
    if len(ret_buf) > vol_len:
        ro = ret_buf.popleft()
        state["sum_ret"] -= ro
        state["sumsq_ret"] -= ro * ro

    # conditional window
    cond_buf = state["cond_buf"]
    cond_buf.append((r, flag))
    if flag:
        state["sum_cond"] += r
    if len(cond_buf) > lookback:
        ro, fo = cond_buf.popleft()
        if fo:
            state["sum_cond"] -= ro

    state["last_close"] = c
    state["last_vol"] = v

    if icase < front_bad or len(ret_buf) < vol_len:
        return 0.0, state

    s = state["sum_cond"] / math.sqrt(float(lookback))

    mean = state["sum_ret"] / float(vol_len)
    var = (state["sumsq_ret"] / float(vol_len)) - mean * mean
    if var <= 0.0:
        return 0.0, state

    s /= math.sqrt(var)

    return 100.0 * normal_cdf(0.5 * s) - 50.0, state


# Volume Ratio

def short_long_volume_ratio_indicator(volume, short_length=10, mult=5):
    volume = list(volume)

    n = len(volume)
    if n == 0:
        return [], None
    if short_length <= 0:
        raise ValueError("short_length must be > 0")
    if mult <= 0:
        raise ValueError("mult must be > 0")

    long_length = short_length * mult

    # Find first bar with valid (positive) volume
    first_volume = None
    for i in range(n):
        if volume[i] > 0.0:
            first_volume = i
            break
    if first_volume is None:
        out = [0.0] * n
        # state that will keep returning 0.0
        state = {
            "icase": n - 1,
            "short_length": int(short_length),
            "mult": int(mult),
            "long_length": int(long_length),
            "first_volume": 10**18,
            "front_bad": 10**18,
            "denom": math.exp(math.log(float(mult)) / 3.0),
            "q_short": deque(),
            "q_long": deque(),
            "sum_short": 0.0,
            "sum_long": 0.0,
        }
        return out, state

    front_bad = (long_length - 1) + first_volume
    denom = math.exp(math.log(float(mult)) / 3.0)

    out = [0.0] * n

    # Rolling windows
    q_short = deque()
    q_long  = deque()
    sum_short = 0.0
    sum_long  = 0.0

    for icase in range(n):
        v = float(volume[icase])

        # update short window
        q_short.append(v)
        sum_short += v
        if len(q_short) > short_length:
            sum_short -= q_short.popleft()

        # update long window
        q_long.append(v)
        sum_long += v
        if len(q_long) > long_length:
            sum_long -= q_long.popleft()

        if icase < front_bad:
            out[icase] = 0.0
            continue

        short_avg = sum_short / float(short_length)
        long_avg  = sum_long / float(long_length)

        if long_avg > 0.0 and short_avg > 0.0:
            x = math.log((short_avg / long_avg) / denom)
            out[icase] = 100.0 * normal_cdf(3.0 * x) - 50.0
        else:
            out[icase] = 0.0

    state = {
        "icase": n - 1,
        "short_length": int(short_length),
        "mult": int(mult),
        "long_length": int(long_length),
        "first_volume": int(first_volume),
        "front_bad": int(front_bad),
        "denom": float(denom),

        "q_short": q_short,
        "q_long": q_long,
        "sum_short": float(sum_short),
        "sum_long": float(sum_long),
    }

    return out, state

def short_long_volume_ratio_indicator_update(volume_new, state):
    """
    Online update. Returns the new indicator value for this bar (0.0 during warmup).
    Updates `state` in-place.
    """
    state["icase"] += 1
    icase = state["icase"]

    short_length = state["short_length"]
    long_length  = state["long_length"]
    denom        = state["denom"]

    v = float(volume_new)

    # (Optional) If you ever cold-start with first_volume unknown, you can update it here.
    # With a proper training init, front_bad is already correct.

    # update short window
    q_short = state["q_short"]
    q_short.append(v)
    state["sum_short"] += v
    if len(q_short) > short_length:
        state["sum_short"] -= q_short.popleft()

    # update long window
    q_long = state["q_long"]
    q_long.append(v)
    state["sum_long"] += v
    if len(q_long) > long_length:
        state["sum_long"] -= q_long.popleft()

    if icase < state["front_bad"]:
        return 0.0, state

    short_avg = state["sum_short"] / float(short_length)
    long_avg  = state["sum_long"] / float(long_length)

    if long_avg > 0.0 and short_avg > 0.0:
        x = math.log((short_avg / long_avg) / denom)
        return 100.0 * normal_cdf(3.0 * x) - 50.0, state
    
    return 0.0, state


# PPO

def ppo_init(close, short_length=10, long_length=100, n_to_smooth=10):
    """
    Batch PPO (same behavior as your original), plus returns state for online updates.

    Returns:
        output, state
    """
    close = list(close)
    n = len(close)
    if n == 0:
        return [], None

    if short_length <= 0 or long_length <= 0:
        raise ValueError("short_length and long_length must be > 0")
    if n_to_smooth < 0:
        raise ValueError("n_to_smooth must be >= 0")

    short_alpha = 2.0 / (short_length + 1.0)
    long_alpha  = 2.0 / (long_length + 1.0)

    output = [0.0] * n

    short_ema = close[0]
    long_ema  = close[0]
    output[0] = 0.0

    # optional smoothing state (only if n_to_smooth > 1)
    sm_alpha = None
    smoothed = 0.0
    if n_to_smooth > 1:
        sm_alpha = 2.0 / (n_to_smooth + 1.0)
        smoothed = output[0]  # 0.0

    for i in range(1, n):
        c = float(close[i])
        short_ema = short_alpha * c + (1.0 - short_alpha) * short_ema
        long_ema  = long_alpha  * c + (1.0 - long_alpha)  * long_ema

        ppo_val = 0.0
        if long_ema != 0.0:
            ppo_val = 100.0 * (short_ema - long_ema) / long_ema

        if sm_alpha is not None:
            smoothed = sm_alpha * ppo_val + (1.0 - sm_alpha) * smoothed
            output[i] = ppo_val - smoothed
        else:
            output[i] = ppo_val

    state = {
        "icase": n - 1,
        "short_length": int(short_length),
        "long_length": int(long_length),
        "n_to_smooth": int(n_to_smooth),

        "short_alpha": float(short_alpha),
        "long_alpha": float(long_alpha),
        "short_ema": float(short_ema),
        "long_ema": float(long_ema),

        # smoothing (signal) state
        "sm_alpha": float(sm_alpha) if sm_alpha is not None else None,
        "smoothed": float(smoothed),
    }

    return output, state

def ppo_update(close_new, state):
    """
    Online PPO update. Returns the next value (PPO or PPO - smoothed PPO).
    Updates `state` in-place.
    """
    state["icase"] += 1
    c = float(close_new)

    short_ema = state["short_ema"]
    long_ema  = state["long_ema"]
    sa = state["short_alpha"]
    la = state["long_alpha"]

    short_ema = sa * c + (1.0 - sa) * short_ema
    long_ema  = la * c + (1.0 - la) * long_ema

    state["short_ema"] = short_ema
    state["long_ema"]  = long_ema

    ppo_val = 0.0
    if long_ema != 0.0:
        ppo_val = 100.0 * (short_ema - long_ema) / long_ema

    sm_alpha = state["sm_alpha"]
    if sm_alpha is not None:
        smoothed = state["smoothed"]
        smoothed = sm_alpha * ppo_val + (1.0 - sm_alpha) * smoothed
        state["smoothed"] = smoothed
        return ppo_val - smoothed, state

    return ppo_val, state


# ADX

def adx_init(high, low, close, lookback=14):
    n = len(high)

    if not (len(low) == len(close) == n):
        raise ValueError("high, low, close must have the same length")
    if lookback <= 0:
        raise ValueError("lookback must be > 0")
    if n == 0:
        return [], None

    front_bad = 2 * lookback - 1
    output = [0.0] * n
    output[0] = 0.0

    DMSplus = 0.0
    DMSminus = 0.0
    ATR = 0.0
    ADXS = 0.0
    ADX = 0.0

    # ---------------------- Primary initialization ----------------------
    end1 = min(lookback, n - 1)
    for i in range(1, end1 + 1):
        DMplus = high[i] - high[i - 1]
        DMminus = low[i - 1] - low[i]

        if DMplus >= DMminus:
            DMminus = 0.0
        else:
            DMplus = 0.0

        if DMplus < 0.0:
            DMplus = 0.0
        if DMminus < 0.0:
            DMminus = 0.0

        DMSplus += DMplus
        DMSminus += DMminus

        tr = high[i] - low[i]
        if high[i] - close[i - 1] > tr:
            tr = high[i] - close[i - 1]
        if close[i - 1] - low[i] > tr:
            tr = close[i - 1] - low[i]
        ATR += tr

        DIplus = DMSplus / (ATR + 1e-10)
        DIminus = DMSminus / (ATR + 1e-10)
        adx_term = abs(DIplus - DIminus) / (DIplus + DIminus + 1e-10)

        ADXS += adx_term
        output[i] = 100.0 * ADXS

    if n - 1 < lookback:
        state = {
            "icase": n - 1,
            "lookback": lookback,
            "front_bad": front_bad,
            "phase": 0,  # not enough data to be in main loop
            "DMSplus": DMSplus,
            "DMSminus": DMSminus,
            "ATR": ATR,
            "ADX": ADX,
            "ADXS": ADXS,
            "last_high": high[-1],
            "last_low": low[-1],
            "last_close": close[-1],
        }
        return output, state

    # ---------------------- Secondary initialization ----------------------
    end2 = min(2 * lookback - 1, n - 1)
    for i in range(lookback + 1, end2 + 1):
        DMplus = high[i] - high[i - 1]
        DMminus = low[i - 1] - low[i]

        if DMplus >= DMminus:
            DMminus = 0.0
        else:
            DMplus = 0.0

        if DMplus < 0.0:
            DMplus = 0.0
        if DMminus < 0.0:
            DMminus = 0.0

        DMSplus = (lookback - 1.0) / lookback * DMSplus + DMplus
        DMSminus = (lookback - 1.0) / lookback * DMSminus + DMminus

        tr = high[i] - low[i]
        if high[i] - close[i - 1] > tr:
            tr = high[i] - close[i - 1]
        if close[i - 1] - low[i] > tr:
            tr = close[i - 1] - low[i]
        ATR = (lookback - 1.0) / lookback * ATR + tr

        DIplus = DMSplus / (ATR + 1e-10)
        DIminus = DMSminus / (ATR + 1e-10)
        ADXS += abs(DIplus - DIminus) / (DIplus + DIminus + 1e-10)

        output[i] = 100.0 * ADXS / (i - lookback + 1)

    ADX = ADXS / lookback

    if n - 1 < 2 * lookback:
        state = {
            "icase": n - 1,
            "lookback": lookback,
            "front_bad": front_bad,
            "phase": 1,  # initialized but not in main loop yet
            "DMSplus": DMSplus,
            "DMSminus": DMSminus,
            "ATR": ATR,
            "ADX": ADX,
            "ADXS": ADXS,
            "last_high": high[-1],
            "last_low": low[-1],
            "last_close": close[-1],
        }
        return output, state

    # ---------------------- Main loop (after initialization) ----------------------
    for i in range(2 * lookback, n):
        DMplus = high[i] - high[i - 1]
        DMminus = low[i - 1] - low[i]

        if DMplus >= DMminus:
            DMminus = 0.0
        else:
            DMplus = 0.0

        if DMplus < 0.0:
            DMplus = 0.0
        if DMminus < 0.0:
            DMminus = 0.0

        DMSplus = (lookback - 1.0) / lookback * DMSplus + DMplus
        DMSminus = (lookback - 1.0) / lookback * DMSminus + DMminus

        tr = high[i] - low[i]
        if high[i] - close[i - 1] > tr:
            tr = high[i] - close[i - 1]
        if close[i - 1] - low[i] > tr:
            tr = close[i - 1] - low[i]
        ATR = (lookback - 1.0) / lookback * ATR + tr

        DIplus = DMSplus / (ATR + 1e-10)
        DIminus = DMSminus / (ATR + 1e-10)

        dx = abs(DIplus - DIminus) / (DIplus + DIminus + 1e-10)
        ADX = (lookback - 1.0) / lookback * ADX + dx / lookback

        output[i] = 100.0 * ADX

    state = {
        "icase": n - 1,
        "lookback": lookback,
        "front_bad": front_bad,
        "phase": 2,  # steady-state main loop recurrence is valid
        "DMSplus": DMSplus,
        "DMSminus": DMSminus,
        "ATR": ATR,
        "ADX": ADX,
        "ADXS": ADXS,  # not used in phase 2, but kept
        "last_high": high[-1],
        "last_low": low[-1],
        "last_close": close[-1],
    }

    return output, state

def adx_update(high_new, low_new, close_new, state):
    """
    Online update for ADX.
    Returns the new ADX output value for this bar (scaled by 100 like the batch output).
    """
    state["icase"] += 1
    i = state["icase"]
    lookback = state["lookback"]

    last_high = state["last_high"]
    last_low = state["last_low"]
    last_close = state["last_close"]

    # DM+, DM-
    DMplus = high_new - last_high
    DMminus = last_low - low_new

    if DMplus >= DMminus:
        DMminus = 0.0
    else:
        DMplus = 0.0

    if DMplus < 0.0:
        DMplus = 0.0
    if DMminus < 0.0:
        DMminus = 0.0

    # TR
    tr = high_new - low_new
    if high_new - last_close > tr:
        tr = high_new - last_close
    if last_close - low_new > tr:
        tr = last_close - low_new

    # Wilder-style smoothing (matches your code)
    DMSplus = state["DMSplus"]
    DMSminus = state["DMSminus"]
    ATR = state["ATR"]
    ADX = state["ADX"]

    DMSplus = (lookback - 1.0) / lookback * DMSplus + DMplus
    DMSminus = (lookback - 1.0) / lookback * DMSminus + DMminus
    ATR = (lookback - 1.0) / lookback * ATR + tr

    DIplus = DMSplus / (ATR + 1e-10)
    DIminus = DMSminus / (ATR + 1e-10)

    dx = abs(DIplus - DIminus) / (DIplus + DIminus + 1e-10)
    ADX = (lookback - 1.0) / lookback * ADX + dx / lookback

    # store updated state
    state["DMSplus"] = DMSplus
    state["DMSminus"] = DMSminus
    state["ATR"] = ATR
    state["ADX"] = ADX

    state["last_high"] = float(high_new)
    state["last_low"] = float(low_new)
    state["last_close"] = float(close_new)

    # warmup behavior: match batch output = 0.0 before front_bad
    if i < state["front_bad"]:
        return 0.0, state

    return 100.0 * ADX, state


def compute_macd_initial(price, fast=12, slow=26, signal=9):
    """
    Computes MACD line and Signal line.
    - price: pd.Series of prices
    - fast: short EMA window (default 12)
    - slow: long EMA window (default 26)
    - signal: signal line EMA window (default 9)
    Returns: pd.DataFrame with columns ['macd', 'signal', 'hist']
    """
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - signal_line
    macd_df = pd.DataFrame({
        'macd': macd,
        'signal': signal_line,
        'hist': macd_hist}, index=price.index)
    return macd_df, ema_fast, ema_slow, signal_line


def update_macd(new_prices, last_ema_fast, last_ema_slow, last_signal, fast=12, slow=26, signal=9):
    """
    Incrementally computes MACD values for a date-indexed price series.
    Assumes previous EMA and signal values are provided (no internal init).

    Args:
        new_prices (pd.Series): Price series with datetime index.
        last_ema_fast (float): Previous fast EMA.
        last_ema_slow (float): Previous slow EMA.
        last_signal (float): Previous signal EMA.
        fast (int): Fast EMA span (default 12).
        slow (int): Slow EMA span (default 26).
        signal (int): Signal line EMA span (default 9).

    Returns:
        df (pd.DataFrame): DataFrame with datetime index and columns ['macd', 'signal', 'hist'].
        last_ema_fast, last_ema_slow, last_signal: Updated states.
    """

    alpha_fast = 2 / (fast + 1)
    alpha_slow = 2 / (slow + 1)
    alpha_signal = 2 / (signal + 1)

    results = []

    for date, price in new_prices.items():
        last_ema_fast = alpha_fast * price + (1 - alpha_fast) * last_ema_fast
        last_ema_slow = alpha_slow * price + (1 - alpha_slow) * last_ema_slow
        macd = last_ema_fast - last_ema_slow
        last_signal = alpha_signal * macd + (1 - alpha_signal) * last_signal
        hist = macd - last_signal
        results.append((date, macd, last_signal, hist))

    df = pd.DataFrame(results, columns=['date', 'macd', 'signal', 'hist']).set_index('date')
    return df, last_ema_fast, last_ema_slow, last_signal