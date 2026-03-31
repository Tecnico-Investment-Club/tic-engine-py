
import numpy as np

def rsi(close, lookback):
    """
    Computes RSI using exponential smoothing as defined in the book.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)

    output = np.zeros(n)

    front_bad = lookback  # number of undefined initial values
    upsum = 1e-60
    dnsum = 1e-60

    # Initialization
    for icase in range(1, front_bad):
        diff = close[icase] - close[icase - 1]
        if diff > 0.0:
            upsum += diff
        else:
            dnsum -= diff

    upsum /= (lookback - 1)
    dnsum /= (lookback - 1)

    # Main RSI computation
    for icase in range(front_bad, n):
        diff = close[icase] - close[icase - 1]
        if diff > 0.0:
            upsum = ((lookback - 1) * upsum + diff) / lookback
            dnsum *= (lookback - 1.0) / lookback
        else:
            dnsum = ((lookback - 1) * dnsum - diff) / lookback
            upsum *= (lookback - 1.0) / lookback

        output[icase] = 100.0 * upsum / (upsum + dnsum)

    return output



def detrended_rsi(close, short_length, long_length, length):
    """
    Computes detrended RSI based on:
    - short RSI (optionally inverse-logistic transformed when short_length == 2)
    - long RSI
    - regression over 'length' samples
    """
    close = np.asarray(close, dtype=float)
    n = len(close)

    work1 = np.zeros(n)  # short-term RSI (transformed when needed)
    work2 = np.zeros(n)  # long-term RSI
    output = np.zeros(n)

    # -------------------------------
    # Short-term RSI computation
    # -------------------------------
    upsum = 1e-60
    dnsum = 1e-60

    for icase in range(1, short_length):
        diff = close[icase] - close[icase - 1]
        if diff > 0.0:
            upsum += diff
        else:
            dnsum -= diff

    upsum /= (short_length - 1)
    dnsum /= (short_length - 1)

    for icase in range(short_length, n):
        diff = close[icase] - close[icase - 1]
        if diff > 0.0:
            upsum = ((short_length - 1.0) * upsum + diff) / short_length
            dnsum *= (short_length - 1.0) / short_length
        else:
            dnsum = ((short_length - 1.0) * dnsum - diff) / short_length
            upsum *= (short_length - 1.0) / short_length

        work1[icase] = 100.0 * upsum / (upsum + dnsum)

        # Apply inverse logistic transform when short_length == 2
        if short_length == 2:
            val = 1 + 0.00999 * (2 * work1[icase] - 100)
            work1[icase] = -10.0 * np.log(2.0 / val - 1.0)

    # -------------------------------
    # Long-term RSI computation
    # -------------------------------
    upsum = 1e-60
    dnsum = 1e-60

    for icase in range(1, long_length):
        diff = close[icase] - close[icase - 1]
        if diff > 0.0:
            upsum += diff
        else:
            dnsum -= diff

    upsum /= (long_length - 1)
    dnsum /= (long_length - 1)

    for icase in range(long_length, n):
        diff = close[icase] - close[icase - 1]
        if diff > 0.0:
            upsum = ((long_length - 1.0) * upsum + diff) / long_length
            dnsum *= (long_length - 1.0) / long_length
        else:
            dnsum = ((long_length - 1.0) * dnsum - diff) / long_length
            upsum *= (long_length - 1.0) / long_length

        work2[icase] = 100.0 * upsum / (upsum + dnsum)

    # -------------------------------
    # Regression and detrending
    # -------------------------------
    front_bad = long_length + length - 1

    for icase in range(front_bad, n):
        xmean = 0.0
        ymean = 0.0

        # Means over regression window
        for i in range(length):
            k = icase - i
            xmean += work2[k]
            ymean += work1[k]

        xmean /= length
        ymean /= length

        xSS = 0.0
        xy = 0.0

        # Sum squares and cross products
        for i in range(length):
            k = icase - i
            xdiff = work2[k] - xmean
            ydiff = work1[k] - ymean
            xSS += xdiff * xdiff
            xy += xdiff * ydiff

        coef = xy / (xSS + 1e-60)

        xdiff = work2[icase] - xmean
        ydiff = work1[icase] - ymean

        output[icase] = ydiff - coef * xdiff

    return output



def stochastic(high, low, close, lookback, n_to_smooth):
    """
    Computes raw, once-smoothed, or twice-smoothed stochastic values.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)

    output = np.zeros(n)

    front_bad = lookback - 1

    sto_1 = None
    sto_2 = None

    for icase in range(front_bad, n):
        min_val = 1e60
        max_val = -1e60

        # Historical high-low range
        for j in range(lookback):
            if high[icase - j] > max_val:
                max_val = high[icase - j]
            if low[icase - j] < min_val:
                min_val = low[icase - j]

        sto_0 = (close[icase] - min_val) / (max_val - min_val + 1e-60)

        if n_to_smooth == 0:
            output[icase] = 100.0 * sto_0
        else:
            if icase == front_bad:
                sto_1 = sto_0
                output[icase] = 100.0 * sto_0
            else:
                sto_1 = 0.66666667 * sto_1 + 0.33333333 * sto_0

                if n_to_smooth == 1:
                    output[icase] = 100.0 * sto_1
                else:
                    if icase == front_bad + 1:
                        sto_2 = sto_1
                        output[icase] = 100.0 * sto_1
                    else:
                        sto_2 = 0.66666667 * sto_2 + 0.33333333 * sto_1
                        output[icase] = 100.0 * sto_2

    return output



import math

def legendre_3(n):
    """
    Computes the first three Legendre polynomial vectors (rescaled to unit length).
    Returns c1, c2, c3 as numpy arrays of length n.
    """

    c1 = np.zeros(n)
    c2 = np.zeros(n)
    c3 = np.zeros(n)

    # ---------------------------
    # First-order polynomial (linear)
    # ---------------------------
    # Values spaced in [-1, 1]
    sum_sq = 0.0
    for i in range(n):
        c1[i] = 2.0 * i / (n - 1.0) - 1.0
        sum_sq += c1[i] * c1[i]

    # Normalize to unit length
    norm = math.sqrt(sum_sq)
    c1 /= norm

    # ---------------------------
    # Second-order polynomial
    # ---------------------------
    # Uncentered values = c1^2
    tmp = np.zeros(n)
    for i in range(n):
        tmp[i] = c1[i] * c1[i]

    # Centering
    mean = np.mean(tmp)
    tmp -= mean

    # Normalize to unit length
    norm = math.sqrt(np.sum(tmp * tmp))
    c2 = tmp / norm

    # ---------------------------
    # Third-order polynomial
    # ---------------------------
    # Uncentered values = c1^3
    tmp = np.zeros(n)
    for i in range(n):
        tmp[i] = c1[i] * c1[i] * c1[i]

    # Center (theoretically zero, but done for numerical stability)
    mean = np.mean(tmp)
    tmp -= mean

    # Preliminary normalization (needed for the projection removal)
    norm = math.sqrt(np.sum(tmp * tmp))
    tmp /= norm

    # Remove projection onto c1 to enforce orthogonality
    proj = np.dot(c1, tmp)
    tmp -= proj * c1

    # Final normalization to unit length
    norm = math.sqrt(np.sum(tmp * tmp))
    c3 = tmp / norm

    return c1, c2, c3



import math

def atr_(use_log, icase, length, open_prices, high_prices, low_prices, close_prices):
    """
    Compute ATR over `length` bars ending at `icase`.
    Supports both raw ATR (use_log=0) and log ATR (use_log=1).
    """
    start = icase - length + 1
    sum_val = 0.0

    for i in range(start, icase + 1):
        if use_log:
            # Ratios correspond to log-domain true range
            term = high_prices[i] / low_prices[i]
            if high_prices[i] / close_prices[i - 1] > term:
                term = high_prices[i] / close_prices[i - 1]
            if close_prices[i - 1] / low_prices[i] > term:
                term = close_prices[i - 1] / low_prices[i]
            sum_val += math.log(term)

        else:
            # Standard true range in raw price domain
            term = high_prices[i] - low_prices[i]
            if high_prices[i] - close_prices[i - 1] > term:
                term = high_prices[i] - close_prices[i - 1]
            if close_prices[i - 1] - low_prices[i] > term:
                term = close_prices[i - 1] - low_prices[i]
            sum_val += term

    return sum_val / float(length)



def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))



def legendre_trend(close_prices, open_prices, high_prices, low_prices,
                   lookback, atr_length, var_num):
    """
    Computes linear, quadratic, or cubic Legendre-based trend indicator.
    var_num = 1 (linear), 2 (quadratic), 3 (cubic)
    """

    close_prices = np.asarray(close_prices, float)
    open_prices  = np.asarray(open_prices, float)
    high_prices  = np.asarray(high_prices, float)
    low_prices   = np.asarray(low_prices, float)

    n = len(close_prices)
    output = np.zeros(n)

    # Undefined initial portion
    front_bad = max(lookback - 1, atr_length)

    # Precompute polynomials
    c1, c2, c3 = legendre_3(lookback)

    for icase in range(front_bad, n):

        # Select correct polynomial
        if var_num == 1:
            dptr = c1
        elif var_num == 2:
            dptr = c2
        else:
            dptr = c3

        # ---- Dot product and mean(log price) over lookback window ----
        start = icase - lookback + 1
        end   = icase

        dot_prod = 0.0
        sum_log  = 0.0

        for coef, idx in zip(dptr, range(start, end + 1)):
            val = math.log(close_prices[idx])
            sum_log += val
            dot_prod += val * coef

        mean_log = sum_log / lookback

        # ---- ATR benchmark denominator (Equation 4.9) ----
        k = lookback - 1
        if lookback == 2:
            k = 2  # exact adjustment

        denom = atr(1, icase, atr_length,
                     open_prices, high_prices, low_prices, close_prices) * k

        raw_trend = (dot_prod * 2.0) / (denom + 1e-60)

        # ---- Compute R-square for devaluation ----
        yss = 0.0
        rss = 0.0

        for coef, idx in zip(dptr, range(start, end + 1)):
            true_y = math.log(close_prices[idx]) - mean_log
            yss += true_y * true_y

            pred_y = dot_prod * coef
            diff = true_y - pred_y
            rss += diff * diff

        rsq = 1.0 - rss / (yss + 1e-60)

        # ---- Final trend value ----
        val = raw_trend * rsq
        val = 100.0 * normal_cdf(val) - 50.0

        output[icase] = val

    return output

