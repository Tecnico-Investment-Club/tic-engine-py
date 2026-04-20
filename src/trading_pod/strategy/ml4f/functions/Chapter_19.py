########################################################
# Chapter 19: Microstructural Features
########################################################

import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from scipy.stats import poisson
from scipy.optimize import minimize

########################################################
# First Generation
########################################################

'''
The first generation of microstructural models concerned themselves with estimating
the bid-ask spread and volatility as proxies for illiquidity. They did so with limited
data and without imposing a strategic or sequential structure to the trading process.
'''

def tick_rule(df, price_col='close', volume_col='volume', tick_rule_col='tick_rule'):
    """
    Calculate the tick rule for each row in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing price and volume data.
    price_col (str): Column name for prices.
    volume_col (str): Column name for volumes.
    tick_rule_col (str): Column name for the output tick rule.
    
    Returns:
    pd.DataFrame: DataFrame with the tick rule column added.
    """
    
    # Step 1: Calculate the price difference
    price_diff = df[price_col].diff()

    # Step 2: Initialize the tick rule column with zeros
    df[tick_rule_col] = 0

    # Step 3: Assign 1 for upward price movements
    df.loc[price_diff > 0, tick_rule_col] = 1

    # Step 4: Assign -1 for downward price movements
    df.loc[price_diff < 0, tick_rule_col] = -1
    
    return df

def roll_model(price_series):
    """
    Implements the Roll model to estimate the bid-ask spread and the variance of true price changes.

    Parameters:
    price_series (pd.Series): A series of observed prices {pt}.

    Returns:
    dict: A dictionary containing the bid-ask spread ('c') and the variance of true price changes ('sigma_u_squared').
    """
    # Step 1: Calculate price changes (delta p_t)
    price_changes = price_series.diff().dropna()

    # Step 2: Calculate the variance of price changes (sigma_quadrado de delta p_t)
    variance_price_changes = np.var(price_changes, ddof=1)

    # Step 3: Calculate the serial covariance of price changes (sigma de delta p_t-1, delta p_t)
    serial_covariance = np.cov(price_changes[:-1], price_changes[1:])[0, 1]

    # Step 4: Estimate the bid-ask spread (c)
    bid_ask_spread = np.sqrt(max(0, -serial_covariance))

    # Step 5: Estimate the true(unobserved) price's noise, excluding microstructural noise (sigma quadrado de u)
    sigma_u_squared = variance_price_changes + 2 * serial_covariance

    return {
        'c': bid_ask_spread,
        'sigma_u_squared': sigma_u_squared
    }

def parkinson_high_low_volatility_k2(df: pd.DataFrame, high_col='high', low_col='low', window: int = None) -> pd.Series:
    """
    Calculate volatility using the high-low price ratio under Geometric Brownian Motion assumptions.
    
    For a GBM with volatility sigma_HL, the expected value of (log(high/low))² is approximately 4*log(2)*sigma_HL^2
    This function implements this relationship to estimate volatility.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'high' and 'low' price columns.
    high_col : str
        Column name for the high price.
    low_col : str
        Column name for the low price.
    window : int, optional
        If provided, calculates rolling volatility over the specified window.
        If None, returns point estimates.
        
    Returns
    -------
    pd.Series
        Estimated volatility for each bar using the GBM high-low estimator.
        If window is provided, returns rolling volatility estimates.
    """
    # Calculate the squared log high-low range
    log_hl_ratio = np.log(df[high_col] / df[low_col])
    squared_log_hl = log_hl_ratio ** 2
    
    
    k2 = np.sqrt(8/math.pi)
    
    if window is None:
        # Point estimates
        volatility = np.sqrt(squared_log_hl / k2)
    else:
        # Rolling estimates
        volatility = log_hl_ratio.rolling(window=window).mean() / k2
    
    return volatility

def parkinson_high_low_volatility_k1(df: pd.DataFrame, high_col='high', low_col='low', window: int = None) -> pd.Series:
    """
    Calculate volatility using the high-low price ratio under Geometric Brownian Motion assumptions.
    
    For a GBM with volatility sigma_HL, the expected value of (log(high/low))² is approximately 4*log(2)*sigma_HL^2
    This function implements this relationship to estimate volatility.
    """
    log_hl_ratio = np.log(df[high_col] / df[low_col])
    squared_log_hl = log_hl_ratio ** 2
    
    # The theoretical constant for GBM is 4*log(2)
    k1 = 4 * np.log(2)
    
    if window is None:
        volatility = np.sqrt(squared_log_hl / k1)
    else:
        volatility = np.sqrt(squared_log_hl.rolling(window=window).mean() / k1)
    
    return volatility


# The next 4 functions are for the Corwin-Schultz algorithm

def getBeta(series,sl):

    hl=series[['High','Low']].values
    hl=np.log(hl[:,0]/hl[:,1])**2
    hl=pd.Series(hl,index=series.index)
    beta=pd.stats.moments.rolling_sum(hl,window=2)
    beta=pd.stats.moments.rolling_mean(beta,window=sl)
    return beta.dropna()
#———————————————————————————————————————-
def getGamma(series):
    h2=pd.stats.moments.rolling_max(series['High'],window=2)
    l2=pd.stats.moments.rolling_min(series['Low'],window=2)
    gamma=np.log(h2.values/l2.values)**2
    gamma=pd.Series(gamma,index=h2.index)
    return gamma.dropna()
#———————————————————————————————————————-
def getAlpha(beta,gamma):
    den=3-2*2**.5
    alpha=(2**.5-1)*(beta**.5)/den
    alpha-=(gamma/den)**.5
    alpha[alpha<0]=0 # set negative alphas to 0 (see p.727 of paper)
    return alpha.dropna()
#———————————————————————————————————————-
def corwinSchultz(series,sl=1):
    # Note: S<0 iif alpha<0
    beta=getBeta(series,sl)
    gamma=getGamma(series)
    alpha=getAlpha(beta,gamma)
    spread=2*(np.exp(alpha)-1)/(1+np.exp(alpha))
    startTime=pd.Series(series.index[0:spread.shape[0]],index=spread.index)
    spread=pd.concat([spread,startTime],axis=1)
    spread.columns=['Spread','Start_Time'] # 1st loc used to compute beta
    return spread


# Although the algorithm does not give us the volatility, we can calculate it from beta and gamma (using Parkinson's k2)
def getSigma(beta, gamma):
    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2 ** 0.5
    sigma = (2**-.5 - 1) * beta**0.5 / (k2 * den)
    sigma += (gamma / (k2**2 * den))**0.5
    sigma[sigma < 0] = 0
    return sigma


########################################################
# Second Generation
########################################################

'''
Second generation microstructural models focus on understanding and measuring
illiquidity. Illiquidity is an important informative feature in financial ML models,
because it is a risk that has an associated premium. These models have a stronger
theoretical foundation than first-generation models, in that they explain trading as
the strategic interaction between informed and uninformed traders. In doing so,they
pay attention to signed volume and order flow imbalance.
'''

def kyles_lambda(df: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume',
                 tick_rule_col: str = 'tick_rule', window: int = None) -> pd.Series | float:
    """
    Compute Kyle's Lambda via linear regression:
    Δpₜ = λ . (bₜ . Vₜ) + εₜ
    """
    # Step 1: Compute price change
    price_changes = df[price_col].diff()

    # Step 2: Make sure tick rule exists
    if tick_rule_col not in df.columns:
        raise ValueError("Tick rule column not found. Provide or compute tick_rule_col.")

    # Step 3: Compute signed volume
    signed_volume = df[volume_col] * df[tick_rule_col]

    # Step 4: Drop NaNs
    df_clean = pd.DataFrame({
        'price_changes': price_changes,
        'signed_volume': signed_volume
    }).dropna()

    if window is None:
        # Regress price_changes ~ signed_volume
        X = df_clean['signed_volume'].values.reshape(-1, 1)
        y = df_clean['price_changes'].values
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        return reg.coef_[0]
    else:
        # Rolling version
        lambdas = []
        index = []

        for i in range(window, len(df_clean) + 1):
            window_data = df_clean.iloc[i - window:i]
            X = window_data['signed_volume'].values.reshape(-1, 1)
            y = window_data['price_changes'].values

            if np.var(X) == 0:
                lambdas.append(np.nan)
            else:
                reg = LinearRegression(fit_intercept=False).fit(X, y)
                lambdas.append(reg.coef_[0])
                index.append(df_clean.index[i - 1])

        return pd.Series(lambdas, index=index)

def kyles_lambda_no_regression(df: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume', 
                tick_rule_col: str = 'tick_rule', window: int = None) -> pd.Series:
    """
    Calculate Kyle's Lambda, a measure of market impact or price impact.
    
    Kyle's Lambda (λ) measures how much the price moves in response to order flow.
    It is calculated as the ratio of the covariance between price changes and signed volume
    to the variance of signed volume.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price and volume data
    price_col : str
        Column name for prices
    volume_col : str
        Column name for volumes
    tick_rule_col : str
        Column name for the tick rule (used to sign the volume)
    window : int, optional
        If provided, calculates rolling Kyle's Lambda over the specified window.
        If None, returns a single value for the entire series.
        
    Returns
    -------
    pd.Series or float
        Kyle's Lambda values. If window is provided, returns rolling estimates.
        If window is None, returns a single value for the entire series.
    """
    # Calculate price changes
    price_changes = df[price_col].diff().dropna()
    
    # Calculate signed volume (using tick rule to determine sign)
    if tick_rule_col not in df.columns:
        df = tick_rule(df, price_col, volume_col, tick_rule_col)
    
    signed_volume = df[volume_col] * df[tick_rule_col]
    signed_volume = signed_volume.dropna()
    
    # Align the series
    price_changes = price_changes[signed_volume.index]
    
    if window is None:
        # Calculate single value for entire series
        cov = np.cov(price_changes, signed_volume)[0,1]
        var = np.var(signed_volume, ddof=1)
        return cov / var if var != 0 else np.nan
    else:
        # Calculate rolling values
        lambda_values = []
        for i in range(window, len(price_changes) + 1):
            window_price_changes = price_changes[i-window:i]
            window_signed_volume = signed_volume[i-window:i]
            
            cov = np.cov(window_price_changes, window_signed_volume)[0,1]
            var = np.var(window_signed_volume, ddof=1)
            
            lambda_values.append(cov / var if var != 0 else np.nan)
            
        return pd.Series(lambda_values, index=price_changes.index[window-1:])

def amihud_lambda(df, price_col='price', volume_col='volume'):
    """
    Computes Amihud's Lambda using sklearn's LinearRegression.
    
    Parameters:
        df (pd.DataFrame): DataFrame with at least 'price' and 'volume' columns.
        price_col (str): Name of the price column.
        volume_col (str): Name of the volume column.

    Returns:
        dict: {
            'amihud_lambda': float,
            'r_squared': float,
            'intercept': float
        }
    """
    df = df.copy()
    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    df['abs_log_return'] = df['log_return'].abs()
    df['dollar_volume'] = df[price_col] * df[volume_col]
    df = df.dropna()

    if df.empty or df['dollar_volume'].sum() == 0:
        return {'amihud_lambda': np.nan, 'r_squared': np.nan, 'intercept': np.nan}

    if df.empty or (df['dollar_volume'] == 0).all():
        return np.nan

    amihud = (df['abs_log_return'] / df['dollar_volume']).mean()
    return amihud

def rolling_amihud_lambda(df, window, price_col='price', volume_col='volume'):
    """
    Computes rolling Amihud's Lambda using a moving window and sklearn.
    
    Returns:
        pd.Series of lambda values.
    """
    lambdas = []
    for i in range(len(df)):
        if i < window:
            lambdas.append(np.nan)
        else:
            window_df = df.iloc[i - window + 1: i + 1]
            lambda_ = amihud_lambda(window_df, price_col, volume_col)
            lambdas.append(lambda_)

    return pd.Series(lambdas, index=df.index, name='amihud_lambda')

def hasbroucks_lambda(df, price_col='price', volume_col='volume', tick_rule_col='tick_rule', window=None):
    """
    Compute Hasbrouck's Lambda: price impact per unit of signed sqrt dollar volume.
    
    Parameters:
    - df: pd.DataFrame with price, volume, tick_rule
    - window: int (rolling window size) or None for full regression

    Returns:
    - float (if window=None) or pd.Series (if window specified)
    """
    df = df.copy()
    
    # Step 1: Log return
    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    
    # Step 2: Signed sqrt dollar volume
    df['signed_sqrt_dollar_volume'] = df[tick_rule_col] * np.sqrt(df[price_col] * df[volume_col])
    
    # Step 3: Drop NaNs
    df.dropna(subset=['log_return', 'signed_sqrt_dollar_volume'], inplace=True)

    if window is None:
        X = df[['signed_sqrt_dollar_volume']].values
        y = df['log_return'].values
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        return reg.coef_[0]
    
    else:
        lambdas = []
        index = []
        for i in range(window, len(df) + 1):
            window_df = df.iloc[i-window:i]
            X = window_df[['signed_sqrt_dollar_volume']].values
            y = window_df['log_return'].values
            if np.var(X) == 0:
                lambdas.append(np.nan)
            else:
                reg = LinearRegression(fit_intercept=False).fit(X, y)
                lambdas.append(reg.coef_[0])
            index.append(df.index[i-1])
        
        return pd.Series(lambdas, index=index)

########################################################
# Third Generation
########################################################

'''
This section contrasts strategic trade models with sequential trade models, where traders arrive randomly and independently over time. 
These models are popular with market makers because they reflect real-world uncertainties—like the chance of new info emerging, whether it's bad news, 
and the rates at which noise and informed traders show up. Market makers use these factors to adjust prices and manage inventory dynamically.
'''
# to compute PIN, i implemented the standard version of likelihood as described by the book; but we can use log_likelihood because:
# 1- It's numerically more stable (avoids underflow from multiplying small probabilities); 2- It's standard in MLE procedures.

'''
from scipy.special import gammaln
# Helper: Poisson log PMF (more stable than using scipy.stats.poisson directly)
def log_poisson(k, lam):
    return k * np.log(lam) - lam - gammaln(k + 1)

# Log-likelihood function
def neg_log_likelihood(params, buys, sells):
    alpha, mu, epsilon = params
    
    if not (0 < alpha < 1 and mu > 0 and epsilon > 0):
        return np.inf  # penalize invalid params

    log_likelihood = 0
    for B, S in zip(buys, sells):
        # Noise only
        ll_noise = np.exp(log_poisson(B, epsilon) + log_poisson(S, epsilon))
        
        # Informed day, info is good
        ll_info_good = np.exp(log_poisson(B, mu + epsilon) + log_poisson(S, epsilon))
        
        # Informed day, info is bad
        ll_info_bad = np.exp(log_poisson(B, epsilon) + log_poisson(S, mu + epsilon))
        
        # Total likelihood
        ll_total = alpha * 0.5 * (ll_info_good + ll_info_bad) + (1 - alpha) * ll_noise
        
        if ll_total <= 0:
            return np.inf  # avoid log(0)
        
        log_likelihood += np.log(ll_total)

    return -log_likelihood  # minimize negative log-likelihood 
'''

# 19.5.1
def compute_pin(buys, sells):
    def likelihood(params, buys, sells): 
        """
        Computes the (negative) likelihood for a sequential trade model with informed and noise traders.

        Parameters
        ----------
        params : tuple or list of floats
            Model parameters (alpha, mu, epsilon):
                alpha   : Probability that a trader is informed (0 < alpha < 1).
                mu      : Mean of the informed trader's order flow (mu > 0).
                epsilon : Mean of the noise trader's order flow (epsilon > 0).
        buys : iterable of int
            Observed buy order counts for each period. - can take this from tick rule
        sells : iterable of int
            Observed sell order counts for each period. - can take this from tick rule

        Returns
        -------
        float
            Negative total likelihood (for minimization). Returns np.inf for invalid parameters or likelihoods.
        
        Notes
        -----
        - The likelihood is computed as a mixture of informed and noise trader models.
        - Penalizes invalid parameter values or zero likelihoods by returning np.inf.
        - Suitable for use in optimization routines that minimize the negative likelihood.
        """
        alpha, mu, epsilon = params

        # Invalid params are penalized
        if not (0 < alpha < 1 and mu > 0 and epsilon > 0):
            return np.inf

        total_likelihood = 1.0
        for B, S in zip(buys, sells):
            # Noise component
            noise = poisson.pmf(B, epsilon) * poisson.pmf(S, epsilon)

            # Informed component (avg of good and bad news days)
            informed_good = poisson.pmf(B, mu + epsilon) * poisson.pmf(S, epsilon)
            informed_bad  = poisson.pmf(B, epsilon) * poisson.pmf(S, mu + epsilon)
            informed = 0.5 * (informed_good + informed_bad)

            # Total likelihood for the day
            L = alpha * informed + (1 - alpha) * noise

            # If any term is 0, the product will be 0 (bad), so penalize
            if L <= 0:
                return np.inf
            
            total_likelihood *= L

        # We're minimizing, so return negative likelihood (to maximize likelihood, minimize neg likelihood)
        return -total_likelihood

    x0 = [0.5, 10, 10]
    bounds = [(1e-5, 1 - 1e-5), (1e-5, None), (1e-5, None)]

    res = minimize(likelihood, x0, args=(buys, sells), bounds=bounds, method='L-BFGS-B')

    if res.success:
        alpha, mu, epsilon = res.x
        pin = (alpha * mu) / (alpha * mu + 2 * epsilon)
        return pin
    else:
        raise RuntimeError(f"Optimization failed: {res.message}")

# 19.5.2 (execute by this order on the pipeline)
def create_volume_buckets(df, target_volume, volume_col='Volume BTC'):
    df = df.copy()
    df['bucket_id'] = -1
    current_volume = 0.0
    bucket_id = 0

    for idx, row in df.iterrows():
        vol = row[volume_col]
        if current_volume + vol > target_volume:
            bucket_id += 1
            current_volume = 0.0
        current_volume += vol
        df.at[idx, 'bucket_id'] = bucket_id

    return df

def tick_rule_vpin(df, base_tick_col='tick_rule', new_col='tick_rule_vpin'): # this assumes a df where tick rule was applied !
    """
    Forward-fills zero tick directions into a VPIN-specific tick rule column.
    """
    df = df.copy()
    df[new_col] = df[base_tick_col].replace(0, method='ffill').fillna(1)
    return df

def estimate_trade_volume(df, volume_col='Volume BTC', tick_col='tick_rule_vpin'):
    df = df.copy()
    df['buy_volume'] = df[volume_col] * (df[tick_col] == 1)
    df['sell_volume'] = df[volume_col] * (df[tick_col] == -1)
    return df

def compute_ofi(df, volume_col='Volume BTC'):
    """
    Computes Order Flow Imbalance (OFI) per volume bucket.
    Assumes buy/sell volume columns are already in the DataFrame.
    """
    bucket_stats = df.groupby('bucket_id').apply(
        lambda x: pd.Series({
            'buy_volume': x['buy_volume'].sum(),
            'sell_volume': x['sell_volume'].sum(),
            'total_volume': x[volume_col].sum()
        })
    )
    bucket_stats['ofi'] = (bucket_stats['buy_volume'] - bucket_stats['sell_volume']).abs() / bucket_stats['total_volume']
    return bucket_stats

def compute_vpin(bucket_stats, window=50):
    bucket_stats = bucket_stats.copy()
    bucket_stats['vpin'] = bucket_stats['ofi'].rolling(window=window).mean()
    return bucket_stats

# 19.6 -  ADDITIONAL FEATURES FROM MICROSTRUCTURAL
'''
Here Lopez de Prado goes crazy and basically says f* the theory: there are alternative features that "we suspect carry important information about 
the way market participants operate, and their future intentions"
Also, almost the whole section is based on papers of David Easley so we might want to check this guy
'''

# 19.6.1 Distibution of Order Sizes
'''
(didnt implement this because we dont have the size of the trades, but here's the ideia: )
if the sizes are "round" (ie, 10, 50, 100, 150, 200) -> lots of uninformed traders -> price will probably follow a trend
else, lots of informed traders -> price moves sideways
'''

# 19.6.2 Cancellation Rates, Limit Orders, Market Orders
'''
(didnt implement this because we dont have the order book, but the idea is: )
traders can use predator algorithms to: slow competitors algorithms, force other to chase a price they dont want to, 
drain the maximum of liquidity from a trader who's undwinding his position, form packs and work together to trigger a desired effect
'''

# 19.6.3 Time-Weighted Average Price Execution Algorithms
'''
(didnt implement this algorithm because we are not analysing competitors, but the idea is: )
recognize the presence of execution algorithms that target a particular time-weighted average price
'''

# 19.6.4 Options Markets
'''
(didnt implement implement this because we dont have options data nor quotes, but the idea is: )

extract information from the options market that may not be reflected in spot prices. This includes comparing the implied bid-ask range from put-call parity
to the actual bid-ask range, analyzing patterns in the Greeks, volatility spreads, and the implied distribution of future prices derived from options.
'''

# 19.6.5 - i believe this is just doing signed volume. Sounds simple i my interpretation was correct
def compute_signed_orderflow_autocorr(df, volume_col='Volume BTC', tick_col='tick_rule', lag=1):
    """
    Computes the autocorrelation of signed order flow (signed volume)
    over a given lag (default: 1).

    Parameters:
    - df: pd.DataFrame with tick rule and volume columns
    - volume_col: name of the volume column
    - tick_col: name of the tick rule column (must be -1 or 1)
    - lag: time lag over which to compute autocorrelation

    Returns:
    - float: autocorrelation value of signed order flow
    """
    signed_volume = df[tick_col] * df[volume_col]
    return signed_volume.autocorr(lag=lag)


########################### MORE INFORMATION ###################################

"""
================================================================================
Microstructural Features for Financial Data Analysis (Chapter 19)
================================================================================

--------------------------------------------------------------------------------
Overview
--------------------------------------------------------------------------------

Microstructure models help us understand how prices are formed in financial markets,
how liquidity and trading costs arise, and how informed/uninformed traders interact.
These features are crucial for risk management, alpha research, and understanding
market behavior at high frequency.

The code is organized into three generations of microstructure models:

1. **First Generation**:  
   - Focuses on estimating bid-ask spreads and volatility as proxies for illiquidity.
   - Uses only price and volume data, without modeling strategic behavior.
   - Includes: Tick Rule, Roll Model, Parkinson Volatility, Corwin-Schultz Spread Estimator.

2. **Second Generation**:  
   - Models the strategic interaction between informed and uninformed traders.
   - Focuses on measuring illiquidity and price impact using signed volume/order flow.
   - Includes: Kyle's Lambda, Amihud Lambda, Hasbrouck Lambda.

3. **Third Generation**:  
   - Models sequential and strategic trade arrival (e.g., PIN model).
   - Useful for market makers and for understanding the impact of information events.
   - Includes: Probability of Informed Trading (PIN), VPIN, Order Flow Imbalance.

--------------------------------------------------------------------------------
Key Functions and Their Use Cases
--------------------------------------------------------------------------------

- **tick_rule**: Assigns trade direction (+1/-1/0) based on price changes.  
  *Use: Required for all models that need signed volume or order flow.*

- **roll_model**: Estimates bid-ask spread and efficient price variance from price series.  
  *Use: Quick liquidity diagnostics when only prices are available.*

- **parkinson_high_low_volatility_k1/k2**: Volatility estimators using high/low prices.  
  *Use: More efficient volatility estimates than close-to-close returns.*

- **Corwin-Schultz (getBeta, getGamma, getAlpha, corwinSchultz, getSigma)**:  
  *Use: Estimate effective bid-ask spread and volatility from high/low prices.*

- **kyles_lambda / kyles_lambda_no_regression**: Measures price impact per unit of signed volume.  
  *Use: Quantifies how much prices move in response to order flow.*

- **amihud_lambda / rolling_amihud_lambda**: Measures illiquidity as price response to dollar volume.  
  *Use: Standard illiquidity metric in empirical finance.*

- **hasbroucks_lambda**: Measures price impact per unit of signed sqrt dollar volume.  
  *Use: Alternative to Kyle/Amihud, more robust to outliers.*

- **compute_pin**: Estimates the Probability of Informed Trading (PIN) using buy/sell counts.  
  *Use: Quantifies the fraction of trades likely to be informed (information risk).*

- **create_volume_buckets, tick_rule_vpin, estimate_trade_volume, compute_ofi, compute_vpin**:  
  *Use: Implements the VPIN (Volume-Synchronized Probability of Informed Trading) pipeline.*

- **compute_signed_orderflow_autocorr**: Computes autocorrelation of signed order flow.  
  *Use: Detects persistence in order flow, which may indicate informed trading or herding(mimic others).*

--------------------------------------------------------------------------------
How to Use
--------------------------------------------------------------------------------

- For basic liquidity/volatility diagnostics, use the first-generation models.
- For price impact and order flow analysis, use the second-generation models.
- For advanced information risk and sequential trade modeling, use the third-generation models.
- See individual function docstrings for parameter details and expected input formats.
================================================================================
"""