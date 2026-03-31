import pandas as pd
import numpy as np


def csw_cusum_windowed(log_price, window=50):
    T = len(log_price)
    S_t = np.full(T, np.nan)

    for t in range(2, T):
        S_n_t_values = []
        n_min = max(1, t - window)
        for n in range(n_min, t):
            delta_y = np.diff(log_price[n:t+1])
            sigma_hat_sq = np.sum(delta_y[1:] ** 2) / (t - n) if (t - n) > 1 else np.nan
            if sigma_hat_sq > 0:
                sigma_hat = np.sqrt(sigma_hat_sq)
                S_n_t = (log_price.iloc[t] - log_price.iloc[n]) / (sigma_hat * np.sqrt(t - n))
                S_n_t_values.append(S_n_t)
        S_t[t] = np.nanmax(S_n_t_values) if S_n_t_values else np.nan

    return S_t

def lagDF(df0, lags):
    """
    Applies lags to a Pandas DataFrame.

    Args:
        df0 (pd.DataFrame): The input DataFrame.
        lags (int or list): The number of lags or a list of lags to apply.

    Returns:
        pd.DataFrame: A new DataFrame with the specified lags applied to the original data.
    """
    df1 = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(lags + 1)  # if lags is an integer, create a range from 0 to lags
    else:
        lags = [int(lag) for lag in lags] # if lags is a list/series, convert to integer list

    for lag in lags:
        df_lagged = df0.shift(lag).copy(deep=True) # Shift the DataFrame by the lag
        df_lagged.columns = [str(col) + '_' + str(lag) for col in df0.columns] # Rename columns
        df1 = df1.join(df_lagged, how='outer') # Join the lagged DataFrame
    return df1

def getBetas(y, x):
    """
    Fits the regression and returns the estimated coefficients and their variances.

    Args:
        y (np.array): The dependent variable in the regression.
        x (np.array): The independent variables in the regression.

    Returns:
        tuple: A tuple containing:
            - bMean (np.array): The estimated regression coefficients.
            - bVar (np.array): The variance-covariance matrix of the coefficients.
    """
    xy = np.dot(x.T, y)          # x transpose times y
    xx = np.dot(x.T, x)          # x transpose times x
    xxinv = np.linalg.inv(xx)    # Inverse of xx
    bMean = np.dot(xxinv, xy)    # Calculate regression coefficients
    err = y - np.dot(x, bMean)    # Calculate the residuals
    bVar = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xxinv # Calculate coefficient variance-covariance matrix
    return bMean, bVar

def getYX_ADF(series, constant, lags):
    """
    Prepares the numpy arrays needed for the recursive ADF tests.

    Args:
        series (pd.Series): The input time series.
        constant (str):  Indicates the time trend component:
            - 'nc': no time trend, only a constant
            - 'ct': a constant plus a linear time trend
            - 'ctt': a constant plus a second-degree polynomial time trend
        lags (int): The number of lags used in the ADF specification.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - y (np.array): The dependent variable for the ADF regression.
            - x (np.array): The independent variables for the ADF regression.
    """
    series_diff = series.diff().dropna()
    x = lagDF(series, lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0] - 1:-1]
    y = series_diff.iloc[-x.shape[0]:].values

    if constant != 'nc':
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
    if constant[:2] == 'ct':
        trend = np.arange(x.shape[0]).reshape(-1, 1)
        x = np.append(x, trend, axis=1)
    if constant == "ctt":
        x = np.append(x, trend**2, axis=1)
    return y, x

def getYX_SMT(series, trend_type):
    """
    Prepares the numpy arrays needed for the Sub-Martingale Tests.

    Args:
        series (pd.Series): The input time series.
        trend_type (str):  Indicates the trend component:
            - 'poly1': Linear and quadratic trend for y.
            - 'poly2': Linear and quadratic trend for log(y).
            - 'exp': Exponential trend.
            - 'power': Power trend.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - y (np.array): The dependent variable for the regression.
            - x (np.array): The independent variables for the regression.
    """
    t = np.arange(1, len(series) + 1).reshape(-1, 1)  # Time variable
    if trend_type == 'poly1':
        y = series.values
        x = np.concatenate((np.ones_like(t), t, t**2), axis=1)
    elif trend_type == 'poly2':
        y = np.log(series.values)
        x = np.concatenate((np.ones_like(t), t, t**2), axis=1)
    elif trend_type == 'exp':
        y = np.log(series.values)
        x = t
    elif trend_type == 'power':
        y = np.log(series.values)
        x = np.log(t)
    return y, x



def getBSADF(logP, minSL, constant, lags):
    """
    Calculates the BSADF statistic for a given time series.

    Args:
        logP (pd.Series): A pandas Series containing log-prices.
        minSL (int): The minimum sample length (r) used for the final regression.
        constant (str): The time trend component in the ADF regression
        lags (int): The number of lags in the ADF regression.

    Returns:
        dict: A dictionary containing:
            - 'Time' (pd.Index): The time index of the input series.
            - 'bsadf' (float): The BSADF statistic.
    """
    y, x = getYX_ADF(logP, constant, lags)
    startPoints = range(0, y.shape[0] - minSL + 1)
    allADF = []
    bsadf = -np.inf

    for start in startPoints:
        y_sub = y[start:]
        x_sub = x[start:]
        bMean, bStd = getBetas(y_sub, x_sub)
        bStd = bStd[0, 0]**0.5
        allADF.append(bMean[0] / bStd)
        if allADF[-1] > bsadf:
            bsadf = allADF[-1]

    out = {'Time': logP.index[lags:], 'bsadf': bsadf}
    return out

def getSMT(logP, minSL, trend_type, phi=0):
    """
    Calculates the Sub-Martingale Test (SMT) statistic.

    Args:
        logP (pd.Series): A pandas Series containing log-prices.
        minSL (int): The minimum sample length.
        trend_type (str): The type of trend to include in the regression
                        ('poly1', 'poly2', 'exp', 'power').
        phi (float, optional): Correction factor for bias. Defaults to 0.

    Returns:
        dict: A dictionary containing:
            - 'Time' (pd.Index): The time index of the input series.
            - 'smt' (float): The SMT statistic.
    """
    T = len(logP)
    startPoints = range(0, T - minSL + 1)
    allSMT = []
    smt = -np.inf

    for start in startPoints:
        y, x = getYX_SMT(logP.iloc[start:], trend_type)  # Use logP.iloc[start:]
        bMean, bVar = getBetas(y, x)
        bStd = bVar[0, 0]**0.5
        smt_value = abs(bMean[1] / bStd)  # Use bMean[1] for the trend coefficient
        if phi > 0:
            t0 = start + 1  # Start point of the subsample
            t = T         # End point of the subsample
            smt_value = smt_value / ((t - t0)**phi)
        allSMT.append(smt_value)
        if allSMT[-1] > smt:
            smt = allSMT[-1]
    out = {'Time': logP.index, 'smt': smt}
    return out

def calculate_sadf_sequence(logP, minSL, constant, lags):
    """
    Calculates the sequence of SADF statistics for an advancing window.

    Args:
        logP (pd.Series): A pandas Series containing log-prices.
        minSL (int): The minimum sample length.
        constant (str): The time trend component ('nc', 'ct', or 'ctt').
        lags (int): The number of lags in the ADF regression.

    Returns:
        pd.DataFrame: A DataFrame containing the SADF sequence.
    """
    T = len(logP)
    sadf_values = []
    time_values = []

    for t in range(minSL, T + 1):
        # Use only the data up to time t
        logP_window = logP.iloc[:t].copy()
        sadf_result = getBSADF(logP_window, minSL, constant, lags)
        sadf_values.append(sadf_result['bsadf'])
        time_values.append(logP_window.index[-1])

    return pd.DataFrame({'Time': time_values, 'SADF': sadf_values}).set_index('Time')

def calculate_smt_sequence(logP, minSL, trend_type, phi=0):
    """
    Calculates the sequence of SMT statistics for an advancing window.

    Args:
        logP (pd.Series): A pandas Series containing log-prices.
        minSL (int): The minimum sample length.
        trend_type (str): The trend type for the SMT test ('poly1', 'poly2', 'exp', 'power').
        phi (float, optional): The correction factor for the SMT test. Defaults to 0.

    Returns:
        pd.DataFrame: A DataFrame containing the SMT sequence.
    """
    T = len(logP)
    smt_values = []
    time_values = []

    for t in range(minSL, T + 1):
        logP_window = logP.iloc[:t].copy()
        smt_result = getSMT(logP_window, minSL, trend_type, phi)
        smt_values.append(smt_result['smt'])
        time_values.append(logP_window.index[-1])

    return pd.DataFrame({'Time': time_values, 'SMT': smt_values}).set_index('Time')
