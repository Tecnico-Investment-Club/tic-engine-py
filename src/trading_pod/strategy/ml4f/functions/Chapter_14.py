import pandas as pd
import numpy as np
from functions.Chapter_10 import getSignal
from scipy.stats import norm
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score   


def calculate_strategy_metrics(bet_sizes, closes, starting_money):
    """
    Simulate a strategy where bet_sizes indicate the *fraction* of capital invested in the asset.

    Parameters:
        bet_sizes (pd.Series): Fraction of capital to be invested at each timestamp.
        closes (pd.Series): Asset close prices indexed by timestamp.
        starting_money (float): Initial capital.

    Returns:
        dict: Contains capital over time, total PnL, and game_over flag.
    """
    # Ensure matching index (forward fill signals)
    bet_sizes = bet_sizes.reindex(closes.index, method='ffill').fillna(0)

    # Initialize state
    capital = starting_money
    cash = capital  # Initially, all is cash
    position = 0.0  # No asset held
    capital_over_time = []

    for i in range(1, len(closes)):
        prev_price = closes.iloc[i - 1]
        curr_price = closes.iloc[i]
        t = closes.index[i]

        # Target investment fraction
        target_frac = bet_sizes.iloc[i]
        total_value = cash + position * curr_price

        # Rebalance position (buy/sell asset to reach target_frac of total_value)
        target_position_value = target_frac * total_value
        current_position_value = position * curr_price
        delta_position_value = target_position_value - current_position_value

        # Update cash and position based on trade
        if curr_price != 0:
            position += delta_position_value / curr_price
        cash -= delta_position_value

        # Update total capital and track
        capital = cash + position * curr_price
        capital_over_time.append(capital)

        if capital <= 0:
            return {
                'PnL': capital - starting_money,
                'capital_over_time': pd.Series(capital_over_time, index=closes.index[1:]),
                'game_over': True
            }

    return {
        'PnL': capital - starting_money,
        'capital_over_time': pd.Series(capital_over_time, index=closes.index[1:]),
        'game_over': False
    }


def simple_sharpe_ratio(capital_over_time, risk_free_rate=0.01, trading_days=252):
    '''
    pegar nos retornos, calcular a media e o desvio padrao, e depois calcular o sharpe ratio

    Notem que a media e o desvio padrao sao desconhecidos. O que fazemos é usar a media e o desvio padrao da nossa amostra, 
    para estimar a media e o desvio padrao da populacao. 
    Problemas: 
        -Isto assume que os retornos sao IID;
        -Overfitting;
    '''
    
    returns = capital_over_time.pct_change().dropna() # percentagem de variação do capital
    excess_returns = returns - (risk_free_rate / trading_days)  # assumindo 252 trading days/year
    sharpe = excess_returns.mean() / excess_returns.std() # media e desvio padrao 
    sharpe_annualized = sharpe * (trading_days ** 0.5)
    return sharpe_annualized



def probabilistic_sharpe_ratio(returns, benchmark_sr=0, risk_free_rate=0.0, periods_per_year=252):
    """
    Computes the Probabilistic Sharpe Ratio (PSR) as per López de Prado (2018).
    
    Args:
        returns (pd.Series or np.ndarray): Series of returns.
        benchmark_sr (float): Benchmark Sharpe ratio to test against (e.g., 0).
        risk_free_rate (float): Annual risk-free rate.
        periods_per_year (int): Number of periods per year (e.g., 252 for daily).
        
    Returns:
        float: Probability that the true Sharpe ratio exceeds the benchmark.
    """
    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / periods_per_year)
    n = len(excess_returns)
    mean = np.mean(excess_returns)
    std = np.std(excess_returns, ddof=1)
    sr_hat = mean / std

    skew = ((excess_returns - mean)**3).mean() / (std**3)
    kurt = ((excess_returns - mean)**4).mean() / (std**4)
    
    
    # PSR formula
    numerator = (sr_hat - benchmark_sr) * np.sqrt(n - 1)
    denominator = np.sqrt(1 - skew * sr_hat +  ( ((kurt - 1) * sr_hat**2) / 4 ) )
    psr = norm.cdf(numerator / denominator)
    return psr



def metrics_all_paths(all_paths, pd_closes_total, starting_money, events_meta, bins, bin_meta, long_only=True):

    metrics_dict = {
    'pnl': [],
    'sharpe_ratio': [],
    'prob_sharpe_ratio': [],
    'f1_first_model': [],
    'f1_second_model': [],
    'accuracy_first_model': [],
    'accuracy_second_model': []
    }
   
    for path in all_paths:

        events_bet = events_meta.drop(columns=['barrier'])
        events_bet['side'] = path[0]
        bet_size = getSignal(events_bet, stepSize=0.1, prob=path[2], pred=path[1], numClasses=2, numThreads=1)
        bet_size_zeroed = bet_size.copy()

        if long_only:
            bet_size_zeroed[bet_size_zeroed < 0] = 0 # a tal estratégia de não fazermos short
        
     
        closes = pd_closes_total[bet_size.index]
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        bet_size.index = pd.to_datetime(bet_size.index).tz_localize(None)

        pnL_metrics = calculate_strategy_metrics(bet_size, pd_closes_total[bet_size.index], starting_money)
        sharpe_ratio = simple_sharpe_ratio(pnL_metrics['capital_over_time'])
        psr = probabilistic_sharpe_ratio(pnL_metrics['capital_over_time'].pct_change().dropna(), benchmark_sr=0)
        f1_second = f1_score(bin_meta['bin'], path[1])
        f1_first = f1_score(bins['bin'], path[0], average='macro')
        acc_first = accuracy_score(bins['bin'], path[0])
        acc_second = accuracy_score(bin_meta['bin'], path[1])

        metrics_dict['pnl'].append(pnL_metrics['PnL'])
        metrics_dict['sharpe_ratio'].append(sharpe_ratio)
        metrics_dict['prob_sharpe_ratio'].append(psr)
        metrics_dict['f1_first_model'].append(f1_first)
        metrics_dict['f1_second_model'].append(f1_second)
        metrics_dict['accuracy_first_model'].append(acc_first)
        metrics_dict['accuracy_second_model'].append(acc_second)

    metrics_avg = {key: sum(values) / len(values) for key, values in metrics_dict.items()}
    
    return metrics_dict, metrics_avg
        