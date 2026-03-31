import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from tqdm import tqdm 

def is_stationary(series, significance=0.05):
    result = adfuller(series.dropna())
    return result[1] < significance  # p-value

def getWeights(d, size):
    # thres>0 drops insignificant weights
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w
#———————————————————————————————————————-
def plotWeights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size = size)
        w_ = pd.DataFrame(w_, index = range(w_.shape[0])[ : : -1], columns = [d])
        w = w.join(w_, how = 'outer')
    ax = w.plot()
    ax.legend(loc = 'upper left')
    plt.show()
    return

#-----------------------------------------

def fracDiff(series,d,thres=0.001):
    '''
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w=getWeights(d,series.shape[0])
    print("Weights: ", w)
    #2) Determine initial calcs to be skipped based on weight-loss threshold
    T = len(w)
    current_ratio = 1
    previous_ratio = 1
    skip = 0
    w_sum = np.sum(np.abs(w))
    w_subset_sum = 0
    for l in range(T):
        w_subset_sum += np.abs(w[l]) # Smaller weights are at the start of the series
        previous_ratio = current_ratio
        current_ratio = w_subset_sum/w_sum
        if previous_ratio <= thres and current_ratio > thres:
            skip = l
            break
    print("Number of Initial Calculations to Skip:", skip)
    #3) Apply weights to values
    df={}

    for name in series.columns:
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series()
        for iloc in range(skip,seriesF.shape[0]):
            loc=seriesF.index[iloc]
            if not np.isfinite(series.loc[loc,name]):continue # exclude NAs
            df_[loc]=np.dot(w[skip:iloc+1].T, seriesF.iloc[skip:iloc+1][::-1])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df


def fracDiff_book_impl(series,d,thres=.01): 
    '''
    This function corresponds to the book implementation. We currently do not use it.
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1]. 
    '''
    #1) Compute weights for the longest series 
    w=getWeights(d,series.shape[0])
    #2) Determine initial calcs to be skipped based on weight-loss threshold 
    w_=np.cumsum(abs(w))
    #w_/=w_[-1] # normalization step, not sure why it is applied
    skip=w_[w_>thres].shape[0]
    #3) Apply weights to values
    df={}
    for name in series.columns:
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series() 
        for iloc in range(skip,seriesF.shape[0]):
            loc=seriesF.index[iloc]
            if not np.isfinite(series.loc[loc,name]): continue # exclude NAs
            weights = w[-(iloc+1):,:].T
            values = seriesF.loc[:loc]
            df_[loc]=np.dot(weights, values)[0,0]
        df[name]=df_.copy(deep=True) 
    df=pd.concat(df,axis=1)
    return df

def getWeights_FFD(d,size,thres):
    # thres>0 drops insignificant weights
    w=[1.]
    for k in range(1,size):
        w_=-w[-1]/k*(d-k+1)
        if(abs(w_) <= thres):
            break
        w.append(w_)
    w=np.array(w[::-1]).reshape(-1,1)
    return w

def fracDiff_FFD(series,d,thres=1e-5):
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w=getWeights_FFD(d,len(series),thres)
    width=len(w)-1
    print('size of frac_diff weights', len(w))
    #2) Apply weights to values
    df={}
    for name in series.columns:
        seriesF,df_=series[[name]].ffill().dropna(),pd.Series()
        for iloc1 in tqdm(range(width,seriesF.shape[0])):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue # exclude NAs
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)    
    df=pd.concat(df,axis=1)
    return df

def fracDiff_FFD_initial(series,d,thres=1e-5):
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w=getWeights_FFD(d,len(series),thres)
    width=len(w)-1
    print('size of frac_diff weights', len(w))
    #2) Apply weights to values
    df={}
    for name in series.columns:
        seriesF,df_=series[[name]].ffill().dropna(),pd.Series()
        for iloc1 in tqdm(range(width,seriesF.shape[0])):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue # exclude NAs
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)    
    df=pd.concat(df,axis=1)
    return df, w


def fracDiff_FFD_live(new_series, w, prev_tail):
    width = len(w) - 1
    prev_tail = prev_tail.iloc[-len(w):]  # Clip to what's strictly necessary

    combined = pd.concat([prev_tail, new_series]).ffill().dropna()
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()

    df = {}
    name = 'Adj Close'
    df_ = pd.Series(dtype='float64')
    seriesF = combined[[name]]

    for iloc1 in range(width, seriesF.shape[0]):
        loc0 = seriesF.index[iloc1 - width]
        loc1 = seriesF.index[iloc1]

        df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]

        df[name] = df_

    new_diffed = pd.concat(df, axis=1)
    new_diffed = new_diffed.loc[new_series.index.intersection(new_diffed.index)]
    return new_diffed



def plotMinFFD(series, column_name, tresh=1e-5):
    out=pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])
    for d in np.linspace(0,1,11):
        df2=fracDiff_FFD(series,d,thres=.01)
        corr=np.corrcoef(series.loc[df2.index,column_name],df2[column_name])[0,1]
        df2=adfuller(df2[column_name],maxlag=1,regression='c',autolag=None)
        out.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr] # with critical value
    out.to_csv('data_testMinFFD.csv')
    out[['adfStat','corr']].plot(secondary_y='adfStat')
    print('adfStat 95%confidence: ', out['95% conf'].mean())
    plt.axhline(out['95% conf'].mean(),linewidth=1,color='r',linestyle='dotted')
    plt.savefig('data_testMinFFD.png')


def plot_fracdiff_price(close, fracdiff_series):
    # Create figure and axis objects with a single subplot
    fig, ax1 = plt.subplots(figsize=(12,6))

    # Plot fracDiff series on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('FracDiff Values', color=color)
    ax1.plot(fracdiff_series.index, fracdiff_series.values, color=color, label='FracDiff')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create second y-axis that shares x-axis
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Close Price', color=color)
    ax2.plot(close.index, close.values, color=color, label='Close')
    ax2.tick_params(axis='y', labelcolor=color)

    # Format time axis
    plt.gcf().autofmt_xdate()

    plt.title('Fractional Differentiation vs Close Price')
    plt.show()

#Changed the adfuller arguments to autolag = 'AIC', since this has an adaptive number of lags and is more precise
def optimal_ffd(data, column_name='price', start=0, end=1, interval=10, t=1e-5):
    column_data = pd.DataFrame(data[column_name])
    for d in np.linspace(start, end, interval): #scipy minimize
        dfx = fracDiff_FFD(column_data, d, thres=t)
        if not dfx.empty:
            if sm.tsa.stattools.adfuller(dfx[column_name], autolag='AIC')[1] < 0.05:
                return d
    print('no optimal d')
    return d

def ___optimal_ffd___(data, column_name='price', start=0, end=1, interval=10, t=1e-5):
    column_data = pd.DataFrame(data[column_name])
    for d in np.linspace(start, end, interval): #scipy minimize
        dfx = fracDiff_FFD(column_data, d, thres=t)
        if not dfx.empty:
            if sm.tsa.stattools.adfuller(dfx[column_name], maxlag=1, regression='c', autolag=None)[1] < 0.05:
                return d
    print('no optimal d')
    return d

def get_optimal_ffd_for_all_columns(data, start=0, end=1, interval=10, t=1e-5):
    optimal_d_values = np.zeros(data.shape[1])  # Initialize array to store optimal d values
    
    for i, column_name in enumerate(data.columns):
        print(column_name)
        optimal_d_values[i] = optimal_ffd(data, column_name, start, end, interval, t)
    
    return optimal_d_values