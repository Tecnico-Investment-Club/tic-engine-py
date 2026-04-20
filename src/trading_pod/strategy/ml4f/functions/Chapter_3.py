import pandas as pd
import numpy as np
import multiprocessing as mp
from Parallel_Processing_utils import mpPandasObj


#get daily volatility
def getDailyVol(close,span0=100):
    # daily vol, reindexed to close
    df0=close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0=df0[df0>0]
    df0 = df0 - 1
    df0=pd.Series(close.index[df0], index=close.index[close.shape[0]-df0.shape[0]:])
    time = pd.to_datetime(df0.values)
    df0.index = pd.to_datetime(df0.index).tz_localize(None)
    time = pd.to_datetime(time).tz_localize(None)
    df0=close.loc[df0.index]/close.loc[time].values -1 # daily returns
    df0=df0.ewm(span=span0).std()
    return df0

####    TRIPLE BARRIER METHOD     ####

#Triple-barrier Labeling Method
def applyPtSlOnT1(close,events,ptSl,molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_=events.loc[molecule]
    out=events_[['t1']].copy(deep=True)
    if ptSl[0]>0:pt=ptSl[0]*events_['trgt']
    else:pt=pd.Series(index=events.index) # NaNs
    if ptSl[1]>0:sl=-ptSl[1]*events_['trgt']
    else:sl=pd.Series(index=events.index) # NaNs
    events_['t1'] = events_['t1'].fillna(close.index[-1]).infer_objects(copy=False)
    for loc, t1 in events_['t1'].items():
        df0=close[loc:t1] # path prices
        df0=(df0/close[loc]-1)*events_.at[loc,'side'] # path returns
        out.loc[loc, 'sl'] = (df0[df0 < sl[loc]].index.min() 
                      if (df0 < sl[loc]).any() else np.nan) 
        out.loc[loc, 'pt'] = (df0[df0 < sl[loc]].index.min() 
                      if (df0 < sl[loc]).any() else np.nan)
    return out

def getEvents(close,tEvents,ptSl,trgt,minRet,numThreads,t1=False):
    print("started getEvents")
    #1) get target
    trgt=trgt.loc[tEvents]
    trgt=trgt[trgt>minRet] # minRet
    #2) get t1 (max holding period)
    if t1 is False:t1=pd.Series(pd.NaT,index=tEvents).dt.tz_localize('UTC')
    # Ensure timezone
    t1 = t1.dt.tz_localize('UTC') if t1.dt.tz is None else t1   
    #3) form events object, apply stop loss on t1
    side_=pd.Series(1.,index=trgt.index)
    events=pd.concat({'t1':t1,'trgt':trgt,'side':side_}, \
    axis=1).dropna(subset=['trgt'])
    df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index), \
    numThreads=numThreads,close=close,events=events,ptSl=[ptSl,ptSl])
    for col in ['pt', 'sl']:
        df0[col] = pd.to_datetime(df0[col], utc=True)
    events['t1']=df0.dropna(how='all').min(axis=1, skipna=True) # pd.min ignores nan
    events=events.drop('side',axis=1)
    return events



def getBins(events,close):
#1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    events_['t1']=pd.to_datetime(events_['t1'], utc=True)
    events_['t1'] = events_['t1'].tz_convert(close.index.tz)
    px=events_.index.union(events_['t1']).drop_duplicates()
    px = close.reindex(px,method='bfill')
    px = px.ffill()
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1']].values/px.loc[events_.index]-1
    out['bin']=np.sign(out['ret'])
    return out


def getEvents_meta(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    #1) get target
    trgt=trgt.loc[tEvents] 
    trgt=trgt[trgt>minRet] 
    #2) get t1 (max holding period)
    if t1 is False: 
        t1 = pd.Series(pd.NaT, index=tEvents) 
    #3) form events object, apply stop loss on t1
    if side is None: 
        side_ = pd.Series(1.,index=trgt.index)
        ptSl_ = [ptSl[0],ptSl[0]] 
    else:
        side_ = side.loc[trgt.index] 
        ptSl_ = ptSl[:2]
    events=(pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1).dropna(subset=['trgt']))
    df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index),
        numThreads=numThreads,close=close,events=events,ptSl=ptSl_)
    
    def min_with_timestamps_and_column(series):
        min_timestamp = pd.NaT
        min_column = None
        for column, value in series.items():
            if isinstance(value, pd.Timestamp):
                if min_timestamp is pd.NaT or value < min_timestamp:
                    min_timestamp = value
                    min_column = column                  
        return pd.Series({'t1': min_timestamp, 'barrier': min_column})

    events[['t1', 'barrier']] = df0.apply(min_with_timestamps_and_column, axis=1)
    if side is None:
        events=events.drop('side',axis=1)
    return events



def getBins_meta(events, close):
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1']).drop_duplicates()
    px=close.reindex(px,method='bfill')

    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1']].values/px.loc[events_.index]-1

    if 'side' in events_:
        out['ret']*=events_['side'] # meta-labelling

    out['bin']=np.sign(out['ret'])
    out['bin'] = out['bin'].replace(0, pd.NA).ffill() 

    if 'side' in events_:
        out.loc[out['ret']<=0,'bin']=0.0 # meta-labelling
    return out


#DROPPING INSIGNIFICANT LABELS

def dropLabels(events,minPtc=.05):
# apply weights, drop labels with insufficient examples
    while True:
        df0=events['bin'].value_counts(normalize=True)
        if (df0.min()>minPtc or df0.shape[0]<3):break
        print('dropped label',df0.argmin(),df0.min())
        events=events[events['bin']!=df0.argmin()]
    return events


#These functions are adaptions of the anterior ones that label data with zeros if the vertical barrier is the first touch

def getEvents_zero(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    """
    Find the time of the first barrier touch
    :param close: A pd series of prices
    :param tEvents: The pd timeindex containing the timestamps that produce triple barriers
    :param ptSl: A non-neg float that sets the width of the two barriers. 0 means the respective horizontal barrier will be disabled
    :param trgt: A pd series of targets, expressed in terms of absolute returns
    :param minRet: constant, The min target return required for the running a triple barrier search
    :param numThreads: constant, The no. of threads concurrently used by the function
    :param t1: A pd series with the timestamps of the vertical barriers. (index: eventStart, value: eventEnd).
        -   If trgt = False, vBarrier is disabled
    :param side: A pd series with the timestamps of the vertical barriers. (eventStart, eventEnd). 
        -   If trgt = False, vBarrier is disabled

    :return: A pd dataframe with columns:
        -   Index: time when 1) trgt > threshold, 2) triple barriers are triggered
        -   t1: the timestamps at which the first barrier is touched
        -   trgt: the target that was used to generate the horizontal barriers
        -   side (optional): the side of the trade
    """
    #1) get target
    trgt=trgt.loc[tEvents] # get a list of targets when triple barriers are triggered
    trgt=trgt[trgt>minRet] # only select those triple barriers events when the targets are above a certain threshold
    
    #2) get t1 (max holding period)
    if t1 is False: # if no limit on holding period (no vBarriers)
        t1 = pd.Series(pd.NaT, index=tEvents) # create an NaT pd.series that havs the same index as the pd.series of tEvents

    #3) form events object, apply stop loss on t1
    if side is None: # if side is not fed into the function
        side_ = pd.Series(1.,index=trgt.index) # create a pd.series of '1' that havs the same index as the pd.series of trgt
        ptSl_ = [ptSl,ptSl] # assume symmetric barriers, uBarrier and lBarrier have the same width
    else:
        side_ =side.loc[trgt.index] # only select those sides when 1) trgt > threshold, 2) triple barriers are triggered
        ptSl_ = ptSl[:2] # barriers are the same as the input

    # I think the index of `events` is the same as the trgt
    # events: 'index',' t1', 'trgt', 'side'
    events=(pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1).dropna(subset=['trgt']))
    
    # Multiprocessing
    # Input: 1) close prices, 2) events' timestamp, 3) the width of the barriers
    # Output: df0, A pandas dataframe containing the timestamps at which each barrier was touched
    # df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index),
    #                 numThreads=numThreads, close=close, events=events,
    #                 ptSl=ptSl_)

    df0 = mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index),
        numThreads=numThreads,close=close,events=events,ptSl=ptSl_)
    # Drop the data in df0 if no limit is touched
    # Find the earliest of the three dates (NaN is ignored)
    def min_with_timestamps_and_column(series):
        # Inicializa a variável que vai guardar o menor timestamp encontrado e sua coluna
        min_timestamp = pd.NaT
        min_column = None
        
        # Itera sobre cada coluna na série (linha do DataFrame)
        for column, value in series.items():
            if isinstance(value, pd.Timestamp):
                # Atualiza o menor timestamp e a coluna se o timestamp atual é menor ou se é o primeiro encontrado
                if min_timestamp is pd.NaT or value < min_timestamp:
                    min_timestamp = value
                    min_column = column
                    
        return pd.Series({'t1': min_timestamp, 'barrier': min_column})

    # Apply the custom function to find the earliest timestamp along the axis
    events[['t1', 'barrier']] = df0.apply(min_with_timestamps_and_column, axis=1)
    
    # If `side` is not fed, drop the column
    if side is None:
        events=events.drop('side',axis=1)
    
    return events

def getBins_zero(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    #1) prices aligned with events
    # drop the events with no t1 (should have cleaned before enter)
    events_=events.dropna(subset=['t1'])
    # px is the list of timestamps that have either 'start' or 'end' events
    px=events_.index.union(events_['t1']).drop_duplicates()
    # find all the prices when there is a either 'start' or 'end' event
    px=close.reindex(px,method='bfill')

    #2) create out object
    # create a pd.df with the indexes same as events_.index
    out=pd.DataFrame(index=events_.index)
    # create a `ret` column that calculate the returns from the 'end' event to the 'start' events
    # it calculates the returns of signals
    out['ret']=px.loc[events_['t1']].values/px.loc[events_.index]-1

    # Define conditions and choices
    conditions = [
        events['barrier'] == 'pt',  # Condition for 'pt'
        events['barrier'] == 't1',  # Condition for 't1'
        events['barrier'] == 'sl'   # Condition for 'sl'
    ]
    choices = [
        1,   # Choice for 'pt'
        0,   # Choice for 't1'
        -1   # Choice for 'sl'
    ]

    # Apply conditions and choices using np.select
    out['bin'] = np.select(conditions, choices, default=np.nan)

    return out
