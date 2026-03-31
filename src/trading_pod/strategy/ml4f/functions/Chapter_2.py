import pandas as pd
import numpy as np

####    CSV PARSING    ####

def process_ticker_csv(file_path):
    """Processes bitcoin tick data csv file into pandas dataframe

    Args:
        file_path (string): path to csv file.

    Returns:
        DataFrame: Dataframe with date, price, trad size btc and trade siz usdt columns
    """
    columns = ['Seq Number', 'Price', 'Volume BTC', 'Volume USDT', 'Date', 'is buyer maker', 'is best match']
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, names=columns)
    
    # Select only the columns we're interested in
    df = df[['Date', 'Price', 'Volume BTC', 'Volume USDT']]

    # Convert the unix timestamp column to utc time
    df["Date"] = pd.to_datetime(df["Date"], unit='ms', utc=True)
    
    return df

####    REGULAR BARS    ####    

def general_bars(df,column,m,tick = False):
    """
    Compute tick bars
    
    # Args:
        df: pd.DataFrame()
        column: name for price data
        m: int(), threshold value for ticks
    # Returns:
        idx: list of indices
    """
    t = df[column]
    ts = 0
    idx = []

    for i,x in enumerate(t):
        ts += 1 if tick else x
        if ts >= m:
            idx.append(i)
            ts = 0
    return idx

def general_bar_df(df,column,m, tick = False):
    idx = general_bars(df, column, m, tick)
    return df.iloc[idx]

def tick_bars(df,price_column,m):
    return general_bars(df,price_column,m, tick = True)

def volume_bars(df,volume_column,m):
    return general_bars(df,volume_column,m)

def dollar_bars(df,dollar_column,m):
    return general_bars(df,dollar_column,m)

def tick_bar_df(df,tick_column,m):
    return general_bar_df(df,tick_column,m,tick = True)

def volume_bar_df(df,volume_column,m):
    return general_bar_df(df,volume_column,m)

def dollar_bar_df(df,dollar_column,m):
    return general_bar_df(df,dollar_column,m)


####    INFORMATION-DRIVEN BARS    ####

def capture_imbalance_bars(df_input, price_header, alpha = 0.2, initial_e_T = 1, is_volume_dollar = False, volume_dollar_header = None):
    df = df_input.copy()
    
    #S Tick Rule
    df["Previous Price"] = df[price_header].shift(1)
    df = df.dropna()
    df["Delta p"] = df[price_header] - df["Previous Price"]

    # Initialize 'bt' with NaN
    df["bt"] = np.nan

    # Set 'bt' based on the tick rule
    df.loc[df['Delta p'] > 0, 'bt'] = 1
    df.loc[df['Delta p'] < 0, 'bt'] = -1

    # Carry forward the last valid value when 'Delta p' is zero
    df['bt'] = df['bt'].ffill()

    if is_volume_dollar:
        df["bt*vt"] = df['bt'] * df[volume_dollar_header]

    df_return = pd.DataFrame(columns = df.columns)
    
    # Calculate the cumulative EWMA using the .ewm() method
    relevant_term = "bt*vt" if is_volume_dollar else "bt"
    ewma_cumulative_term = df[relevant_term].ewm(alpha=alpha).mean().values
    
    # Add the cumulative EWMA values to the DataFrame as a new column
    df[f'EWMA_{relevant_term}'] = np.abs(ewma_cumulative_term)

    # Initialize the expected value of T as 1
    e_T = initial_e_T

    while(True):
        # Define theta
        df["theta"] = np.abs(df[relevant_term].cumsum())
    
        #Expected Value of theta
        df["E_theta"] = e_T * df[f'EWMA_{relevant_term}']
    
        #Condition
        df["Condition"] = np.where(df['theta'] > df['E_theta'], 1, 0)
        #see the first index that satisfies condition
        first_one_index = df["Condition"].idxmax()

        new_bar_size = first_one_index - df.index[0] + 1
        
        # Update e_T using EWMA formula
        e_T = alpha * new_bar_size + (1 - alpha) * e_T

        #eliminate columns to be able to calculate in next cycle
        df = df.drop(['theta', 'E_theta', 'Condition'], axis=1)
        #append first row to my df_return
        df_return = pd.concat([df_return, df.loc[first_one_index].to_frame().T])
        #create df to be used in next cycle
        df_final = df[df.index > first_one_index]
        
        #Break Condition
        if(len(df_final) == 0): break
        
        df = df_final.copy()

    #same structure of original df
    df_return = df_return.drop(['Previous Price', 'Delta p', 'bt'], axis=1)

    return df_return



def capture_run_bars(df_input, price_header, alpha = 0.2, initial_e_T = 1, is_volume_dollar = False, volume_dollar_header = None, initial_buy_portion = 0.5, initial_buy_vol = 1, initial_sell_vol = 1):
    df = df_input.copy()
    
    # Tick Rule
    df["Previous Price"] = df[price_header].shift(1)
    df = df.dropna()
    df["Delta p"] = df[price_header] - df["Previous Price"]

    # Initialize 'bt' with NaN
    df["bt"] = np.nan

    # Set 'bt' based on the tick rule
    df.loc[df['Delta p'] > 0, 'bt'] = 1
    df.loc[df['Delta p'] < 0, 'bt'] = -1

    # Carry forward the last valid value when 'Delta p' is zero
    df['bt'] = df['bt'].ffill()

    df['bt_buy'] = df['bt'].apply(lambda x: 1 if x == 1 else 0)
    df['bt_sell'] = df['bt'].apply(lambda x: -1 if x == -1 else 0)

    if is_volume_dollar:
        df['bt_buy*vt'] = df['bt_buy'] * df[volume_dollar_header]
        df['bt_sell*vt'] = df['bt_sell'] * df[volume_dollar_header]
        df[f"buy_{volume_dollar_header}"] = df[volume_dollar_header].where(df['bt'] == 1, 0) 
        df[f"sell_{volume_dollar_header}"] = df[volume_dollar_header].where(df['bt'] == -1, 0)

    df_return = pd.DataFrame(columns = df.columns)
    
    # Calculate the cumulative EWMA using the .ewm() method
    relevant_term_buy = "bt_buy*vt" if is_volume_dollar else "bt_buy"
    relevant_term_sell = "bt_sell*vt" if is_volume_dollar else "bt_sell"

    # Initialize the expected value of T as 1
    e_T = initial_e_T
    buy_portion = initial_buy_portion
    buy_vol = initial_buy_vol
    sell_vol = initial_sell_vol

    while(True):
        a = False
        # Define theta
        df["bt_buy_sum"] = df['bt_buy'].cumsum()
        df["bt_sell_sum"] = df['bt_sell'].cumsum()

        if is_volume_dollar:
            df["bt_buy*vt_sum"] = df[relevant_term_buy].cumsum()
            df["bt_sell*vt_sum"] = df[relevant_term_sell].cumsum()
            df["theta"] = df[['bt_buy*vt_sum', 'bt_sell*vt_sum']].abs().max(axis=1) # atenção
        else:
            df["theta"] = df[['bt_buy_sum', 'bt_sell_sum']].abs().max(axis=1) # atenção


    
        #Expected Value of theta
        if not is_volume_dollar:
            df["E_theta"] = max(e_T * buy_portion, e_T * (1 - buy_portion))
        else:
            df['E_theta'] = max(e_T * buy_portion * buy_vol, e_T * (1 - buy_portion) * sell_vol)
        
        #Condition
        df["Condition"] = np.where(df['theta'] > df['E_theta'], 1, 0)
    
        if df["Condition"].sum() == 0: # atenção
            a = True
        #see the first index that satisfies condition
        first_one_index = df["Condition"].idxmax() #E quando df condition é 0?

        new_bar_size = first_one_index - df.index[0] + 1
        new_portion = (df['bt_buy_sum'].loc[first_one_index])/new_bar_size
        if is_volume_dollar:
            new_buy_vol = (df['bt_buy*vt_sum'].loc[first_one_index])
            new_sell_vol = abs((df['bt_sell*vt_sum'].loc[first_one_index])) # atenção
        
        # Update e_T using EWMA formula
        e_T = alpha * new_bar_size + (1 - alpha) * e_T
        buy_portion  = alpha * new_portion + (1 - alpha) * buy_portion

        if is_volume_dollar:
            buy_vol = alpha * new_buy_vol + (1 - alpha) * buy_vol
            sell_vol = alpha * new_sell_vol + (1 - alpha) * sell_vol
        #eliminate columns to be able to calculate in next cycle
        df = df.drop(['theta', 'E_theta', 'Condition'], axis=1)
        #append first row to my df_return
        df_return = pd.concat([df_return, df.loc[first_one_index].to_frame().T])
        #create df to be used in next cycle
        df_final = df[df.index > first_one_index]
        
        #Break Condition
        #if a : break # atenção
        if(len(df_final) == 0): break
        
        df = df_final.copy()

    #same structure of original df
    if is_volume_dollar:
        df_return = df_return.drop(['Previous Price', 'Delta p', 'bt','bt_buy','bt_sell', 'bt_buy*vt','bt_sell*vt', f"buy_{volume_dollar_header}", f"sell_{volume_dollar_header}", 'bt_buy_sum', 'bt_sell_sum', 'bt_buy*vt_sum', 'bt_sell*vt_sum'], axis=1)
    else:
        df_return = df_return.drop(['Previous Price', 'Delta p', 'bt','bt_buy','bt_sell','bt_buy_sum','bt_sell_sum'], axis=1)

    return df_return


### CUSUM FILTER ###

def cusum_filter(df_input, price_header, h, molecule = None):
    #multiprocessing
    if molecule is not None:
        df_input = df_input.loc[molecule]   

    tEvents, sPos, sNeg = [], 0, 0
    diff = df_input[price_header].diff()
    
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    
    # Create a new DataFrame with only the rows corresponding to tEvents
    filtered_df = df_input.loc[tEvents]
    
    return filtered_df


def cusum_filter_live(df_input, price_header, h, molecule=None, sPos_prev=0, sNeg_prev=0):
    
    # Ensure 'date' is a datetime type and set as index
    df_input = df_input.copy()
    df_input['date'] = pd.to_datetime(df_input['date'])
    df_input.set_index('date', inplace=True)

    # multiprocessing
    if molecule is not None:
        df_input = df_input.loc[molecule]

    sPos, sNeg = sPos_prev, sNeg_prev
    tEvents = []
    diff = df_input[price_header].diff()
    
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    
    # Create a new DataFrame with only the rows corresponding to tEvents
    filtered_df = df_input.loc[tEvents]
    
    return filtered_df, sPos, sNeg