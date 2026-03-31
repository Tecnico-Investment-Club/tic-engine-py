import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm


def plugIn(msg,w):
    # Compute plug-in (ML) entropy rate
    pmf=pmf1(msg,w)
    out=-sum([pmf[i]*np.log2(pmf[i]) for i in pmf])/w 
    return out,pmf
#———————————————————————————————————————
def pmf1(msg,w):
    # Compute the prob mass function for a one-dim discrete rv 
    # len(msg)-w occurrences
    lib={}
    if not isinstance(msg,str):msg=''.join(map(str,msg))
    for i in range(w,len(msg)): 
        msg_=msg[i-w:i]
        if msg_ not in lib:lib[msg_]=[i-w]
        else:lib[msg_]=lib[msg_]+[i-w]
    pmf=float(len(msg)-w) 
    pmf={i:len(lib[i])/pmf for i in lib} 
    return pmf

#———————————————————————————————————————
def lempelZiv_lib(msg):
    i,lib=1,[msg[0]]
    while i<len(msg):
        for j in range(i,len(msg)):
            msg_=msg[i:j+1]
            if msg_ not in lib:
                lib.append(msg_)
                break
        i=j+1
    return lib

#———————————————————————————————————————
def matchLength(msg,i,n):
    # Maximum matched length+1, with overlap.
    # i>=n & len(msg)>=i+n
    subS=''
    for l in range(n):
        msg1=msg[i:i+l+1]
        for j in range(i-n,i):
            msg0=msg[j:j+l+1] 
            if msg1==msg0:
                subS=msg1
                break # search for higher l. 
    return len(subS)+1,subS # matched length + 1
#———————————————————————————————————————
def konto(msg,window=None): 
    '''
    * Kontoyiannis LZ entropy estimate, 2013 version (centered window). 
    * Inverse of the avg length of the shortest non-redundant substring. 
    * If non-redundant substrings are short, the text is highly entropic. 
    * window==None for expanding window, in which case len(msg)%2==0
    * If the end of msg is more relevant, try konto(msg[::-1])
    '''
    out={'num':0,'sum':0,'subS':[]}
    if not isinstance(msg,str):msg=''.join(map(str,msg))
    if window is None: 
        points=range(1,int(len(msg)/2)+1)
    else:
        window=min(window,int(len(msg)/2)) 
        points=range(window,len(msg)-window+1)
    for i in points:
        if window is None:
            l,msg_=matchLength(msg,i,i)
            out['sum']+=np.log2(i+1)/l # to avoid Doeblin condition
        else:
            l,msg_=matchLength(msg,i,window)
            out['sum']+=np.log2(window+1)/l # to avoid Doeblin condition 
        out['subS'].append(msg_)
        out['num']+=1
    out['h']=out['sum']/out['num'] 
    out['r']=1-out['h']/np.log2(len(msg)) # redundancy, 0<=r<=1 
    return out
#———————————————————————————————————————
def Entropy_Volume_Bars(df_input, v_threshold, molecule=None):
    # Multiprocessing
    if molecule is not None:
        df_input = df_input.loc[molecule]

    df = df_input.copy()

    df_return = pd.DataFrame(columns = df.columns)
    
    # Set volume clock
    V = v_threshold
    
    # Calculate price change
    df['price_change'] = df['Adj Close'].diff()
    df = df.dropna()
    
    # Initialize variables
    current_volume = 0
    current_buy_volume = 0
    sigma_deltaP = df['price_change'].std()  # Estimate of the standard deviation of price changes
    
    #bars = []
    buy_proportions = []
    aux_idx = 0
    
    # Iterate over rows
    #for index, row in df.iterrows():
    for index, row in tqdm(df.iterrows(), total=len(df)):
    # your loop code
        # Add volume and buy volume to current bar
        current_volume += row['Volume BTC']
        
        # Calculate buy volume contribution
        price_change = row['price_change']
        current_buy_volume += row['Volume BTC'] * norm.cdf(price_change / sigma_deltaP)

        # Add one to the auxiliary index
        aux_idx += 1
        
        # Check if current volume exceeds threshold
        if current_volume >= V:
            # Compute the buy proportion
            buy_proportion= current_buy_volume/current_volume
            # Append the buy_proportion aux_idx times to buy_proportions list
            buy_proportions.extend([buy_proportion] * aux_idx)
            
            # Reset current volume, buy volume and auxiliary index to zero
            current_volume = 0
            current_buy_volume = 0
            aux_idx = 0
        
    # If there are leftover rows that didn't complete a full bar
    if aux_idx > 0:
        buy_proportion = current_buy_volume / current_volume if current_volume > 0 else 0
        buy_proportions.extend([buy_proportion] * aux_idx)
    
    # Add the buy_proportion to the original DataFrame
    df['buy_proportion'] = buy_proportions
    
    return df
#——————————————————————————————————————— 
def compute_quantiles(buy_proportions, q):
    # Compute quantiles
    quantiles = np.quantile(buy_proportions, np.linspace(0, 1, q + 1))
    
    return quantiles
#——————————————————————————————————————— 
def produce_mapping(buy_proportions, quantiles):
    # Initialize mapping array
    mapping = np.zeros(len(buy_proportions), dtype=int)
    
    # Assign index to each buy proportion based on quantiles
    for i, vB_tau in enumerate(buy_proportions):
        for j, quantile in enumerate(quantiles):
            if vB_tau <= quantile:
                mapping[i] = j + 1  # Indexing starts from 1
                break
    
    return mapping
#——————————————————————————————————————— 
def rolling_window_entropy(quantized_message, window_size, step_size):
    # Initialize Entropy list
    entropy_values = []
    for i in tqdm(range(0, len(quantized_message) - window_size + 1, step_size), desc="Rolling Entropy"):
        j = len(quantized_message) - i
        window = quantized_message[j-window_size:j]
        entropy = konto(window)['h']
        entropy_values.append(entropy)

    entropy_values = entropy_values[::-1]

    return entropy_values
#——————————————————————————————————————— 
def eprob(sample: np.array, x:float)->float:
    return np.sum(sample<=x)/len(sample)
#——————————————————————————————————————— 
def create_entropy_feature(data, v_threshold, q, window_size, step_size):
    # Create Entropy Volume Bars
    entropy_volume_bars = Entropy_Volume_Bars(data, v_threshold)
    # Determine the portion of volume classified as buy 
    buy_proportions = entropy_volume_bars['buy_proportion']
    # Compute the q-quantiles
    quantiles = compute_quantiles(buy_proportions, q)
    # Produce a mapping to one of the disjoint subsets
    mapping = produce_mapping(buy_proportions, quantiles)
    # Estimate entropy using Kontoyiannis' LZ algorithm
    rolling_window_entropy_values = rolling_window_entropy(mapping, window_size, step_size)
    # Derive the CDF
    F = [eprob(rolling_window_entropy_values, i) for i in tqdm(rolling_window_entropy_values)]
    none_vector = (window_size-step_size+1) * [np.nan]
    F = np.concatenate((none_vector, F))
    F = pd.Series(F, index=data.index)
    return F







