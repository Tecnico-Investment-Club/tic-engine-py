import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


def Tick_Imbalance_Bars(df_input, T_start):
    df = df_input.copy()
    # Tick Rule
    df["Adj Close Past"] = df["Adj Close"].shift(1)
    df = df.dropna() # here we lose a tick when calculating bt to solve this we can artificially give the first bt as parameter
    df["Delta p"] = df["Adj Close"] - df["Adj Close Past"]

    # Initialize 'bt' with NaN
    df["bt"] = np.nan

    # Set 'bt' based on the tick rule
    df.loc[df['Delta p'] > 0, 'bt'] = 1
    df.loc[df['Delta p'] < 0, 'bt'] = -1

    # Carry forward the last valid value when 'Delta p' is zero
    df['bt'] = df['bt'].ffill()

    # Defining the expected values of the imbalance 
    alpha = 0.5
    # Calculate the cumulative EWMA using the .ewm() method
    ewma_cumulative_bt = df['bt'].ewm(alpha=alpha).mean().values
    # Add the cumulative EWMA values to the DataFrame as a new column
    df['EWMA_bt'] = np.abs(ewma_cumulative_bt)

    # Starting the variables
    theta = 0 
    t_length = 0
    t_lengths = [T_start]
    e_theta = 0
    e_T = t_lengths[0]

    for x in t_lengths[1:]:
        e_T = alpha * x + (1 - alpha) * e_T

    final_index = []

    # Iterate trough the rows
    for index, row in df.iterrows():
        t_length += 1
        theta += row['bt']
        e_theta = row['EWMA_bt'] * e_T

        # Check for condition
        if np.abs(theta) > e_theta:

            # Save the index to make a bar
            final_index += [index]
            
            # Reset and save length of bar
            theta = 0
            e_theta = 0
            t_lengths += [t_length]
            t_length = 0

            # Calculate new E_T 
            e_T = t_lengths[0]
            for x in t_lengths[1:]:
                e_T = alpha * x + (1 - alpha) * e_T

    # After going trough the df get the output df

    df_return = df.loc[final_index]
    df_return = df_return.drop(["Delta p","EWMA_bt","bt","Adj Close Past"], axis = 1)
    df_return = df_return.sort_index()  
    return df_return

