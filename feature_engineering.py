# feature_engineering.py

import numpy as np
import pandas as pd

def fill_missing_values(df):
    print("Missing values before fill:")
    print(df.isna().sum())
    df = df.fillna(0)
    print("Missing values after fill:")
    print(df.isna().sum())
    return df

def safe_division(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0

def add_calculated_columns(df):
    df['(call_in_num_ep_0+call_out_num_ep_0) /(call_in_duration_ep_0+call_out_duration_ep_0)'] = df.apply(
        lambda row: safe_division(
            row.get('call_in_num_ep_0', 0) + row.get('call_out_num_ep_0', 0),
            row.get('call_in_duration_ep_0', 0) + row.get('call_out_duration_ep_0', 0)
        ), axis=1
    )
    df['loc_home_unlock_num/loc_home_unlock_duration'] = df.apply(
        lambda row: safe_division(row.get('loc_home_unlock_num', 0), row.get('loc_home_unlock_duration', 0)), axis=1
    )
    # (Add additional calculated columns here as in your original code)
    df['sleep_end - sleep_start'] = df.apply(
        lambda row: row.get('sleep_end', np.nan) - row.get('sleep_start', np.nan), axis=1
    )
    df['unlock_num_ep_0/ unlock_duration_ep_0'] = df.apply(
        lambda row: safe_division(row.get('unlock_num_ep_0', 0), row.get('unlock_duration_ep_0', 0)), axis=1
    )
    df = df.fillna(0)
    return df

if __name__ == '__main__':
    # For testing, you could load a sample dataframe and check the outputs.
    df = pd.DataFrame()  # Placeholder
