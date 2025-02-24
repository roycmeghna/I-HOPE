# data_loader.py

import pandas as pd
from config import DATA_FILE, EMA_FILE

def load_and_merge_data():
    df = pd.read_csv(DATA_FILE)
    ema = pd.read_csv(EMA_FILE)
    merged = pd.merge(df, ema, on=['uid', 'day'], how='inner')
    merged = merged.dropna(subset=['phq4_score']).copy()
    return merged

def categorize_phq4(score):
    if 0 <= score <= 2:
        return 0  # NORMAL
    elif 3 <= score <= 5:
        return 1  # MILD
    elif 6 <= score <= 8:
        return 2  # MODERATE
    elif 9 <= score <= 12:
        return 3  # SEVERE
    else:
        return 'Unknown'

def add_phq4_category(df):
    df['phq4_category'] = df['phq4_score'].apply(categorize_phq4)
    return df

if __name__ == '__main__':
    data = load_and_merge_data()
    data = add_phq4_category(data)
    print(data.head())
