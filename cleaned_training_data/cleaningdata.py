import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def clean_df(df):
    df = df.drop_duplicates()

    # Remove extreme target outliers (always safe)
    low = df['target'].quantile(0.01)
    high = df['target'].quantile(0.99)
    df = df[(df['target'] > low) & (df['target'] < high)]

    # Cap feature outliers (always safe)
    for c in df.columns:
        if c != 'target':
            q1 = df[c].quantile(0.01)
            q99 = df[c].quantile(0.99)
            df[c] = np.clip(df[c], q1, q99)

    # Only run Isolation Forest if dataset is large enough
    if len(df) > 500:
        X = df.drop('target', axis=1)
        iso = IsolationForest(contamination=0.02, random_state=42)
        mask = iso.fit_predict(X)
        df = df[mask == 1]

    return df.reset_index(drop=True)


df = pd.read_csv('cleaned_training_data/stock_4_train.csv')
df = clean_df(df)
df.to_csv('cleaned_training_data/stock_4_train_cleaned.csv', index=False)

df = pd.read_csv('cleaned_training_data/stock_7_train.csv')
df = clean_df(df)
df.to_csv('cleaned_training_data/stock_7_train_cleaned.csv', index=False)
