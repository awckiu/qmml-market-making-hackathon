import pandas as pd


df = pd.read_csv('cleaned_training_data/stock_4_train.csv')

print(df.shape)        # ✅ no ()
print(df.isnull().sum())
print(df.describe())




