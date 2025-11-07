import pandas as pd

df = pd.read_csv('data/raw/data.csv', nrows=100)
print('Month column sample:')
print(df['Month'].head(20))
print(f'\nMonth dtype: {df["Month"].dtype}')
print(f'\nMonth unique values: {sorted(df["Month"].unique())}')
print(f'\nYear column sample:')
print(df['Year'].head(20))
print(f'\nYear dtype: {df["Year"].dtype}')
