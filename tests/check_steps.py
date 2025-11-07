"""Quick check of what columns exist at each step."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing.data_processor import DataProcessor

data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'data.csv'
df = pd.read_csv(data_path, nrows=1000)

target_col = 'Crime Solved'
y = df[target_col]
X = df.drop(columns=[target_col])

processor = DataProcessor()

# Step by step
print("Original columns:", X.columns.tolist())
print("Original dtypes:", X.dtypes.value_counts().to_dict())

X1, _ = processor.validate_data(X.copy())
print("\nAfter validate:", X1.columns.tolist())
print("Dtypes:", X1.dtypes.value_counts().to_dict())

X2 = processor.handle_missing_values(X1.copy())
X3 = processor.handle_outliers(X2.copy())
X4 = processor.engineer_features(X3.copy())

print("\nAfter engineer_features:", X4.columns.tolist())
print("Dtypes:", X4.dtypes.value_counts().to_dict())

X5 = processor.encode_categorical(X4.copy())
print("\nAfter encode_categorical:", X5.columns.tolist())
print("Dtypes:", X5.dtypes.value_counts().to_dict())
print("Object columns still present:", X5.select_dtypes(include='object').columns.tolist())

print("\nAbout to scale. Numeric columns to scale:", processor.numeric_columns)
print("Do they exist in X5?", [col in X5.columns for col in processor.numeric_columns])
print("Year values sample:", X5['Year'].head(10).tolist() if 'Year' in X5.columns else "N/A")
print("Year mean:", X5['Year'].mean() if 'Year' in X5.columns else "N/A")
print("Year std:", X5['Year'].std() if 'Year' in X5.columns else "N/A")
