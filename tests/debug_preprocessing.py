"""
Diagnostic script to debug DataProcessor and understand why integration tests fail.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.data_processor import DataProcessor

def diagnose_data_processing():
    """Debug the data processing pipeline step by step."""
    
    # Load small sample of raw data
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'data.csv'
    
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        return
    
    print("="*80)
    print("DIAGNOSTIC: DataProcessor Debugging")
    print("="*80)
    
    # Load first 1000 rows
    print("\n1. Loading data...")
    df = pd.read_csv(data_path, nrows=1000)
    print(f"   Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)[:10]}...")  # Show first 10
    
    # Check for target column
    print("\n2. Checking for target column...")
    potential_targets = ['Crime Solved', 'Solved', 'target', 'label']
    target_col = None
    for col in potential_targets:
        if col in df.columns:
            target_col = col
            print(f"   Found target: '{target_col}'")
            print(f"   Target values: {df[target_col].value_counts().to_dict()}")
            break
    
    if target_col is None:
        print(f"   No target column found. Available columns:")
        print(f"   {list(df.columns)}")
        return
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    print(f"\n3. Data before preprocessing:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   X dtypes:\n{X.dtypes.value_counts()}")
    print(f"   y dtype: {y.dtype}")
    
    # Initialize processor
    processor = DataProcessor()
    
    print(f"\n4. Expected categorical columns: {len(processor.categorical_columns)}")
    print(f"   {processor.categorical_columns}")
    print(f"\n5. Expected numeric columns: {len(processor.numeric_columns)}")
    print(f"   {processor.numeric_columns}")
    
    # Check which expected columns exist
    print(f"\n6. Column availability check:")
    missing_cat = [col for col in processor.categorical_columns if col not in X.columns]
    missing_num = [col for col in processor.numeric_columns if col not in X.columns]
    if missing_cat:
        print(f"   Missing categorical: {missing_cat}")
    if missing_num:
        print(f"   Missing numeric: {missing_num}")
    
    # Process data step by step
    print(f"\n7. Running validation...")
    X_validated, messages = processor.validate_data(X.copy())
    print(f"   After validation: {X_validated.shape}")
    print(f"   Validation messages: {messages[:3] if messages else 'None'}...")
    print(f"   NaN count: {X_validated.isna().sum().sum()}")
    
    print(f"\n8. Running handle_missing_values...")
    X_no_missing = processor.handle_missing_values(X_validated.copy())
    print(f"   After handle_missing_values: {X_no_missing.shape}")
    print(f"   NaN count: {X_no_missing.isna().sum().sum()}")
    
    print(f"\n9. Running handle_outliers...")
    X_no_outliers = processor.handle_outliers(X_no_missing.copy())
    print(f"   After handle_outliers: {X_no_outliers.shape}")
    print(f"   NaN count: {X_no_outliers.isna().sum().sum()}")
    
    print(f"\n10. Running engineer_features...")
    X_engineered = processor.engineer_features(X_no_outliers.copy())
    print(f"   After engineer_features: {X_engineered.shape}")
    print(f"   NaN count: {X_engineered.isna().sum().sum()}")
    
    print(f"\n11. Running encode_categorical...")
    X_encoded = processor.encode_categorical(X_engineered.copy())
    print(f"   After encode_categorical: {X_encoded.shape}")
    print(f"   NaN count: {X_encoded.isna().sum().sum()}")
    print(f"   Dtypes: {X_encoded.dtypes.value_counts().to_dict()}")
    
    print(f"\n12. Running scale_numeric...")
    X_scaled = processor.scale_numeric(X_encoded.copy())
    print(f"   After scale_numeric: {X_scaled.shape}")
    print(f"   NaN count: {X_scaled.isna().sum().sum()}")
    
    # Now try the full pipeline
    print(f"\n13. Running full process_data pipeline...")
    X_processed, issues = processor.process_data(X.copy())
    print(f"   Final processed shape: {X_processed.shape}")
    print(f"   Final NaN count: {X_processed.isna().sum().sum()}")
    print(f"   Final dtypes: {X_processed.dtypes.value_counts().to_dict()}")
    
    # Check which columns have all NaN
    all_nan_cols = X_processed.columns[X_processed.isna().all()].tolist()
    if all_nan_cols:
        print(f"\n   WARNING: Columns with all NaN: {all_nan_cols}")
    
    # Check which rows have all NaN
    all_nan_rows = X_processed.isna().all(axis=1).sum()
    print(f"\n   Rows with all NaN: {all_nan_rows} / {len(X_processed)}")
    
    # Check numeric only
    print(f"\n14. Filtering to numeric columns only...")
    X_numeric = X_processed.select_dtypes(include=[np.number])
    print(f"   Numeric-only shape: {X_numeric.shape}")
    print(f"   Numeric-only NaN count: {X_numeric.isna().sum().sum()}")
    
    # Filter rows with any NaN
    print(f"\n15. Dropping rows with any NaN...")
    mask = ~X_numeric.isna().any(axis=1)
    X_clean = X_numeric[mask]
    y_clean = y[mask]
    print(f"   Final clean shape: {X_clean.shape}")
    print(f"   Final clean y shape: {y_clean.shape}")
    
    if len(X_clean) > 0:
        print(f"\n✓ SUCCESS: {len(X_clean)} clean samples available for training")
        print(f"  Features: {X_clean.columns.tolist()}")
    else:
        print(f"\n✗ PROBLEM: No clean samples remaining after preprocessing!")
        print(f"\n   Investigating why all rows were dropped...")
        # Show which columns have the most NaN
        nan_counts = X_numeric.isna().sum().sort_values(ascending=False)
        print(f"   Top 10 columns by NaN count:")
        print(nan_counts.head(10))

if __name__ == '__main__':
    diagnose_data_processing()
