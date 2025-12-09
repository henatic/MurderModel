"""
Reproduce the ROC curve comparison from Phase 4.
This script recreates the roc_comparison plot showing Logistic Regression vs Random Forest.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing.data_processor import DataProcessor


def main():
    print("="*80)
    print("REPRODUCING ROC CURVE COMPARISON")
    print("="*80)
    
    # Load and process data
    print("\n1. Loading and processing data...")
    data_path = Path(__file__).parent / 'data' / 'raw' / 'data.csv'
    
    # Load data
    df = pd.read_csv(data_path, nrows=20000)
    
    # Separate features and target
    if 'Crime Solved' not in df.columns:
        raise ValueError("Target column 'Crime Solved' not found in data")
    
    # Encode target: 'Yes' -> 1, 'No' -> 0
    y = (df['Crime Solved'] == 'Yes').astype(int)
    X_raw = df.drop(columns=['Crime Solved'])
    
    # Process data with leakage mitigation
    processor = DataProcessor(drop_leakage_features=True)
    X, messages = processor.fit_transform(X_raw)
    
    # Split data (70/15/15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Validation set: {len(X_val)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Train Logistic Regression
    print("\n2. Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    lr_model.fit(X_train, y_train)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    
    # Train Random Forest
    print("\n3. Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Generate ROC curves
    print("\n4. Generating ROC curve comparison...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Logistic Regression ROC
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
    lr_auc = auc(lr_fpr, lr_tpr)
    ax.plot(lr_fpr, lr_tpr, color='#3498db', linewidth=2,
            label=f'Logistic Regression (AUC = {lr_auc:.3f})')
    
    # Random Forest ROC
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
    rf_auc = auc(rf_fpr, rf_tpr)
    ax.plot(rf_fpr, rf_tpr, color='#e74c3c', linewidth=2,
            label=f'Random Forest (AUC = {rf_auc:.3f})')
    
    # Random classifier baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.500)')
    
    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Save
    output_dir = Path(__file__).parent / 'data' / 'output'
    output_path = output_dir / 'roc_comparison_reproduced.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ ROC curve saved to: {output_path}")
    print(f"\n5. Results Summary:")
    print(f"   Logistic Regression AUC: {lr_auc:.4f}")
    print(f"   Random Forest AUC: {rf_auc:.4f}")
    print("\n" + "="*80)
    print("REPRODUCTION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
