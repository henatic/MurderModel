"""
Comprehensive end-to-end training and evaluation script for the Murder Model project.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

from src.preprocessing.data_processor import DataProcessor
from src.models.base_model import BaseModel
from src.models.logistic_model import LogisticModel
from src.evaluation.evaluator import ModelEvaluator


def load_and_preprocess_data(data_path: str, target_col: str = 'Crime Solved', 
                             nrows: int = None) -> tuple:
    """
    Load and preprocess data from CSV file.
    
    Args:
        data_path: Path to data CSV file
        target_col: Name of target column
        nrows: Number of rows to load (None for all)
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    print(f"\n{'='*80}")
    print("STEP 1: Loading and Preprocessing Data")
    print(f"{'='*80}")
    
    # Load data
    print(f"Loading data from: {data_path}")
    if nrows:
        print(f"Loading first {nrows} rows...")
    df = pd.read_csv(data_path, nrows=nrows)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check for target column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    print(f"\nTarget distribution:")
    print(df[target_col].value_counts())
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Preprocess features
    print(f"\nPreprocessing features...")
    processor = DataProcessor()
    X_processed, messages = processor.process_data(X)
    
    if messages:
        print(f"Preprocessing messages:")
        for msg in messages[:5]:
            print(f"  - {msg}")
    
    # Filter to numeric columns only
    X_numeric = X_processed.select_dtypes(include=[np.number])
    print(f"Features after preprocessing: {X_numeric.shape[1]} numeric columns")
    
    # Remove rows with any NaN
    mask = ~X_numeric.isna().any(axis=1)
    X_clean = X_numeric[mask]
    y_clean = y[mask]
    
    print(f"Clean samples: {len(X_clean)} / {len(X)} ({len(X_clean)/len(X)*100:.1f}%)")
    
    # Encode target if needed
    if y_clean.dtype == 'object':
        le = LabelEncoder()
        y_clean = pd.Series(le.fit_transform(y_clean), index=y_clean.index, name=y_clean.name)
        print(f"Target encoded: {dict(enumerate(le.classes_))}")
    
    return X_clean, y_clean, X_clean.columns.tolist()


def train_model(X, y, model_params: dict = None, random_state: int = 42) -> tuple:
    """
    Train model with train/val/test split.
    
    Args:
        X: Features
        y: Target
        model_params: Model hyperparameters
        random_state: Random seed
        
    Returns:
        Tuple of (model, X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print(f"\n{'='*80}")
    print("STEP 2: Training Model")
    print(f"{'='*80}")
    
    # Split data
    print(f"Splitting data (test=20%, val=10%, train=70%)...")
    X_train, X_val, X_test, y_train, y_val, y_test = BaseModel.split_data(
        X, y, 
        test_size=0.2, 
        val_size=0.1, 
        stratify=True, 
        random_state=random_state
    )
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Check class balance
    print(f"\nClass distribution:")
    for name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        dist = y_split.value_counts(normalize=True)
        print(f"  {name}: {dist.to_dict()}")
    
    # Initialize and train model
    print(f"\nTraining Logistic Regression model...")
    model_params = model_params or {}
    model = LogisticModel(random_state=random_state, **model_params)
    model.fit(X_train, y_train)
    print(f"Model trained successfully!")
    
    return model, X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_and_visualize(model, X_train, X_val, X_test, y_train, y_val, y_test,
                          feature_names, output_dir: str = None) -> dict:
    """
    Comprehensive model evaluation with visualizations.
    
    Args:
        model: Trained model
        X_train, X_val, X_test: Feature datasets
        y_train, y_val, y_test: Target datasets
        feature_names: List of feature names
        output_dir: Directory for outputs
        
    Returns:
        Evaluation results dictionary
    """
    print(f"\n{'='*80}")
    print("STEP 3: Model Evaluation")
    print(f"{'='*80}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir=output_dir)
    
    # Evaluate on all datasets
    results = evaluator.evaluate_model(
        model, X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Print report
    evaluator.print_evaluation_report(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluator.save_evaluation_report(results, 'logistic_model', timestamp)
    
    # Visualizations
    print(f"\n{'='*80}")
    print("STEP 4: Generating Visualizations")
    print(f"{'='*80}")
    
    output_path = Path(output_dir) if output_dir else Path('data/output')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix for test set
    print("\nGenerating confusion matrix...")
    y_test_pred = model.predict(X_test)
    evaluator.plot_confusion_matrix(
        y_test, y_test_pred,
        title='Test Set Confusion Matrix',
        save_path=str(output_path / f'confusion_matrix_{timestamp}.png')
    )
    
    # ROC curve for test set
    try:
        print("Generating ROC curve...")
        y_test_proba = model.predict_proba(X_test)[:, 1]
        evaluator.plot_roc_curve(
            y_test, y_test_proba,
            title='Test Set ROC Curve',
            save_path=str(output_path / f'roc_curve_{timestamp}.png')
        )
    except Exception as e:
        print(f"Could not generate ROC curve: {e}")
    
    # Feature importance
    try:
        print("Generating feature importance plot...")
        importance = model.get_feature_importance()
        evaluator.plot_feature_importance(
            importance, feature_names,
            title='Feature Importance (Logistic Regression Coefficients)',
            save_path=str(output_path / f'feature_importance_{timestamp}.png')
        )
    except Exception as e:
        print(f"Could not generate feature importance: {e}")
    
    # Classification report
    evaluator.print_classification_report(y_test, y_test_pred)
    
    return results


def save_model(model, output_dir: str = None):
    """Save trained model to disk."""
    print(f"\n{'='*80}")
    print("STEP 5: Saving Model")
    print(f"{'='*80}")
    
    output_path = Path(output_dir) if output_dir else Path('data/output')
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_path / f'logistic_model_{timestamp}.pkl'
    
    model.save_model(str(model_path))
    print(f"Model saved to: {model_path}")
    
    return str(model_path)


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train and evaluate murder prediction model')
    parser.add_argument('--data', type=str, default='data/raw/data.csv',
                       help='Path to data CSV file')
    parser.add_argument('--target', type=str, default='Crime Solved',
                       help='Name of target column')
    parser.add_argument('--nrows', type=int, default=None,
                       help='Number of rows to load (None for all)')
    parser.add_argument('--output-dir', type=str, default='data/output',
                       help='Output directory for results')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving model to disk')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*80}")
    print("MURDER MODEL - END-TO-END TRAINING PIPELINE")
    print(f"{'#'*80}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output_dir}")
    print(f"Random State: {args.random_state}")
    
    try:
        # Load and preprocess
        X, y, feature_names = load_and_preprocess_data(
            args.data, args.target, args.nrows
        )
        
        # Train model
        model, X_train, X_val, X_test, y_train, y_val, y_test = train_model(
            X, y, random_state=args.random_state
        )
        
        # Evaluate and visualize
        results = evaluate_and_visualize(
            model, X_train, X_val, X_test, y_train, y_val, y_test,
            feature_names, args.output_dir
        )
        
        # Save model
        if not args.no_save:
            model_path = save_model(model, args.output_dir)
        
        print(f"\n{'#'*80}")
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'#'*80}\n")
        
        return model, results
        
    except Exception as e:
        print(f"\n{'!'*80}")
        print(f"ERROR: {str(e)}")
        print(f"{'!'*80}\n")
        raise


if __name__ == '__main__':
    main()
