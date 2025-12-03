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
from src.models.random_forest_model import RandomForestModel
from src.evaluation.evaluator import ModelEvaluator


def load_data(data_path: str, target_col: str = 'Crime Solved',
              nrows: int = None) -> tuple:
    """
    Load raw data from CSV and return features/target.
    """
    print(f"\n{'='*80}")
    print("STEP 1: Loading Data")
    print(f"{'='*80}")

    print(f"Loading data from: {data_path}")
    if nrows:
        print(f"Loading first {nrows} rows...")
    df = pd.read_csv(data_path, nrows=nrows)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    df = df.dropna(subset=[target_col])
    df = df.drop_duplicates()

    print(f"\nTarget distribution (after dropping NaN target rows):")
    print(df[target_col].value_counts())

    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def train_model(X, y, model_type: str = 'logistic', model_params: dict = None,
                random_state: int = 42) -> tuple:
    """
    Train model with train/val/test split.
    
    Args:
        X: Features
        y: Target
        model_type: Type of model ('logistic' or 'random_forest')
        model_params: Model hyperparameters
        random_state: Random seed
        
    Returns:
        Tuple of (model, X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print(f"\n{'='*80}")
    print("STEP 2: Training Model")
    print(f"{'='*80}")
    
    # Split data (on raw features)
    print(f"Splitting data (test=20%, val=10%, train=70%)...")
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = BaseModel.split_data(
        X, y,
        test_size=0.2,
        val_size=0.1,
        stratify=True,
        random_state=random_state
    )

    print(f"  Train: {len(X_train_raw)} samples")
    print(f"  Val:   {len(X_val_raw)} samples")
    print(f"  Test:  {len(X_test_raw)} samples")
    
    # Preprocess using train-only fit to avoid leakage
    processor = DataProcessor()
    print("\nFitting preprocessing on training split only...")
    X_train, messages = processor.fit_transform(X_train_raw)
    X_val = processor.transform(X_val_raw)
    X_test = processor.transform(X_test_raw)

    if messages:
        print("Preprocessing messages:")
        for msg in messages[:5]:
            print(f"  - {msg}")

    # Drop rows with NaN if any survived preprocessing
    def _drop_na(X_split, y_split):
        mask = ~X_split.isna().any(axis=1)
        return X_split[mask], y_split[mask]

    X_train, y_train = _drop_na(X_train, y_train)
    X_val, y_val = _drop_na(X_val, y_val)
    X_test, y_test = _drop_na(X_test, y_test)

    # Encode target after split to avoid leakage
    if y_train.dtype == 'object' or y_train.dtype.name == 'category':
        le = LabelEncoder()
        y_train = pd.Series(le.fit_transform(y_train), index=y_train.index, name=y_train.name)
        y_val = pd.Series(le.transform(y_val), index=y_val.index, name=y_val.name)
        y_test = pd.Series(le.transform(y_test), index=y_test.index, name=y_test.name)
        print(f"Target encoded: {dict(enumerate(le.classes_))}")

    # Check class balance
    print(f"\nClass distribution:")
    for name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        dist = y_split.value_counts(normalize=True)
        print(f"  {name}: {dist.to_dict()}")

    # Initialize and train model
    model_params = model_params or {}

    if model_type == 'logistic':
        print(f"\nTraining Logistic Regression model...")
        model = LogisticModel(random_state=random_state, scaler=False, **model_params)
    elif model_type == 'random_forest':
        print(f"\nTraining Random Forest model...")
        model = RandomForestModel(random_state=random_state, scaler=False, **model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'logistic' or 'random_forest'")

    model.fit(X_train, y_train)
    print(f"Model trained successfully!")

    return model, X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_and_visualize(model, X_train, X_val, X_test, y_train, y_val, y_test,
                          feature_names, model_type: str = 'logistic', output_dir: str = None) -> dict:
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
    model_name = model_type.replace('_', '')
    evaluator.save_evaluation_report(results, f'{model_name}_model', timestamp)
    
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
        title=f'Test Set Confusion Matrix ({model_type.replace("_", " ").title()})',
        save_path=str(output_path / f'confusion_matrix_{model_type}_{timestamp}.png')
    )
    
    # ROC curve for test set
    try:
        print("Generating ROC curve...")
        y_test_proba = model.predict_proba(X_test)[:, 1]
        evaluator.plot_roc_curve(
            y_test, y_test_proba,
            title=f'Test Set ROC Curve ({model_type.replace("_", " ").title()})',
            save_path=str(output_path / f'roc_curve_{model_type}_{timestamp}.png')
        )
    except Exception as e:
        print(f"Could not generate ROC curve: {e}")
    
    # Feature importance
    try:
        print("Generating feature importance plot...")
        importance = model.get_feature_importance()
        title_suffix = 'Coefficients' if model_type == 'logistic' else 'Importance Scores'
        evaluator.plot_feature_importance(
            importance, feature_names,
            title=f'Feature Importance ({model_type.replace("_", " ").title()} {title_suffix})',
            save_path=str(output_path / f'feature_importance_{model_type}_{timestamp}.png')
        )
    except Exception as e:
        print(f"Could not generate feature importance: {e}")
    
    # Classification report
    evaluator.print_classification_report(y_test, y_test_pred)
    
    return results


def save_model(model, model_type: str = 'logistic', output_dir: str = None):
    """Save trained model to disk."""
    print(f"\n{'='*80}")
    print("STEP 5: Saving Model")
    print(f"{'='*80}")
    
    output_path = Path(output_dir) if output_dir else Path('data/output')
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_type.replace('_', '')
    model_path = output_path / f'{model_name}_model_{timestamp}.pkl'
    
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
    parser.add_argument('--model', type=str, default='logistic',
                       choices=['logistic', 'random_forest'],
                       help='Type of model to train (logistic or random_forest)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving model to disk')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*80}")
    print("MURDER MODEL - END-TO-END TRAINING PIPELINE")
    print(f"{'#'*80}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Type: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output_dir}")
    print(f"Random State: {args.random_state}")
    
    try:
        # Load raw data
        X_raw, y_raw = load_data(args.data, args.target, args.nrows)

        # Train model
        model, X_train, X_val, X_test, y_train, y_val, y_test = train_model(
            X_raw, y_raw, model_type=args.model, random_state=args.random_state
        )

        feature_names = X_train.columns.tolist()
        
        # Evaluate and visualize
        results = evaluate_and_visualize(
            model, X_train, X_val, X_test, y_train, y_val, y_test,
            feature_names, model_type=args.model, output_dir=args.output_dir
        )
        
        # Save model
        if not args.no_save:
            model_path = save_model(model, model_type=args.model, output_dir=args.output_dir)
        
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
