"""
Model Comparison Script - Compare multiple models side-by-side.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings
import sys
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

from src.preprocessing.data_processor import DataProcessor
from src.models.base_model import BaseModel
from src.models.logistic_model import LogisticModel
from src.models.random_forest_model import RandomForestModel
from src.evaluation.evaluator import ModelEvaluator


def load_and_preprocess_data(data_path: str, target_col: str = 'Crime Solved', 
                             nrows: int = None) -> tuple:
    """Load and preprocess data from CSV file."""
    print(f"\n{'='*80}")
    print("Loading and Preprocessing Data")
    print(f"{'='*80}")
    
    df = pd.read_csv(data_path, nrows=nrows)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    processor = DataProcessor()
    X_processed, messages = processor.process_data(X)
    
    X_numeric = X_processed.select_dtypes(include=[np.number])
    mask = ~X_numeric.isna().any(axis=1)
    X_clean = X_numeric[mask]
    y_clean = y[mask]
    
    print(f"Clean samples: {len(X_clean)} / {len(X)} ({len(X_clean)/len(X)*100:.1f}%)")
    
    if y_clean.dtype == 'object':
        le = LabelEncoder()
        y_clean = pd.Series(le.fit_transform(y_clean), index=y_clean.index, name=y_clean.name)
    
    return X_clean, y_clean, X_clean.columns.tolist()


def train_and_evaluate_model(model_name: str, model, X_train, X_val, X_test, 
                             y_train, y_val, y_test) -> dict:
    """Train and evaluate a single model."""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(
        model, X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    return results


def compare_models(results_dict: dict, output_dir: str = None):
    """
    Compare multiple models and generate comparison visualizations.
    
    Args:
        results_dict: Dictionary mapping model names to evaluation results
        output_dir: Directory for output files
    """
    print(f"\n{'='*80}")
    print("MODEL COMPARISON REPORT")
    print(f"{'='*80}\n")
    
    output_path = Path(output_dir) if output_dir else Path('data/output')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create comparison table
    comparison_data = []
    for model_name, results in results_dict.items():
        row = {'Model': model_name}
        for dataset in ['train', 'validation', 'test']:
            if dataset in results:
                for metric, value in results[dataset].items():
                    key = f"{dataset}_{metric}"
                    row[key] = value
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print comparison table (test set only for readability)
    print("TEST SET PERFORMANCE:")
    print("-" * 80)
    test_cols = ['Model', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']
    print(comparison_df[test_cols].to_string(index=False))
    print()
    
    # Print validation set performance
    print("VALIDATION SET PERFORMANCE:")
    print("-" * 80)
    val_cols = ['Model', 'validation_accuracy', 'validation_precision', 'validation_recall', 
                'validation_f1', 'validation_roc_auc']
    print(comparison_df[val_cols].to_string(index=False))
    print()
    
    # Print train set performance
    print("TRAIN SET PERFORMANCE:")
    print("-" * 80)
    train_cols = ['Model', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1', 'train_roc_auc']
    print(comparison_df[train_cols].to_string(index=False))
    print()
    
    # Calculate differences
    if len(results_dict) == 2:
        models = list(results_dict.keys())
        print("PERFORMANCE DIFFERENCES (Model 2 - Model 1):")
        print("-" * 80)
        for dataset in ['test', 'validation', 'train']:
            print(f"\n{dataset.upper()} SET:")
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                key = f"{dataset}_{metric}"
                diff = comparison_df.iloc[1][key] - comparison_df.iloc[0][key]
                print(f"  {metric:12s}: {diff:+.6f}")
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = output_path / f'model_comparison_{timestamp}.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nComparison saved to: {comparison_file}")
    
    # Save as JSON too
    json_file = output_path / f'model_comparison_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'models': list(results_dict.keys()),
            'results': results_dict,
            'comparison_table': comparison_df.to_dict('records')
        }, f, indent=2)
    print(f"Comparison JSON saved to: {json_file}")
    
    return comparison_df


def main():
    """Main comparison pipeline."""
    parser = argparse.ArgumentParser(description='Compare multiple models')
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
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['logistic', 'random_forest'],
                       choices=['logistic', 'random_forest'],
                       help='Models to compare')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*80}")
    print("MURDER MODEL - MODEL COMPARISON PIPELINE")
    print(f"{'#'*80}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output_dir}")
    print(f"Random State: {args.random_state}")
    
    try:
        # Load and preprocess
        X, y, feature_names = load_and_preprocess_data(
            args.data, args.target, args.nrows
        )
        
        # Split data once for fair comparison
        print(f"\n{'='*80}")
        print("Splitting Data")
        print(f"{'='*80}")
        X_train, X_val, X_test, y_train, y_val, y_test = BaseModel.split_data(
            X, y, 
            test_size=0.2, 
            val_size=0.1, 
            stratify=True, 
            random_state=args.random_state
        )
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train and evaluate each model
        results_dict = {}
        
        for model_type in args.models:
            if model_type == 'logistic':
                model = LogisticModel(random_state=args.random_state)
                model_name = "Logistic Regression"
            elif model_type == 'random_forest':
                model = RandomForestModel(random_state=args.random_state, n_estimators=100)
                model_name = "Random Forest"
            else:
                continue
            
            results = train_and_evaluate_model(
                model_name, model, 
                X_train, X_val, X_test, 
                y_train, y_val, y_test
            )
            results_dict[model_name] = results
        
        # Compare models
        comparison_df = compare_models(results_dict, args.output_dir)
        
        print(f"\n{'#'*80}")
        print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
        print(f"{'#'*80}\n")
        
        return comparison_df
        
    except Exception as e:
        print(f"\n{'!'*80}")
        print(f"ERROR: {str(e)}")
        print(f"{'!'*80}\n")
        raise


if __name__ == '__main__':
    main()

