"""Hyperparameter optimization for Murder Model classifiers.

This script performs hyperparameter tuning using:
1. Grid Search CV
2. Random Search CV
3. Stratified K-Fold cross-validation
4. ROC-AUC scoring metric
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any
import sys

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.data_processor import DataProcessor


class HyperparameterOptimizer:
    """Optimize model hyperparameters using grid/random search."""

    def __init__(self, random_state: int = 42):
        """Initialize optimizer."""
        self.random_state = random_state
        self.results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'random_state': random_state,
            'models': {}
        }

    def load_and_prepare_data(self, data_path: str, nrows: int = None) -> tuple:
        """Load, clean, and preprocess data for training."""
        print(f"\nLoading data from {data_path}...")
        
        df = pd.read_csv(data_path, nrows=nrows)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        if 'Crime Solved' not in df.columns:
            raise ValueError("Target column 'Crime Solved' not found")
        
        df = df.dropna(subset=['Crime Solved']).drop_duplicates()
        y = df['Crime Solved'].map({'Yes': 1, 'No': 0})
        X_raw = df.drop(columns=['Crime Solved'])
        
        processor = DataProcessor(drop_leakage_features=True)
        X, messages = processor.fit_transform(X_raw)
        
        print(f"Preprocessed to {X.shape[1]} features")
        for msg in messages:
            print(f"  - {msg}")
        
        return X, y, processor

    def optimize_logistic_regression(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        search_type: str = 'grid',
        cv_folds: int = 5,
        n_iter: int = 50
    ) -> Dict[str, Any]:
        """Optimize Logistic Regression hyperparameters."""
        print("\n" + "="*80)
        print("OPTIMIZING LOGISTIC REGRESSION")
        print("="*80)
        
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 200, 500, 1000],
            'class_weight': [None, 'balanced']
        }
        
        # For random search, define distributions
        param_distributions = {
            'C': np.logspace(-3, 3, 100),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 200, 500, 1000, 2000],
            'class_weight': [None, 'balanced']
        }
        
        # Create base model (sklearn's LogisticRegression)
        from sklearn.linear_model import LogisticRegression
        base_model = LogisticRegression(random_state=self.random_state)
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scorer = 'roc_auc'  # Use string instead of make_scorer
        
        # Perform search
        if search_type == 'grid':
            print(f"Performing Grid Search with {cv_folds}-fold CV...")
            search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=cv, 
                scoring=scorer,
                n_jobs=-1,
                verbose=2
            )
        else:
            print(f"Performing Random Search with {n_iter} iterations and {cv_folds}-fold CV...")
            search = RandomizedSearchCV(
                base_model,
                param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring=scorer,
                n_jobs=-1,
                verbose=2,
                random_state=self.random_state
            )
        
        # Fit search
        search.fit(X, y)
        
        # Extract results
        results = {
            'search_type': search_type,
            'cv_folds': cv_folds,
            'best_params': search.best_params_,
            'best_score': float(search.best_score_),
            'cv_results': {
                'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': search.cv_results_['std_test_score'].tolist(),
                'params': [str(p) for p in search.cv_results_['params']]
            }
        }
        
        print(f"\nBest parameters: {search.best_params_}")
        print(f"Best ROC-AUC score: {search.best_score_:.4f}")
        
        self.results['models']['logistic_regression'] = results
        
        return results

    def optimize_random_forest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        search_type: str = 'random',
        cv_folds: int = 5,
        n_iter: int = 50
    ) -> Dict[str, Any]:
        """Optimize Random Forest hyperparameters."""
        print("\n" + "="*80)
        print("OPTIMIZING RANDOM FOREST")
        print("="*80)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }
        
        # For random search, define distributions
        param_distributions = {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [None, 5, 10, 20, 30, 50, 100],
            'min_samples_split': [2, 5, 10, 20, 50],
            'min_samples_leaf': [1, 2, 4, 8, 16],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'bootstrap': [True, False]
        }
        
        # Create base model
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(random_state=self.random_state)
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scorer = 'roc_auc'  # Use string instead of make_scorer
        
        # Perform search
        if search_type == 'grid':
            print(f"Performing Grid Search with {cv_folds}-fold CV...")
            # For grid search on RF, use smaller grid due to computational cost
            param_grid_small = {
                'n_estimators': [100, 200],
                'max_depth': [None, 20, 50],
                'min_samples_split': [2, 10],
                'min_samples_leaf': [1, 4],
                'max_features': ['sqrt', 'log2'],
                'class_weight': [None, 'balanced']
            }
            search = GridSearchCV(
                base_model,
                param_grid_small,
                cv=cv,
                scoring=scorer,
                n_jobs=-1,
                verbose=2
            )
        else:
            print(f"Performing Random Search with {n_iter} iterations and {cv_folds}-fold CV...")
            search = RandomizedSearchCV(
                base_model,
                param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring=scorer,
                n_jobs=-1,
                verbose=2,
                random_state=self.random_state
            )
        
        # Fit search
        search.fit(X, y)
        
        # Extract results
        results = {
            'search_type': search_type,
            'cv_folds': cv_folds,
            'best_params': search.best_params_,
            'best_score': float(search.best_score_),
            'cv_results': {
                'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': search.cv_results_['std_test_score'].tolist(),
                'params': [str(p) for p in search.cv_results_['params']]
            }
        }
        
        print(f"\nBest parameters: {search.best_params_}")
        print(f"Best ROC-AUC score: {search.best_score_:.4f}")
        
        self.results['models']['random_forest'] = results
        
        return results

    def save_results(self, output_dir: str = None) -> str:
        """Save optimization results to JSON."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / 'data' / 'output'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self.results['timestamp']
        output_file = output_dir / f'hyperparameter_optimization_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nOptimization results saved to: {output_file}")
        return str(output_file)

    def print_summary(self) -> None:
        """Print optimization summary."""
        print("\n" + "="*80)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("="*80)
        
        for model_name, results in self.results['models'].items():
            print(f"\n{model_name.upper()}:")
            print(f"  Search type: {results['search_type']}")
            print(f"  CV folds: {results['cv_folds']}")
            print(f"  Best ROC-AUC: {results['best_score']:.4f}")
            print(f"  Best parameters:")
            for param, value in results['best_params'].items():
                print(f"    • {param}: {value}")


def main():
    """Run hyperparameter optimization from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize model hyperparameters')
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/data.csv',
        help='Path to raw data CSV file'
    )
    parser.add_argument(
        '--nrows',
        type=int,
        default=50000,
        help='Number of rows to use for optimization'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['logistic', 'random_forest', 'both'],
        default=['both'],
        help='Which models to optimize'
    )
    parser.add_argument(
        '--search-type',
        choices=['grid', 'random'],
        default='random',
        help='Search strategy (grid or random)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=30,
        help='Number of iterations for random search'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Convert relative path to absolute
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent.parent / data_path
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(random_state=args.random_state)
    
    # Load data
    X, y, processor = optimizer.load_and_prepare_data(str(data_path), args.nrows)
    
    # Optimize models
    models_to_optimize = args.models
    if 'both' in models_to_optimize:
        models_to_optimize = ['logistic', 'random_forest']
    
    if 'logistic' in models_to_optimize:
        optimizer.optimize_logistic_regression(
            X, y, 
            search_type=args.search_type,
            cv_folds=args.cv_folds,
            n_iter=args.n_iter
        )
    
    if 'random_forest' in models_to_optimize:
        optimizer.optimize_random_forest(
            X, y,
            search_type=args.search_type,
            cv_folds=args.cv_folds,
            n_iter=args.n_iter
        )
    
    # Save and display results
    optimizer.print_summary()
    optimizer.save_results()


if __name__ == '__main__':
    main()
