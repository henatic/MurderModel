"""
Comprehensive end-to-end training and evaluation script for the Murder Model project.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate, StratifiedKFold, GroupShuffleSplit
import warnings
import sys
from typing import Tuple, Optional

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


def train_model(X_raw, y_raw, model_type: str = 'logistic', model_params: dict = None,
                random_state: int = 42,
                split_strategy: str = 'random',
                geo_column: str = 'State',
                class_weight: Optional[str] = None) -> tuple:
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
    print(f"Splitting data (test=20%, val=10%, train=70%)... strategy={split_strategy}")
    def temporal_split():
        if 'Year' not in X_raw.columns:
            return None
        df = X_raw.copy()
        df['__y'] = y_raw.values
        df = df.sort_values('Year')
        n = len(df)
        test_size = int(0.2 * n)
        val_size = int(0.1 * n)
        train = df.iloc[: n - test_size - val_size]
        val = df.iloc[n - test_size - val_size: n - test_size]
        test = df.iloc[n - test_size:]
        return (train.drop(columns='__y'), val.drop(columns='__y'), test.drop(columns='__y'),
                train['__y'], val['__y'], test['__y'])

    def geo_split():
        if geo_column not in X_raw.columns:
            return None
        gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=random_state)
        groups = X_raw[geo_column]
        train_idx, test_idx = next(gss.split(X_raw, y_raw, groups))
        X_temp, X_test_raw = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_temp, y_test = y_raw.iloc[train_idx], y_raw.iloc[test_idx]
        groups_temp = groups.iloc[train_idx]
        gss_val = GroupShuffleSplit(test_size=0.125, n_splits=1, random_state=random_state)
        train_idx, val_idx = next(gss_val.split(X_temp, y_temp, groups_temp))
        X_train_raw, X_val_raw = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
        y_train, y_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]
        return X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test

    split = None
    if split_strategy == 'temporal':
        split = temporal_split()
    elif split_strategy == 'geo':
        split = geo_split()
    if split is None:
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = BaseModel.split_data(
            X_raw, y_raw,
            test_size=0.2,
            val_size=0.1,
            stratify=True,
            random_state=random_state
        )
    else:
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split

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

    target_mapping = None
    # Encode target after split to avoid leakage
    if y_train.dtype == 'object' or y_train.dtype.name == 'category':
        le = LabelEncoder()
        y_train = pd.Series(le.fit_transform(y_train), index=y_train.index, name=y_train.name)
        y_val = pd.Series(le.transform(y_val), index=y_val.index, name=y_val.name)
        y_test = pd.Series(le.transform(y_test), index=y_test.index, name=y_test.name)
        print(f"Target encoded: {dict(enumerate(le.classes_))}")
        target_mapping = {idx: cls for idx, cls in enumerate(le.classes_)}

    # Check class balance
    print(f"\nClass distribution:")
    for name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        dist = y_split.value_counts(normalize=True)
        print(f"  {name}: {dist.to_dict()}")

    # Initialize and train model
    model_params = model_params or {}

    if model_type == 'logistic':
        print(f"\nTraining Logistic Regression model...")
        model = LogisticModel(random_state=random_state, scaler=False, class_weight=class_weight, **model_params)
    elif model_type == 'random_forest':
        print(f"\nTraining Random Forest model...")
        model = RandomForestModel(random_state=random_state, scaler=False, class_weight=class_weight, **model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'logistic' or 'random_forest'")

    model.fit(X_train, y_train)
    print(f"Model trained successfully!")

    return (
        model,
        (X_train, X_val, X_test, y_train, y_val, y_test),
        (X_train_raw, X_val_raw, X_test_raw),
        target_mapping,
    )


def evaluate_and_visualize(model, X_train, X_val, X_test, y_train, y_val, y_test,
                          feature_names, model_type: str = 'logistic', output_dir: str = None,
                          timestamp: str = None) -> Tuple[dict, str]:
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
    if timestamp is None:
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
    
    return results, timestamp


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


def run_cross_validation(model_type: str, X_train, y_train, cv_folds: int,
                         random_state: int, output_dir: Path, timestamp: str,
                         class_weight: Optional[str] = None) -> dict:
    """
    Run cross-validation on training data and save summary metrics.
    """
    if cv_folds is None or cv_folds < 2:
        return {}

    if model_type == 'logistic':
        model = LogisticModel(random_state=random_state, scaler=False, class_weight=class_weight)
    elif model_type == 'random_forest':
        model = RandomForestModel(random_state=random_state, scaler=False, class_weight=class_weight or 'balanced')
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    print(f"\nRunning {cv_folds}-fold cross-validation ({model_type})...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        error_score='raise'
    )

    summary = {metric: {'mean': float(cv_results[f'test_{metric}'].mean()),
                        'std': float(cv_results[f'test_{metric}'].std())}
               for metric in scoring}
    print("CV summary:")
    for metric, stats in summary.items():
        print(f"  {metric:10s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cv_path = output_dir / f"cv_{model_type}_{timestamp}.json"
    import json
    with open(cv_path, 'w') as f:
        json.dump({'folds': cv_folds, 'model': model_type, 'timestamp': timestamp,
                   'summary': summary}, f, indent=2)
    print(f"Cross-validation results saved to: {cv_path}")
    return summary


def fairness_report(X_raw_test: pd.DataFrame, y_test: pd.Series, model_type: str,
                    output_dir: Path, timestamp: str, target_mapping: dict = None) -> dict:
    """
    Compute simple group-wise positive rates on test split for fairness signal.
    """
    if X_raw_test is None or y_test is None:
        return {}

    df = X_raw_test.copy()
    df['target'] = y_test

    positive_label = 1
    groups = {}
    for col in ['Victim Sex', 'Victim Race']:
        if col in df.columns:
            grp = df.groupby(col)['target'].agg(['count', 'mean']).reset_index()
            grp = grp.rename(columns={'mean': 'positive_rate'})
            groups[col] = grp.to_dict(orient='records')

    if not groups:
        return {}

    report = {
        'model': model_type,
        'timestamp': timestamp,
        'positive_label': target_mapping.get(positive_label) if target_mapping else positive_label,
        'groups': groups,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    import json
    fairness_path = output_dir / f"fairness_{model_type}_{timestamp}.json"
    with open(fairness_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Fairness report saved to: {fairness_path}")
    for col, rows in groups.items():
        print(f"\nGroup positive rates by {col}:")
        for row in rows:
            print(f"  {row[col]!s:12s} count={row['count']:5d} positive_rate={row['positive_rate']:.3f}")
    return report


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
    parser.add_argument('--cv-folds', type=int, default=0,
                       help='If >1, run cross-validation with given folds on training set')
    parser.add_argument('--split-strategy', type=str, default='random',
                       choices=['random', 'temporal', 'geo'],
                       help='Data split strategy: random (default), temporal (by Year), geo (by column)')
    parser.add_argument('--geo-column', type=str, default='State',
                       help='Column to use for geographic splits (default: State)')
    parser.add_argument('--class-weight', type=str, default=None,
                       help="Class weight for models (e.g., 'balanced')")
    parser.add_argument('--logreg-C', type=float, default=None,
                       help="Override Logistic Regression C")
    parser.add_argument('--logreg-penalty', type=str, default=None,
                       help="Override Logistic Regression penalty (e.g., l1, l2)")
    parser.add_argument('--logreg-solver', type=str, default=None,
                       help="Override Logistic Regression solver (e.g., liblinear, saga)")
    parser.add_argument('--logreg-max-iter', type=int, default=None,
                       help="Override Logistic Regression max iterations")
    
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

        cw = None if (args.class_weight is None or str(args.class_weight).lower() == 'none') else args.class_weight

        model_params = {}
        if args.model == 'logistic':
            if args.logreg_C is not None:
                model_params['C'] = args.logreg_C
            if args.logreg_penalty is not None:
                model_params['penalty'] = args.logreg_penalty
            if args.logreg_solver is not None:
                model_params['solver'] = args.logreg_solver
            if args.logreg_max_iter is not None:
                model_params['max_iter'] = args.logreg_max_iter

        # Train model
        (model,
         (X_train, X_val, X_test, y_train, y_val, y_test),
         (X_train_raw, X_val_raw, X_test_raw),
         target_mapping) = train_model(
            X_raw, y_raw, model_type=args.model, random_state=args.random_state,
            split_strategy=args.split_strategy, geo_column=args.geo_column,
            class_weight=cw,
            model_params=model_params or None
        )

        feature_names = X_train.columns.tolist()
        
        # Evaluate and visualize
        results, ts = evaluate_and_visualize(
            model, X_train, X_val, X_test, y_train, y_val, y_test,
            feature_names, model_type=args.model, output_dir=args.output_dir
        )

        output_dir_path = Path(args.output_dir)

        # Optional cross-validation
        run_cross_validation(
            args.model, X_train, y_train, args.cv_folds, args.random_state,
            output_dir_path, ts, class_weight=cw
        )

        # Simple fairness check on raw test split
        fairness_report(X_test_raw, y_test, args.model, output_dir_path, ts, target_mapping)
        
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
