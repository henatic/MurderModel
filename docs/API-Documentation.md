# Murder Model API Documentation

**Version:** 1.0  
**Last Updated:** December 3, 2025  
**Project:** Homicide Case Classification (CSS 581)

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Modules](#core-modules)
5. [API Reference](#api-reference)
6. [Usage Examples](#usage-examples)
7. [Troubleshooting](#troubleshooting)

## Overview

The Murder Model project provides machine learning tools for classifying homicide cases as solved or unsolved based on historical data (1980-2014). The system includes:

- **Data preprocessing** with leakage mitigation
- **Two classification models**: Logistic Regression and Random Forest
- **Comprehensive evaluation** with fairness metrics
- **Hyperparameter optimization** tools
- **Visualization suite** for analysis

### Key Features

✅ Leakage-safe feature engineering  
✅ Multiple validation strategies (random, temporal, geographic)  
✅ Class imbalance handling (SMOTE, ADASYN, class weights)  
✅ Fairness monitoring across demographic groups  
✅ Hyperparameter tuning with cross-validation  
✅ Publication-quality visualizations

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

```powershell
# Clone repository
git clone <repository-url>
cd archive

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```powershell
# Run tests
python -m pytest tests/ -v

# Should see: 21 tests passing
```

## Quick Start

### Train a Model

```powershell
# Train Logistic Regression
python src/models/logistic_model.py --nrows 10000

# Train Random Forest
python src/models/random_forest_model.py --nrows 10000

# Train with class weights
python src/models/logistic_model.py --nrows 10000 --class-weight balanced

# Train with SMOTE
python src/models/random_forest_model.py --nrows 10000 --resample smote
```

### Compare Models

```powershell
# Compare both models on same data split
python src/models/compare.py --nrows 20000
```

### Generate Visualizations

```powershell
# Phase 5 comprehensive visualizations
python src/evaluation/phase5_visualizations.py
```

## Core Modules

### 1. Data Processing (`src/preprocessing/`)

**`data_processor.py`** - Main data processing pipeline

```python
from preprocessing.data_processor import DataProcessor

# Initialize processor
processor = DataProcessor()

# Load and prepare data
df = processor.load_data()
X, y = processor.prepare_features(df)

# Get train/val/test splits
splits = processor.split_data(X, y, random_state=42)
X_train, X_val, X_test, y_train, y_val, y_test = splits
```

**Key Methods:**

- `load_data(nrows=None)`: Load raw CSV data
- `prepare_features(df)`: Engineer features, encode, scale
- `split_data(X, y, **kwargs)`: Create stratified train/val/test splits
- `temporal_split(X, y, split_column='Year')`: Split by time
- `geographic_split(X, y, split_column='State')`: Split by location

### 2. Models (`src/models/`)

#### Base Model (`base_model.py`)

All models inherit from `BaseModel`:

```python
from models.base_model import BaseModel

# Common interface for all models
class MyModel(BaseModel):
    def train(self, X_train, y_train, **kwargs):
        # Training logic
        pass

    def predict(self, X):
        # Prediction logic
        pass
```

**Key Methods:**

- `train(X_train, y_train, **kwargs)`: Train model
- `predict(X)`: Get class predictions
- `predict_proba(X)`: Get probability estimates
- `save_model(filepath)`: Save trained model
- `load_model(filepath)`: Load saved model

#### Logistic Regression (`logistic_model.py`)

```python
from models.logistic_model import LogisticModel

# Initialize
model = LogisticModel(random_state=42)

# Train
model.train(X_train, y_train, class_weight='balanced')

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate
from evaluation.model_evaluator import ModelEvaluator
evaluator = ModelEvaluator(model, model_name='logistic')
results = evaluator.evaluate_model(
    X_train, y_train, X_val, y_val, X_test, y_test
)
```

**Parameters:**

- `random_state`: Random seed for reproducibility
- `class_weight`: None, 'balanced', or dict
- `max_iter`: Maximum iterations (default: 1000)
- `C`: Regularization strength (default: 1.0)

#### Random Forest (`random_forest_model.py`)

```python
from models.random_forest_model import RandomForestModel

# Initialize
model = RandomForestModel(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Train
model.train(X_train, y_train, class_weight='balanced_subsample')

# Feature importance
importance = model.get_feature_importance(feature_names)
```

**Parameters:**

- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: None)
- `min_samples_split`: Minimum samples to split (default: 2)
- `min_samples_leaf`: Minimum samples in leaf (default: 1)
- `class_weight`: None, 'balanced', 'balanced_subsample'
- `random_state`: Random seed

### 3. Evaluation (`src/evaluation/`)

#### Model Evaluator (`model_evaluator.py`)

```python
from evaluation.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(model, model_name='my_model')

# Full evaluation
results = evaluator.evaluate_model(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
)

# Access metrics
print(f"Test ROC-AUC: {results['test']['roc_auc']:.4f}")
print(f"Test Accuracy: {results['test']['accuracy']:.4f}")

# Generate plots
evaluator.plot_roc_curve(y_test, y_pred_proba)
evaluator.plot_confusion_matrix(y_test, y_pred)
evaluator.plot_feature_importance(feature_names, importance_values)
```

**Key Methods:**

- `evaluate_model()`: Comprehensive evaluation on all splits
- `cross_validate()`: K-fold cross-validation
- `evaluate_fairness()`: Fairness metrics by demographic groups
- `plot_roc_curve()`: ROC curve visualization
- `plot_confusion_matrix()`: Confusion matrix heatmap
- `plot_feature_importance()`: Feature importance bar chart

#### Visualizations (`visualizations.py`)

```python
from evaluation.visualizations import ModelVisualizer

visualizer = ModelVisualizer()

# Learning curves
visualizer.plot_learning_curves(
    model=model.model,
    X=X_train,
    y=y_train,
    model_name='Random Forest'
)

# Precision-Recall curve
visualizer.plot_precision_recall_curve(
    y_true=y_test,
    y_proba=y_pred_proba,
    model_name='Logistic Regression'
)

# Threshold analysis
visualizer.plot_threshold_analysis(
    y_true=y_test,
    y_proba=y_pred_proba,
    model_name='Random Forest'
)
```

### 4. Utilities (`src/utils/`)

#### Feature Audit (`feature_audit.py`)

```python
from utils.feature_audit import FeatureAuditor

auditor = FeatureAuditor()
recommendations = auditor.identify_leakage_features(nrows=100000)

# Get leakage features
leakage_features = recommendations['leakage_features']
print(f"Found {len(leakage_features)} potential leakage features")
```

#### Hyperparameter Tuning (`hyperparameter_tuning.py`)

```python
from models.hyperparameter_tuning import HyperparameterOptimizer

optimizer = HyperparameterOptimizer()

# Optimize Logistic Regression
lr_results = optimizer.optimize_logistic_regression(
    X_train, y_train,
    search_type='random',  # or 'grid'
    n_iter=20,
    cv_folds=5
)

# Optimize Random Forest
rf_results = optimizer.optimize_random_forest(
    X_train, y_train,
    search_type='random',
    n_iter=50,
    cv_folds=3
)

# Get best parameters
print(f"Best LR params: {lr_results['best_params']}")
print(f"Best RF params: {rf_results['best_params']}")
```

## API Reference

### DataProcessor

```python
class DataProcessor:
    """Handle data loading, preprocessing, and feature engineering."""

    def __init__(self, data_path: str = None)

    def load_data(self, nrows: int = None) -> pd.DataFrame
        """Load raw data from CSV."""

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]
        """Prepare features (X) and target (y) with encoding and scaling."""

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, ...]
        """Create train/validation/test splits."""

    def temporal_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split_column: str = 'Year',
        train_ratio: float = 0.7,
        val_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, ...]
        """Split data by temporal column."""

    def geographic_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split_column: str = 'State',
        train_ratio: float = 0.7,
        val_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, ...]
        """Split data by geographic column."""
```

### BaseModel

```python
class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, random_state: int = 42)

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None
        """Train the model."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray
        """Predict class labels."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray
        """Predict class probabilities."""

    def save_model(self, filepath: str) -> None
        """Save model to disk."""

    @staticmethod
    def load_model(filepath: str) -> 'BaseModel'
        """Load model from disk."""
```

### ModelEvaluator

```python
class ModelEvaluator:
    """Evaluate model performance with metrics and visualizations."""

    def __init__(
        self,
        model: BaseModel,
        model_name: str = 'model',
        output_dir: str = None
    )

    def evaluate_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict
        """Comprehensive evaluation returning metrics dict."""

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'roc_auc'
    ) -> Dict
        """K-fold cross-validation."""

    def evaluate_fairness(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: List[str]
    ) -> Dict
        """Fairness metrics by demographic groups."""
```

## Usage Examples

### Example 1: Basic Training Pipeline

```python
from preprocessing.data_processor import DataProcessor
from models.logistic_model import LogisticModel
from evaluation.model_evaluator import ModelEvaluator

# 1. Load and prepare data
processor = DataProcessor()
df = processor.load_data(nrows=50000)
X, y = processor.prepare_features(df)

# 2. Split data
X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
    X, y, random_state=42
)

# 3. Train model
model = LogisticModel(random_state=42)
model.train(X_train, y_train, class_weight='balanced')

# 4. Evaluate
evaluator = ModelEvaluator(model, model_name='logistic_basic')
results = evaluator.evaluate_model(
    X_train, y_train, X_val, y_val, X_test, y_test
)

# 5. Print results
print(f"Test ROC-AUC: {results['test']['roc_auc']:.4f}")
print(f"Test Accuracy: {results['test']['accuracy']:.4f}")

# 6. Save model
model.save_model('data/output/my_model.pkl')
```

### Example 2: Temporal Validation

```python
# Use temporal split for realistic evaluation
X_train, X_val, X_test, y_train, y_val, y_test = processor.temporal_split(
    X, y, split_column='Year'
)

# Train and evaluate
model = RandomForestModel(n_estimators=200, max_depth=15)
model.train(X_train, y_train)

evaluator = ModelEvaluator(model, model_name='rf_temporal')
results = evaluator.evaluate_model(
    X_train, y_train, X_val, y_val, X_test, y_test
)
```

### Example 3: Hyperparameter Optimization

```python
from models.hyperparameter_tuning import HyperparameterOptimizer

# Prepare data
processor = DataProcessor()
df = processor.load_data(nrows=50000)
X, y = processor.prepare_features(df)
X_train, _, _, y_train, _, _ = processor.split_data(X, y)

# Optimize
optimizer = HyperparameterOptimizer()
results = optimizer.optimize_random_forest(
    X_train, y_train,
    search_type='random',
    n_iter=50,
    cv_folds=5
)

# Use best parameters
best_params = results['best_params']
model = RandomForestModel(**best_params)
model.train(X_train, y_train)
```

### Example 4: SMOTE for Class Imbalance

```python
# Train with SMOTE resampling (via CLI)
# python src/models/random_forest_model.py --nrows 20000 --resample smote

# Or programmatically:
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = RandomForestModel()
model.train(X_train_resampled, y_train_resampled)
```

### Example 5: Fairness Analysis

```python
# Train model
model = LogisticModel()
model.train(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)

# Evaluate fairness
evaluator = ModelEvaluator(model, model_name='fair_test')
fairness_results = evaluator.evaluate_fairness(
    X_test, y_test, y_pred,
    sensitive_features=['Victim Sex', 'Victim Race']
)

# Check positive rates by group
print(fairness_results)
```

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:**

```powershell
# Make sure to run from project root
cd C:\Users\hdmor\OneDrive\Documents\UWB\CSS581\archive

# Or add to PYTHONPATH
$env:PYTHONPATH = "C:\Users\hdmor\OneDrive\Documents\UWB\CSS581\archive\src"
```

#### 2. Data File Not Found

**Problem:** `FileNotFoundError: data/raw/data.csv not found`

**Solution:**

```python
# Specify custom data path
processor = DataProcessor(data_path='path/to/your/data.csv')
```

#### 3. Memory Issues

**Problem:** `MemoryError` when loading full dataset

**Solution:**

```python
# Use nrows to limit data
df = processor.load_data(nrows=50000)  # Load only 50k rows
```

#### 4. Pickle Loading Errors

**Problem:** `UnpicklingError` when loading saved models

**Solution:**

```python
# Ensure same Python version and scikit-learn version
# Re-train model if necessary
# Save with protocol=4 for backward compatibility
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)
```

#### 5. Scikit-learn Deprecation Warnings

**Problem:** `FutureWarning: needs_proba parameter deprecated`

**Solution:**

```python
# Use string scorer instead of make_scorer
from sklearn.model_selection import GridSearchCV

# OLD (deprecated):
# scorer = make_scorer(roc_auc_score, needs_proba=True)

# NEW (correct):
grid_search = GridSearchCV(model, param_grid, scoring='roc_auc')
```

### Performance Tips

1. **Use nrows for experiments:** Start with `nrows=10000` for quick iteration
2. **Random seed everywhere:** Use consistent `random_state=42` for reproducibility
3. **Cross-validation:** Use StratifiedKFold for imbalanced data
4. **Monitor overfitting:** Check train-test gap in metrics
5. **Save intermediate results:** Don't lose hours of hyperparameter tuning

### Getting Help

- **Tests:** Run `pytest tests/ -v` to verify setup
- **Documentation:** Check `docs/` folder for detailed guides
- **Issues:** See `docs/version-control.md` for contribution guidelines

---

**Document Version:** 1.0  
**Last Updated:** December 3, 2025  
**Maintainer:** CSS 581 Project Team
