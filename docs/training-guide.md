# Murder Model Training Pipeline

## Quick Start

### Training Logistic Regression (Default)

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Train logistic regression on first 10,000 rows
python src/models/train.py --nrows 10000

# Or explicitly specify the model
python src/models/train.py --model logistic --nrows 10000
```

### Training Random Forest

```powershell
# Train random forest on first 10,000 rows
python src/models/train.py --model random_forest --nrows 10000
```

### Comparing Both Models

```powershell
# Compare logistic regression and random forest side-by-side
python src/models/compare.py --nrows 10000

# Compare on full dataset
python src/models/compare.py
```

### Training on Full Dataset

```powershell
# Logistic Regression
python src/models/train.py --model logistic

# Random Forest
python src/models/train.py --model random_forest
```

### Command Line Options (train.py)

```
--data PATH           Path to data CSV file (default: data/raw/data.csv)
--target COLUMN       Name of target column (default: Crime Solved)
--model TYPE          Model type: 'logistic' or 'random_forest' (default: logistic)
--nrows N            Number of rows to load (default: None = all)
--output-dir PATH    Output directory for results (default: data/output)
--random-state SEED  Random seed for reproducibility (default: 42)
--no-save            Skip saving model to disk
```

### Command Line Options (compare.py)

```
--data PATH           Path to data CSV file (default: data/raw/data.csv)
--target COLUMN       Name of target column (default: Crime Solved)
--models TYPE [TYPE]  Models to compare (default: logistic random_forest)
--nrows N            Number of rows to load (default: None = all)
--output-dir PATH    Output directory for results (default: data/output)
--random-state SEED  Random seed for reproducibility (default: 42)
```

## Pipeline Steps

The training pipeline performs the following steps automatically:

1. **Data Loading & Preprocessing**

   - Loads data from CSV
   - Converts month names to numbers
   - Handles missing values
   - Engineers features (seasons, age groups)
   - Encodes categorical variables
   - Scales numeric features
   - Filters to clean numeric data

2. **Model Training**

   - Splits data: 70% train, 10% validation, 20% test
   - Stratified sampling to preserve class distribution
   - Trains selected model (Logistic Regression or Random Forest)
   - **Logistic Regression**: Uses StandardScaler + L2 regularization (C=1.0)
   - **Random Forest**: 100 trees, Gini importance, parallel processing

3. **Model Evaluation**

   - Computes metrics: accuracy, precision, recall, F1, ROC-AUC
   - Evaluates on train, validation, and test sets
   - Saves evaluation report as JSON

4. **Visualizations**

   - Confusion matrix
   - ROC curve
   - Feature importance plot (coefficients for Logistic, Gini importance for Random Forest)
   - All saved as PNG files with model-specific filenames

5. **Model Persistence**
   - Saves trained model as pickle file
   - Includes timestamp in filename

## Output Files

### Training Pipeline (train.py)

All outputs are saved to `data/output/` (or specified directory):

**Logistic Regression:**

- `logisticmodel_evaluation_YYYYMMDD_HHMMSS.json` - Evaluation metrics
- `confusion_matrix_logistic_YYYYMMDD_HHMMSS.png` - Confusion matrix
- `roc_curve_logistic_YYYYMMDD_HHMMSS.png` - ROC curve
- `feature_importance_logistic_YYYYMMDD_HHMMSS.png` - Feature coefficients
- `logisticmodel_YYYYMMDD_HHMMSS.pkl` - Saved model

**Random Forest:**

- `randomforestmodel_evaluation_YYYYMMDD_HHMMSS.json` - Evaluation metrics
- `confusion_matrix_random_forest_YYYYMMDD_HHMMSS.png` - Confusion matrix
- `roc_curve_random_forest_YYYYMMDD_HHMMSS.png` - ROC curve
- `feature_importance_random_forest_YYYYMMDD_HHMMSS.png` - Feature importances
- `randomforestmodel_YYYYMMDD_HHMMSS.pkl` - Saved model

### Comparison Pipeline (compare.py)

- `model_comparison_YYYYMMDD_HHMMSS.csv` - Comparison table (all metrics)
- `model_comparison_YYYYMMDD_HHMMSS.json` - Detailed comparison data

## Example Output

```
################################################################################
MURDER MODEL - END-TO-END TRAINING PIPELINE
################################################################################

STEP 1: Loading and Preprocessing Data
  Loaded: 10000 rows, 24 columns
  Clean samples: 10000 / 10000 (100.0%)

STEP 2: Training Model
  Train: 7000 samples
  Val:   1000 samples
  Test:  2000 samples

STEP 3: Model Evaluation
  TEST SET:
    accuracy       : 1.0000
    precision      : 1.0000
    recall         : 1.0000
    f1             : 1.0000
    roc_auc        : 1.0000

STEP 4: Generating Visualizations
  ✓ Confusion matrix saved
  ✓ ROC curve saved
  ✓ Feature importance saved

STEP 5: Saving Model
  ✓ Model saved to: data/output/logistic_model_YYYYMMDD_HHMMSS.pkl

TRAINING PIPELINE COMPLETED SUCCESSFULLY!
```

## Testing

Run all tests:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Run specific test suites:

```powershell
# Preprocessing tests
python -m unittest tests.test_preprocessing -v

# Model tests
python -m unittest tests.test_logistic_model -v

# Integration tests (uses real data)
python -m unittest tests.test_integration -v
```

## Project Structure

```
archive/
├── src/
│   ├── preprocessing/
│   │   └── data_processor.py      # Data preprocessing pipeline
│   ├── models/
│   │   ├── base_model.py          # Abstract base model class
│   │   ├── logistic_model.py      # Logistic regression implementation
│   │   └── train.py               # End-to-end training script
│   └── evaluation/
│       └── evaluator.py           # Model evaluation utilities
├── tests/
│   ├── test_preprocessing.py      # Preprocessing tests
│   ├── test_logistic_model.py     # Model tests
│   └── test_integration.py        # Integration tests
├── data/
│   ├── raw/
│   │   └── data.csv               # Raw data
│   └── output/                    # Training outputs
└── README.md
```

## Phase 2 Completion

✅ **All Phase 2 objectives completed:**

- Data splitting with stratification
- Baseline logistic regression model
- Comprehensive evaluation metrics
- Model training and validation
- Visualization and reporting
- Model persistence
- Full test coverage (14/14 tests passing)
