# Murder Model Training Pipeline

## Quick Start

### Training on Sample Data

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Train on first 10,000 rows
python src/models/train.py --nrows 10000
```

### Training on Full Dataset

```powershell
python src/models/train.py
```

### Command Line Options

```
--data PATH           Path to data CSV file (default: data/raw/data.csv)
--target COLUMN       Name of target column (default: Crime Solved)
--nrows N            Number of rows to load (default: None = all)
--output-dir PATH    Output directory for results (default: data/output)
--random-state SEED  Random seed for reproducibility (default: 42)
--no-save            Skip saving model to disk
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
   - Trains Logistic Regression model with StandardScaler

3. **Model Evaluation**

   - Computes metrics: accuracy, precision, recall, F1, ROC-AUC
   - Evaluates on train, validation, and test sets
   - Saves evaluation report as JSON

4. **Visualizations**

   - Confusion matrix
   - ROC curve
   - Feature importance plot
   - All saved as PNG files

5. **Model Persistence**
   - Saves trained model as pickle file
   - Includes timestamp in filename

## Output Files

All outputs are saved to `data/output/` (or specified directory):

- `logistic_model_evaluation_YYYYMMDD_HHMMSS.json` - Evaluation metrics
- `confusion_matrix_YYYYMMDD_HHMMSS.png` - Confusion matrix visualization
- `roc_curve_YYYYMMDD_HHMMSS.png` - ROC curve plot
- `feature_importance_YYYYMMDD_HHMMSS.png` - Feature importance plot
- `logistic_model_YYYYMMDD_HHMMSS.pkl` - Saved model

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
