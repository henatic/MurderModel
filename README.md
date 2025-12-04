# Murder Model: Homicide Case Classification

[![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

A machine learning project for classifying historical homicide cases (1980-2014) as solved or unsolved using demographic, geographic, and temporal features. Developed for CSS 581 at University of Washington Bothell.

## 🎯 Project Overview

This project implements and compares two classification algorithms (Logistic Regression and Random Forest) for predicting case resolution outcomes while maintaining ethical standards through comprehensive data leakage mitigation.

**Key Features:**

- ✅ Ethical leakage mitigation (removed perpetrator information, case details)
- ✅ Multiple validation strategies (random, temporal, geographic)
- ✅ Class imbalance handling (SMOTE, ADASYN, class weights)
- ✅ Comprehensive fairness monitoring
- ✅ Hyperparameter optimization with cross-validation
- ✅ 15+ publication-quality visualizations
- ✅ Complete API documentation

**Performance (Post-Leakage Mitigation):**

- Random Forest: ROC-AUC 0.684 (test), best discriminative performance
- Logistic Regression: ROC-AUC 0.579 (test), highly interpretable and stable
- All 21 automated tests passing

## 📁 Project Structure

```
archive/
├── data/
│   ├── raw/              # Original dataset (data.csv)
│   ├── processed/        # Cleaned and preprocessed data
│   └── output/           # Model outputs, metrics, visualizations (150+ files)
├── docs/                 # Comprehensive documentation
│   ├── project-roadmap.md              # Project timeline and phases
│   ├── phase5-algorithm-comparison.md  # Model comparison analysis
│   ├── phase5-interpretability.md      # Error analysis and interpretability
│   ├── phase5-summary.md               # Phase 5 summary
│   ├── API-Documentation.md            # Complete API reference
│   └── ...
├── notebooks/            # Jupyter notebooks for exploration
│   └── 01_data_exploration.ipynb
├── src/
│   ├── preprocessing/    # Data processing pipeline
│   │   └── data_processor.py
│   ├── models/          # Model implementations
│   │   ├── base_model.py
│   │   ├── logistic_model.py
│   │   ├── random_forest_model.py
│   │   ├── compare.py
│   │   └── hyperparameter_tuning.py
│   ├── evaluation/      # Evaluation and visualization
│   │   ├── model_evaluator.py
│   │   ├── visualizations.py
│   │   └── phase5_visualizations.py
│   └── utils/           # Utility functions
│       ├── feature_audit.py
│       └── feature_analysis.py
├── tests/               # Automated test suite (21 tests)
│   ├── test_preprocessing.py
│   ├── test_logistic_model.py
│   ├── test_random_forest_model.py
│   ├── test_integration.py
│   └── test_compare.py
├── .context/            # Project context and diagrams
│   └── diagrams/        # Mermaid diagrams (progress, architecture, data-flow)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup

1. Create and activate the virtual environment (`.venv`):

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Unix/MacOS
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

> `.venv` is the only supported environment; any previous `venv/` folder has been removed.

## Usage

```bash
# Activate environment
.\.venv\Scripts\activate        # Windows
source .venv/bin/activate      # Unix/MacOS

# Train models (random split by default)
python src/models/train.py --model logistic --nrows 10000
python src/models/train.py --model random_forest --nrows 10000

# Alternate splits / imbalance toggles
python src/models/train.py --model random_forest --split-strategy temporal --nrows 50000 --class-weight balanced_subsample
python src/models/train.py --model random_forest --split-strategy geo --geo-column State --nrows 50000 --class-weight balanced_subsample
python src/models/train.py --model logistic --split-strategy geo --nrows 50000 --class-weight balanced
python src/models/train.py --model random_forest --split-strategy random --nrows 20000 --resample smote   # requires imbalanced-learn
python src/models/train.py --model random_forest --split-strategy random --nrows 20000 --resample adasyn  # requires imbalanced-learn

# Compare models
python src/models/compare.py --nrows 10000

# Run tests
python -m unittest discover -s tests -p "test_*.py" -v

## Troubleshooting
- Cross-validation/learning curves are now fixed via classifier tagging; if you see scorer errors, upgrade `scikit-learn` and ensure you’re using the current `src/models/base_model.py`.
- Error-analysis plots and comparison metrics live in `data/output/`; see `docs/phase5-summary.md` and `docs/phase5-algorithm-comparison.md` for paths.
```

## Documentation

- Project proposal: `docs/projectproposal.md`
- Development phases: `docs/development-phases.md`
- Project roadmap: `docs/project-roadmap.md`
- Version control guidelines: `docs/version-control.md`
- Phase 5 plan + summary: `docs/phase5-plan.md`, `docs/phase5-summary.md`
- Latest comparison metrics: `docs/phase5-algorithm-comparison.md`
- Interpretability status: `docs/phase5-interpretability.md` (currently deferred)

## Contributing

1. Create a new branch for your feature (see `docs/version-control.md` for naming/commit conventions)
2. Make your changes
3. Submit a pull request

## License

[To be determined]
