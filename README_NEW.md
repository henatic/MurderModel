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

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Verify installation
python -m pytest tests/ -v  # Should see 21 tests passing
```

```bash
# Linux/macOS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify installation
python -m pytest tests/ -v  # Should see 21 tests passing
```

### 2. Train Models

```powershell
# Train Logistic Regression (quick test)
python src/models/logistic_model.py --nrows 10000

# Train Random Forest (quick test)
python src/models/random_forest_model.py --nrows 10000

# Train with class weights for imbalance
python src/models/random_forest_model.py --nrows 20000 --class-weight balanced_subsample

# Train with SMOTE resampling
python src/models/random_forest_model.py --nrows 20000 --resample smote
```

### 3. Compare Models

```powershell
# Compare both models on same data split
python src/models/compare.py --nrows 20000

# Results saved to data/output/model_comparison_*.json
```

### 4. Generate Visualizations

```powershell
# Generate Phase 5 publication-quality visualizations
python src/evaluation/phase5_visualizations.py

# Outputs: performance comparison, overfitting analysis, metric radar, summary table
```

### 5. Advanced Usage

```powershell
# Temporal validation (train on early years, test on later years)
python src/models/random_forest_model.py --split-strategy temporal --nrows 50000

# Geographic validation (train on some states, test on others)
python src/models/random_forest_model.py --split-strategy geo --geo-column State --nrows 50000

# Hyperparameter optimization
python src/models/hyperparameter_tuning.py --nrows 30000 --models random_forest --search-type random --n-iter 50 --cv-folds 5
```

## 📊 Model Performance

### Summary (Post-Leakage Mitigation)

| Model | Test ROC-AUC | Test Accuracy | Precision | Recall | F1 Score | Overfitting Gap |
|-------|--------------|---------------|-----------|--------|----------|-----------------|
| **Random Forest** | **0.684** | 0.732 | 0.778 | 0.887 | 0.829 | 31.4% |
| **Logistic Regression** | 0.579 | 0.732 | 0.732 | 1.000 | 0.845 | 0.08% |

**Key Insights:**
- Random Forest achieves best discrimination (ROC-AUC 0.684)
- Logistic Regression offers perfect recall and excellent stability
- Geographic validation is hardest (both models ~0.58 ROC-AUC)
- Expect 0.58-0.68 ROC-AUC in production deployment

### Validation Strategy Results

| Strategy | Logistic ROC-AUC | Random Forest ROC-AUC |
|----------|------------------|----------------------|
| Random Split (70/10/20) | 0.579 | 0.684 |
| Temporal Split (by Year) | ~0.58 | ~0.64 |
| Geographic Split (by State) | ~0.58 | ~0.58 |

See `docs/phase5-algorithm-comparison.md` for detailed analysis.

## 📚 Documentation

### Project Documentation
- **[Project Roadmap](docs/project-roadmap.md)** - Complete project timeline (Phases 0-5 complete)
- **[API Documentation](docs/API-Documentation.md)** - Complete API reference with examples
- **[Training Guide](docs/training-guide.md)** - Detailed training instructions

### Phase 5 Analysis (Complete)
- **[Algorithm Comparison](docs/phase5-algorithm-comparison.md)** - Performance, efficiency, interpretability analysis
- **[Interpretability Analysis](docs/phase5-interpretability.md)** - Error patterns, feature importance, fairness
- **[Phase 5 Summary](docs/phase5-summary.md)** - Complete Phase 5 deliverables and findings

### Additional Resources
- **[Development Phases](docs/development-phases.md)** - Development methodology
- **[Version Control](docs/version-control.md)** - Git workflow and conventions
- **[Progress Diagrams](.context/diagrams/)** - Mermaid diagrams (progress, architecture, data-flow)

## 🔬 Key Features

### Ethical Data Handling
- **Leakage Mitigation**: Removed perpetrator information, relationship, crime type, record source
- **Fairness Monitoring**: Regular audits across demographic groups
- **Transparency**: Documented limitations and biases

### Advanced Techniques
- **Multiple Validation Strategies**: Random, temporal, geographic splits
- **Class Imbalance Handling**: SMOTE, ADASYN, class weights
- **Hyperparameter Optimization**: Grid and randomized search with cross-validation
- **Comprehensive Visualization**: 15+ publication-quality plots

### Robust Engineering
- **21 Automated Tests**: Full test coverage with pytest
- **Modular Architecture**: Clean separation of concerns
- **Type Hints**: Type annotations throughout codebase
- **Documentation**: Complete API reference and examples

## 🧪 Testing

```powershell
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_logistic_model.py -v
python -m pytest tests/test_random_forest_model.py -v
python -m pytest tests/test_integration.py -v
python -m pytest tests/test_compare.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

**Test Coverage:**
- ✅ Data preprocessing (8 tests)
- ✅ Logistic Regression model (5 tests)
- ✅ Random Forest model (5 tests)
- ✅ Integration tests (4 tests)
- ✅ Model comparison (1 test)
- **Total: 21 tests passing**

## 🛠️ Troubleshooting

### Common Issues

**ModuleNotFoundError:**
```powershell
# Ensure you're running from project root
cd C:\Users\hdmor\OneDrive\Documents\UWB\CSS581\archive

# Or add to PYTHONPATH
$env:PYTHONPATH = "C:\Users\hdmor\OneDrive\Documents\UWB\CSS581\archive\src"
```

**Memory Issues:**
```python
# Use nrows to limit data
python src/models/random_forest_model.py --nrows 10000  # Smaller sample
```

**Scikit-learn Warnings:**
- Ensure scikit-learn >= 1.0
- Use `scoring='roc_auc'` instead of deprecated `make_scorer` with `needs_proba`

See `docs/API-Documentation.md` for complete troubleshooting guide.

## 📈 Project Phases

- ✅ **Phase 0**: Project Initialization
- ✅ **Phase 1**: Data Collection & Preprocessing
- ✅ **Phase 2**: Model Development (Logistic Regression)
- ✅ **Phase 2.5**: Evaluation Framework
- ✅ **Phase 2.9**: Progress Report
- ✅ **Phase 3**: Second Algorithm (Random Forest) & Comparison
- ✅ **Phase 4**: Data Investigation & Model Refinement
- ✅ **Phase 5**: Analysis & Final Documentation
- 🔄 **Phase 6**: Final Delivery (In Progress)

## 🎓 Academic Context

**Course:** CSS 581 - Machine Learning  
**Institution:** University of Washington Bothell  
**Term:** Fall 2025  
**Focus:** Responsible ML with emphasis on ethics, fairness, and interpretability

## 📝 Citation

If you use this project or its findings, please cite:

```bibtex
@misc{murdermodel2025,
  author = {[Your Name]},
  title = {Murder Model: Ethical Classification of Historical Homicide Cases},
  year = {2025},
  institution = {University of Washington Bothell},
  course = {CSS 581}
}
```

## 🤝 Contributing

This is an academic project, but suggestions and feedback are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See `docs/version-control.md` for detailed git workflow.

## 📄 License

This project is developed for educational purposes as part of CSS 581 at University of Washington Bothell.

## 🙏 Acknowledgments

- University of Washington Bothell CSS 581 course staff
- Scikit-learn and pandas development teams
- Historical homicide data contributors

## 📧 Contact

For questions or discussions about this project, please open an issue in the repository.

---

**Last Updated:** December 3, 2025  
**Status:** Phase 6 (Final Delivery) - In Progress  
**Tests:** 21/21 Passing ✅
