# Project Roadmap - Murder Model (CSS 581)

## Project Overview

A machine learning classification project predicting crime outcomes (solved/unsolved) based on historical homicide data from 1980-2014 using demographic, geographic, and temporal features.

## Phase 0: Project Initialization (Week 1-2) ✅ COMPLETE

- [x] Project proposal submission
- [x] Define project scope and objectives
- [x] Initial research on similar works
- [x] Establish project timeline
- [x] Set up development environment
  - [x] Python virtual environment (.venv)
  - [x] Git repository structure
  - [x] Required packages (sklearn, pandas, numpy, matplotlib, seaborn)
- [x] Project directory structure established
  - [x] src/ (preprocessing, models, evaluation)
  - [x] tests/ (unit and integration tests)
  - [x] data/ (raw and output directories)
  - [x] docs/ (documentation)
  - [x] reports/ (ACM format papers)

## Phase 1: Data Collection & Preprocessing (Week 3-4) ✅ COMPLETE

- [x] Gather historical homicide data (1980-2014)
  - [x] Dataset: 24 features including demographics, location, temporal data
  - [x] Target variable: Crime Solved (Yes/No)
  - [x] Class distribution: ~71% solved, ~29% unsolved
- [x] Data cleaning and validation
  - [x] Type checking and conversion
  - [x] Month name to numeric mapping
  - [x] Missing value detection
- [x] Data preprocessing pipeline (`DataProcessor` class)
  - [x] Missing value imputation (mean for numeric, mode for categorical)
  - [x] Outlier detection and handling
  - [x] Categorical encoding (Label Encoding)
  - [x] Numeric scaling (StandardScaler with zero-std handling)
- [x] Feature engineering
  - [x] Season feature from month
  - [x] Age group binning (victim and perpetrator)
- [x] Comprehensive testing
  - [x] 8 unit tests for preprocessing pipeline
  - [x] All tests passing

## Phase 2: Model Development (Week 5-7) ✅ COMPLETE

- [x] Architecture design
  - [x] Abstract `BaseModel` class with common functionality
  - [x] Stratified train/validation/test splitting
  - [x] Model persistence (save/load)
  - [x] Evaluation metrics integration
- [x] **Algorithm 1: Logistic Regression** (Baseline)
  - [x] `LogisticModel` implementation
  - [x] StandardScaler preprocessing pipeline
  - [x] L2 regularization (C=1.0)
  - [x] LBFGS solver
  - [x] Probability predictions (predict_proba)
  - [x] Feature importance extraction
- [x] Data splitting strategy
  - [x] 70% training (7,000 samples)
  - [x] 10% validation (1,000 samples)
  - [x] 20% test (2,000 samples)
  - [x] Stratified sampling to preserve class balance
- [x] Model training pipeline
  - [x] Automated end-to-end script (`train.py`)
  - [x] Command-line interface
  - [x] Progress reporting
  - [x] Timestamped outputs
- [x] Comprehensive testing
  - [x] 3 unit tests for LogisticModel
  - [x] 3 integration tests with real data
  - [x] All 14 tests passing

## Phase 2.5: Evaluation Framework (Week 7) ✅ COMPLETE

- [x] `ModelEvaluator` class implementation
  - [x] Multiple metrics: Accuracy, Precision, Recall, F1, ROC-AUC
  - [x] Evaluation on train/val/test sets
  - [x] JSON report export with timestamps
- [x] Visualizations
  - [x] Confusion matrix (saved as PNG)
  - [x] ROC curve with AUC score
  - [x] Feature importance plot (top 20 features)
- [x] Results documentation
  - [x] Automated reporting
  - [x] Formatted console output
  - [x] Classification report generation

## Phase 2.9: Progress Report (Week 8) ✅ COMPLETE

- [x] ACM format paper (`ProgressReport.tex`)
  - [x] Abstract and introduction
  - [x] Methodology section (preprocessing, model, evaluation)
  - [x] Implementation details
  - [x] Results with performance table
  - [x] All 3 figures included (confusion matrix, ROC, feature importance)
  - [x] Discussion of perfect accuracy concerns (data leakage analysis)
  - [x] Testing and validation section
  - [x] Future work roadmap
  - [x] Maximum 3 pages as per requirements
- [x] Progress report deliverables
  - [x] PDF compilation ready
  - [x] Demonstrates significant progress
  - [x] Includes flowchart/system design
  - [x] Documents completed work (Phases 0, 1, 2)

## Phase 3: Second Algorithm & Comparison (Week 9-10) 🔄 NEXT

- [ ] **Algorithm 2: Random Forest** (or Gradient Boosting)
  - [ ] `RandomForestModel` class implementation
  - [ ] Hyperparameter configuration
  - [ ] Feature importance analysis
  - [ ] Comparison with logistic regression
- [ ] Alternative Algorithm 2: Neural Network
  - [ ] Simple MLP implementation (if choosing this path)
  - [ ] Architecture design
  - [ ] Training with early stopping
- [ ] Cross-validation implementation
  - [ ] K-fold CV (k=5 or k=10)
  - [ ] Stratified CV for class balance
  - [ ] Performance variance analysis
- [ ] Model comparison framework
  - [ ] Side-by-side performance tables
  - [ ] Statistical significance testing
  - [ ] Visualization of algorithm comparisons

## Phase 4: Data Investigation & Model Refinement (Week 10-11)

- [ ] **Data leakage investigation** (Critical Priority)
  - [ ] Audit features for post-hoc information
  - [ ] Identify perpetrator-related feature issues
  - [ ] Remove/modify problematic features
  - [ ] Re-train models on cleaned feature set
  - [ ] Document performance changes
- [ ] Feature selection and engineering
  - [ ] Correlation analysis
  - [ ] Feature importance ranking
  - [ ] Dimensionality reduction (PCA if needed)
  - [ ] New feature creation based on domain knowledge
- [ ] Hyperparameter optimization
  - [ ] Grid search for both algorithms
  - [ ] Bayesian optimization (optional)
  - [ ] Cross-validation for hyperparameter selection
- [ ] Class imbalance handling
  - [ ] SMOTE or ADASYN investigation
  - [ ] Class weight adjustment
  - [ ] Threshold tuning for precision/recall trade-off
- [ ] Advanced validation
  - [ ] Temporal validation (time-based splits)
  - [ ] Geographic validation (location-based splits)
  - [ ] Robustness testing

## Phase 5: Analysis & Final Documentation (Week 12-13)

- [ ] Comprehensive algorithm comparison
  - [ ] Performance metrics across all models
  - [ ] Computational efficiency analysis
  - [ ] Interpretability comparison
  - [ ] Use case recommendations
- [ ] Visualization suite
  - [ ] Learning curves
  - [ ] Precision-recall curves
  - [ ] Feature importance comparisons
  - [ ] Error analysis visualizations
- [ ] Interpretability analysis
  - [ ] SHAP values (optional advanced)
  - [ ] LIME explanations (optional advanced)
  - [ ] Feature interaction analysis
- [ ] Documentation
  - [ ] Code documentation and docstrings
  - [ ] API reference
  - [ ] Usage examples and tutorials
  - [ ] Troubleshooting guide

## Phase 6: Final Delivery (Week 14-15)

- [ ] Final paper (ACM format, max 15 pages)
  - [ ] Abstract
  - [ ] Introduction with background and related work
  - [ ] Method & Approach (preprocessing, models, evaluation)
  - [ ] Experiment Details and Results
  - [ ] Discussion of findings and limitations
  - [ ] Conclusion and Future Work
  - [ ] References (scientific papers, no wikis)
  - [ ] Appendix for supplementary materials
- [ ] Final presentation preparation
  - [ ] Demo of trained models
  - [ ] Results visualization
  - [ ] Key findings and insights
  - [ ] Lessons learned
- [ ] Code and data packaging
  - [ ] Original dataset (or download link)
  - [ ] All source code files
  - [ ] README with tech specs and reproduction steps
  - [ ] Requirements.txt with dependencies
  - [ ] Screenshots/instructions for running
  - [ ] Trained model files
- [ ] Final model optimization
  - [ ] Best model selection based on validation
  - [ ] Final hyperparameter settings
  - [ ] Performance benchmarking
  - [ ] Deployment considerations (optional)
- [ ] Code cleanup
  - [ ] Remove unused code
  - [ ] Consistent formatting
  - [ ] Complete test coverage
  - [ ] Code review and refactoring

## Key Milestones

1. ✅ Project Proposal (Week 1-2)
2. ✅ Initial Data Processing (Week 3-4)
3. ✅ First Algorithm Implementation - Logistic Regression (Week 5-7)
4. ✅ Progress Report Submission (Week 8)
5. 🔄 Second Algorithm Implementation (Week 9-10)
6. ⏳ Model Optimization & Comparison (Week 10-11)
7. ⏳ Final Presentation (Week 14)
8. ⏳ Final Project Documentation (Week 15)

## Success Metrics

### Technical Metrics

1. ✅ Model Accuracy: Baseline achieved (100% - needs investigation)
2. ✅ Multiple Algorithms: 1/2 complete (Logistic Regression done)
3. 🔄 Cross-validation: Planned for Phase 3
4. ✅ Reproducible Results: Random seeds implemented
5. ✅ Clear Methodology: Documented in code and paper
6. ⏳ Comparative Analysis: Pending second algorithm

### Quality Metrics

1. ✅ Code Quality: Modular, tested, documented
2. ✅ Test Coverage: 14/14 tests passing
3. ✅ Documentation: Comprehensive docstrings and guides
4. ✅ Version Control: Git repository with clear history
5. ✅ Ethical Compliance: Addressed in discussions
6. ✅ Paper Format: ACM template correctly used

### Project Requirements Compliance

- ✅ Real-world problem: Historical crime data
- ✅ Classification problem: Binary (solved/unsolved)
- ✅ At least one algorithm: Logistic Regression implemented
- ⏳ At least two algorithms: Second algorithm pending (Phase 3)
- ✅ Performance comparison: Framework ready, awaiting second model
- ✅ Progress report: 3-page ACM format paper complete
- ⏳ Final paper: Planned for Week 14-15 (max 15 pages)
- ✅ Code submission package: Structure ready, needs final packaging

## Current Status Summary (Week 8)

**Completed:**

- Full preprocessing pipeline with robust error handling
- Baseline logistic regression model with perfect test accuracy
- Comprehensive evaluation framework with visualizations
- 14 passing tests (unit + integration)
- Progress report in ACM format with all required sections
- Automated end-to-end training pipeline

**In Progress:**

- Investigating perfect accuracy (potential data leakage)
- Planning second algorithm implementation

**Next Steps:**

1. Implement second algorithm (Random Forest or Gradient Boosting)
2. Investigate and resolve data leakage issues
3. Perform thorough model comparison
4. Hyperparameter tuning
5. Prepare for final paper and presentation

## Notes and Considerations

### Critical Issues to Address

1. **Perfect Accuracy Concern**: The 100% accuracy suggests data leakage. Perpetrator features may only be available for solved cases. Need to audit features and potentially retrain without problematic features.
2. **Temporal Dependencies**: Current random split may not respect time ordering. Consider time-based splitting for more realistic evaluation.
3. **Feature Engineering**: More domain-specific features could improve generalization.

### Technical Debt

- Consider adding logging framework for better debugging
- Implement model versioning system
- Add configuration file support for hyperparameters
- Create Docker container for reproducibility

### Future Enhancements

- Web interface for predictions (optional)
- API endpoint for model serving (optional)
- Real-time prediction capabilities (optional)
- Geographic visualization of predictions (optional)
