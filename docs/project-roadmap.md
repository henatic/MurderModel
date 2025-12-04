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

- [x] Gather historical homicide data (1980-2014) — 24 features incl. demographics, location, temporal; target: Crime Solved (Yes/No); class distribution ~71% solved / 29% unsolved.
- [x] Data cleaning and validation — type checking, month name to numeric, missing value detection.
- [x] Data preprocessing pipeline (`DataProcessor`):
  - [x] Missing value imputation (median/mode)
  - [x] Outlier detection/handling
  - [x] Categorical encoding (Label Encoding)
  - [x] Numeric scaling (safe zero-std handling)
- [x] Feature engineering — season from month; age-group binning.
- [x] Comprehensive testing — 8 preprocessing tests; all passing.

## Phase 2: Model Development (Week 5-7) ✅ COMPLETE

- [x] Architecture design — `BaseModel`, stratified splits, persistence, metrics integration.
- [x] Algorithm 1: Logistic Regression (baseline)
  - [x] `LogisticModel` implementation, L2 regularization, LBFGS, predict_proba, feature importance.
- [x] Data splitting strategy — 70/10/20 train/val/test stratified.
- [x] Model training pipeline — CLI, progress reporting, timestamped outputs.
- [x] Comprehensive testing — logistic model + integration tests.

## Phase 2.5: Evaluation Framework (Week 7) ✅ COMPLETE

- [x] `ModelEvaluator` class — accuracy, precision, recall, F1, ROC-AUC; JSON export; confusion matrix, ROC, feature importance plots; classification report.

## Phase 2.9: Progress Report (Week 8) ✅ COMPLETE

- [x] ACM-format progress paper (≤3 pages) with preprocessing, model, evaluation, figures, results, discussion of leakage concerns, roadmap.

## Phase 3: Second Algorithm & Comparison (Week 9-10) ✅ COMPLETE

- [x] Algorithm 2: Random Forest (`RandomForestModel`) with configurable hyperparameters and feature importances.
- [x] Model comparison framework (`compare.py`) — side-by-side metrics, CSV/JSON exports, shared splits (now leakage-safe).
- [x] Training pipeline updates — multi-model support, model-specific outputs/titles, no double-scaling.
- [x] Comprehensive testing — random forest unit tests; compare pipeline test added; total tests now 21/21 passing.
- [x] Results documentation — Dec 3, 2025 sample (20k rows, perpetrator features removed):
  - Logistic: test acc 0.7550, ROC-AUC 0.8303; CV (3-fold) acc 0.7461 ± 0.0109, F1 0.8329 ± 0.0067.
  - Random Forest: test acc 0.8540, ROC-AUC 0.9120; CV (3-fold) acc 0.8594 ± 0.0025, F1 0.9010 ± 0.0017.
  - Fairness signal (test): Victim Sex pos-rate Female 0.739 / Male 0.710; Victim Race ranges 0.556–0.730 across groups.

## Phase 4: Data Investigation & Model Refinement (Week 10-11) ✅ COMPLETE

- [x] Data leakage investigation (Critical)
  - [x] Initial perpetrator-feature removal in preprocessing
  - [x] Comprehensive feature audit identifying 7 leakage/low-value features
  - [x] Relationship field identified as high-correlation (post-hoc knowledge)
  - [x] Re-trained models on cleaned feature set; documented performance impact
- [x] Feature selection and engineering
  - [x] Correlation analysis; importance ranking framework created
  - [x] Zero highly correlated feature pairs (no redundancy)
  - [x] Correlation heatmap visualization generated
- [x] Hyperparameter optimization
  - [x] Cross-validation framework (3-fold and 5-fold) implemented
  - [x] Grid/Random Search infrastructure with ROC-AUC scorer created
  - [x] Full hyperparameter search runs completed (Logistic: ROC-AUC 0.5846; RF: ROC-AUC 0.7206)
- [x] Class imbalance handling — class weights integrated into models and CLI; SMOTE/ADASYN utilities created
- [x] Advanced validation — temporal splits (by Year); geographic splits (by State) implemented and tested
- [x] Visualization suite — learning curves, PR curves, ROC comparisons, threshold analysis, confusion matrices generated

## Phase 5: Analysis & Final Documentation (Week 12-13)

- [ ] Comprehensive algorithm comparison (performance, efficiency, interpretability, recommendations)
- [ ] Visualization suite — learning curves; PR curves; feature importance comparisons; error analysis
- [ ] Interpretability — SHAP/LIME (optional); feature interactions
- [ ] Documentation — code docs, API reference, examples, troubleshooting

## Phase 6: Final Delivery (Week 14-15)

- [ ] Final paper (ACM, ≤15 pages): abstract, intro, method, experiments/results, discussion/limitations, conclusion/future work, references.
- [ ] Final presentation: demo, results visuals, key findings.
- [ ] Code/data packaging: dataset/link, source, README, requirements, run steps/screenshots, trained models.
- [ ] Final model optimization; benchmarking; deployment considerations (optional).
- [ ] Code cleanup: remove unused code, formatting, full test coverage, code review/refactoring.

## Key Milestones

1. ✅ Project Proposal (Week 1-2)
2. ✅ Initial Data Processing (Week 3-4)
3. ✅ First Algorithm (Logistic Regression) (Week 5-7)
4. ✅ Progress Report Submission (Week 8)
5. ✅ Second Algorithm (Random Forest) (Week 9-10)
6. ✅ Model Optimization & Comparison (Week 10-11) - PHASE 4 COMPLETE
7. 🔄 Final Analysis & Documentation (Week 12-13) - PHASE 5 IN PROGRESS
8. 🟦 Final Presentation (Week 14)
9. 🟦 Final Project Documentation (Week 15)

## Success Metrics

### Technical

1. Model Accuracy: ~0.85 test (RF) after leakage mitigation; further improvement post-feature audit.
2. Multiple Algorithms: 2/2 (Logistic + Random Forest).
3. Cross-validation: wired and executed (3-fold); expand with tuning.
4. Reproducible Results: random seeds set.
5. Clear Methodology: documented in code and paper.
6. Comparative Analysis: framework implemented with CSV/JSON outputs.

### Quality

1. Code Quality: modular, tested, documented.
2. Test Coverage: 21/21 tests passing.
3. Documentation: guides and summaries updated.
4. Version Control: Git history maintained.
5. Ethical/Fairness: initial fairness signal added; deeper analysis pending.
6. Paper Format: ACM template used for progress report.

### Requirements Compliance

- Real-world problem: historical crime data.
- Classification: binary (solved/unsolved).
- At least two algorithms: Logistic Regression + Random Forest.
- Performance comparison: CSV/JSON outputs.
- Progress report: 3-page ACM format complete.
- Final paper: planned for Week 14-15 (max 15 pages).
- Code package: structure ready; final packaging pending.

## Current Status Summary (Week 10)

**Completed:**

- Preprocessing pipeline with comprehensive leakage mitigation (7 features removed total), robust handling.
- Logistic + Random Forest models; evaluation framework; comparison framework.
- CV and fairness reporting integrated; 21 tests passing.
- Progress report delivered; training pipeline supports multiple models.
- **Phase 4 COMPLETE:** Feature audit, correlation analysis, hyperparameter tuning (best RF ROC-AUC: 0.7206 on 50k), class imbalance handling (class weights), temporal/geographic validation splits, comprehensive visualization suite.
- **Realistic Performance:** Post-leakage models achieve ROC-AUC 0.58-0.72 depending on validation strategy.

**Current Focus:**

- Phase 4 successfully completed with all major objectives achieved
- Ready to proceed to Phase 5 (Analysis & Final Documentation)

**Next Steps:**

1. Begin Phase 5: Comprehensive algorithm comparison and analysis
2. Create final visualization suite for paper
3. Write interpretability analysis (feature importance, error patterns)
4. Prepare final documentation and paper
5. Code cleanup and final testing

## Notes and Considerations

### Critical Issues

1. Leakage risk persists beyond perpetrator fields; need full feature audit and alternative splits (temporal/geo).
2. Current splits are random; time-aware evaluation likely needed.
3. Class imbalance handling and threshold tuning may improve minority class performance.

### Technical Debt

- Add logging/versioning/config support; consider Docker for reproducibility.

### Future Enhancements

- Optional: web/API for predictions; real-time capability; geographic visualizations.
