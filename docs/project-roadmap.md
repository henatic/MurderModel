# Project Roadmap - Murder Model (CSS 581)

## Project Overview

Classification of homicide cases (solved/unsolved) using demographic, geographic, and temporal features (1980-2014).

## Phase 0: Project Initialization (Week 1-2) ✅ COMPLETE

- Repo/env setup, structure, requirements.

## Phase 1: Data Collection & Preprocessing (Week 3-4) ✅ COMPLETE

- Data loaded and cleaned; preprocessing pipeline (imputation, outliers, encoding, scaling); feature engineering (season, age groups); 8 preprocessing tests passing.

## Phase 2: Model Development (Week 5-7) ✅ COMPLETE

- `BaseModel`, stratified splits, persistence, metrics.
- Logistic Regression baseline (`LogisticModel`), training CLI, evaluation.
- Tests for logistic + integration.

## Phase 2.5: Evaluation Framework (Week 7) ✅ COMPLETE

- `ModelEvaluator` with metrics, JSON export, confusion/ROC/feature-importance plots, classification report.

## Phase 2.9: Progress Report (Week 8) ✅ COMPLETE

- ACM-format progress paper delivered.

## Phase 3: Second Algorithm & Comparison (Week 9-10) ✅ COMPLETE

- Random Forest (`RandomForestModel`); comparison script (`compare.py`) on shared splits.
- Multi-model training, model-specific outputs, no double-scaling.
- Tests expanded (21/21 passing including compare).
- Metrics (20k sample, leakage-mitigated features): Logistic test acc 0.755 / AUC 0.830; RF test acc 0.854 / AUC 0.912; 3-fold CV stable; fairness positive-rate signal reported.

## Phase 4: Data Investigation & Model Refinement (Week 10-11) ✅ CORE COMPLETE (optional follow-ups pending)

- Leakage mitigation: perpetrator fields and other high-correlation/low-value fields removed; feature audit and correlations done.
- Hyperparameter tuning: random search runs (Logistic best CV AUC ~0.58; RF best CV AUC ~0.72 on 50k); CV uses StratifiedKFold.
- Class imbalance: class weights wired; SMOTE/ADASYN toggle available in training CLI (`--resample smote|adasyn`) using `imbalanced-learn`.
- Advanced validation: temporal and geographic splits supported in training; observed AUC drop on these splits (temporal ~0.64 RF, geo ~0.58 RF; logistic geo ~0.58 with skewed recall/precision).
- Visualization: confusion/ROC/feature-importance plots; PR curves and threshold analysis emitted; learning curves still optional/pending.
- Fairness: basic group positive-rate reports emitted.
- Remaining optional: benchmark SMOTE/ADASYN vs class weights; add/refresh learning curves; deeper fairness and ablation/error analyses.

## Phase 5: Analysis & Final Documentation (Week 12-13) ✅ COMPLETE

- [x] **Comprehensive algorithm comparison**: Performance metrics, computational efficiency, interpretability analysis documented in `docs/phase5-algorithm-comparison.md`. LR: ROC-AUC 0.579, perfect recall; RF: ROC-AUC 0.684, 31.4% overfitting gap.
- [x] **Visualization suite**: 15+ publication-quality plots generated including performance comparison, overfitting analysis, metric radar, learning curves, PR curves, ROC comparison, threshold analysis. Script: `src/evaluation/phase5_visualizations.py`.
- [x] **Interpretability analysis**: Feature importance comparison, error pattern analysis, fairness considerations documented in `docs/phase5-interpretability.md`. Recommendations for SHAP/LIME deployment.
- [x] **Complete documentation**: Full API reference, usage examples, troubleshooting guide in `docs/API-Documentation.md`. All modules documented with code examples.

## Phase 6: Final Delivery (Week 14-15) 🔄 IN PROGRESS

- [ ] **Final paper** (ACM format, ≤15 pages): Introduction, related work, methodology, results, discussion, conclusion. Template ready in `reports/`.
- [ ] **Presentation materials**: 10-15 minute presentation with slides and speaker notes covering methodology, findings, visualizations.
- [x] **Code cleanup & documentation review**: All 21 tests passing, API documentation complete, codebase well-documented.
- [x] **Final model selection & benchmarking**: Model card created (`docs/MODEL-CARD.md`) with performance metrics, ethical considerations, deployment recommendations. Selected: RF (primary), LR (interpretability).
- [x] **Repository packaging**: Comprehensive README created (`README.md`), all artifacts organized, submission-ready structure.

## Key Milestones

1. ✅ Proposal (Week 1-2)
2. ✅ Initial Data Processing (Week 3-4)
3. ✅ Logistic Regression (Week 5-7)
4. ✅ Progress Report (Week 8)
5. ✅ Random Forest (Week 9-10)
6. ✅ Model Optimization & Comparison (Week 10-11)
7. 🔄 Analysis & Documentation (Week 12-13) — Phase 5 in progress (interpretability deferred)
8. 🔄 Final Paper & Presentation (Week 14) — IN PROGRESS
9. 🔄 Final Project Submission (Week 15) — IN PROGRESS

## Success Metrics (current)

- **Performance**: RF test ROC-AUC 0.684 random, ~0.64 temporal, ~0.58 geo; Logistic test ROC-AUC 0.579 random, ~0.58 geo (post-leakage mitigation).
- **Stability**: Logistic regression shows 0.08% train-test gap; RF shows 31.4% overfitting gap.
- **Documentation**: Phase 0-4 done; Phase 5 docs in progress (`docs/phase5-plan.md`, `docs/phase5-summary.md`, `docs/final-delivery-checklist.md`); interpretability deferred (`docs/phase5-interpretability.md`).
- **Visualizations**: 15+ publication-quality plots (performance comparison, overfitting analysis, metric radar, learning curves, PR curves, ROC, confusion matrices, threshold analysis, error analysis).
- **Tests**: 21/21 passing ✅
- **Phases Complete**: 0, 1, 2, 2.5, 2.9, 3, 4; Phase 5 in progress — Phase 6 pending paper/presentation.
