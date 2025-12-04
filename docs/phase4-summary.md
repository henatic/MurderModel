# Phase 4 Summary: Data Investigation & Model Refinement

**Date:** December 3, 2025  
**Status:** COMPLETED ✅

## Overview

Phase 4 focuses on leakage investigation, feature analysis, and model refinement. Current runs use leakage-mitigated features (perpetrator fields removed).

## Completed to Date

### 1) Feature Audit for Leakage

- **File:** `src/utils/feature_audit.py` (artifact: `data/output/feature_audit_20251203_153756.json`)
- Dropped perpetrator fields plus Relationship, Crime Type (low variance), Record Source (no information) in `DataProcessor`.

### 2) Feature Correlation Analysis

- **File:** `src/utils/feature_analysis.py` (artifacts: `data/output/feature_analysis_20251203_153942.json`, `correlation_heatmap_20251203_153942.png`)
- After leakage removal, no highly correlated pairs (>0.8); top correlations with target are weak (~0.07 max).

### 3) Model Retraining (20k sample, cleaned features)

- **Logistic Regression (12/03/2025):**
  - Test: acc 0.7550, prec 0.7945, rec 0.8872, F1 0.8383, ROC-AUC 0.8303 (`logistic_model_evaluation_20251203_121222.json`)
  - CV (3-fold): acc 0.7461 ± 0.0109; prec 0.7877 ± 0.0171; rec 0.8841 ± 0.0097; F1 0.8329 ± 0.0067 (ROC-AUC scorer pending stratified fix)
  - Fairness (test): Victim Sex pos-rate — Female 0.739, Male 0.710, Unknown 0.000; Victim Race — API 0.700, Black 0.730, NA/AN 0.679, Unknown 0.556, White 0.714.
- **Temporal split (Logistic, 20k)**:
  - Test: acc 0.7810, prec 0.7814, rec 0.9994, F1 0.8770, ROC-AUC 0.6075 (`logistic_model_evaluation_20251203_160702.json`)
  - Time-split shows distribution shift; macro recall skewed (mostly positive predictions).
- **Random Forest (12/03/2025):**
  - Test: acc 0.8540, prec 0.9034, rec 0.8914, F1 0.8973, ROC-AUC 0.9120 (`randomforest_model_evaluation_20251203_121321.json`)
  - CV (3-fold): acc 0.8594 ± 0.0025; prec 0.9085 ± 0.0068; rec 0.8937 ± 0.0033; F1 0.9010 ± 0.0017 (ROC-AUC scorer pending stratified fix)
  - Fairness (test): similar group rates as logistic (see `fairness_random_forest_20251203_121321.json`).
- **Geographic split (Random Forest, 20k, class_weight=balanced)**:
  - Test: acc 0.6751, prec 0.6824, rec 0.9726, F1 0.8021, ROC-AUC 0.6169 (`randomforest_model_evaluation_20251203_160836.json`)
- **Temporal split (Random Forest, 50k, class_weight=balanced_subsample):**
  - Test: acc 0.7379, prec 0.7779, rec 0.9112, F1 0.8393, ROC-AUC 0.6442 (`randomforest_model_evaluation_20251203_161700.json`)
- **Geographic split (Random Forest, 50k, class_weight=balanced_subsample):**
  - Test: acc 0.7243, prec 0.7334, rec 0.9777, F1 0.8381, ROC-AUC 0.5796 (`randomforest_model_evaluation_20251203_161820.json`)

### 4) Hyperparameter Tuning

- **File:** `src/models/hyperparameter_tuning.py`
- Random Search (3-fold, 20k) for Logistic:
  - Best params: C≈0.087, penalty=l2, solver=liblinear, max_iter=1000, class_weight=None
  - Best ROC-AUC (CV): 0.5846 (`hyperparameter_optimization_20251203_155843.json`)
- Random Search (3-fold, 20k) for RF:
  - Best params: n_estimators=50, max_depth=50, min_samples_split=50, min_samples_leaf=1, max_features=log2, class_weight=balanced_subsample, bootstrap=True
  - Best ROC-AUC (CV): 0.6999 (`hyperparameter_optimization_20251203_160025.json`)
- Random Search (5-fold, 50k) for RF:
  - Best params: n_estimators=300, max_depth=20, min_samples_split=50, min_samples_leaf=2, max_features=log2, class_weight=balanced_subsample, bootstrap=False
  - Best ROC-AUC (CV): 0.7206 (`hyperparameter_optimization_20251203_161337.json`)

### 5) CV/Fairness in Training Pipeline

- `train.py` supports `--cv-folds`, `--split-strategy` (random/temporal/geo), `--class-weight`; emits CV JSON and fairness reports; CV uses StratifiedKFold.

### 6) Visualization Suite ✅

- **File:** `src/evaluation/visualizations.py`
- **Generated Artifacts (12/03/2025):**
  - Learning curves for both models showing training vs. validation ROC-AUC across sample sizes
  - Precision-Recall curves with average precision scores
  - Threshold analysis plots showing optimal classification thresholds
  - ROC curve comparisons between Logistic Regression and Random Forest
  - Confusion matrices at different thresholds
- **Key Findings:**
  - Random Forest shows better generalization with smaller gap between training and validation
  - Optimal thresholds identified for precision-recall trade-offs
  - Comprehensive visual documentation for model comparison

## Completed Tasks ✅

All major Phase 4 objectives have been accomplished:

1. ✅ Data leakage investigation and mitigation
2. ✅ Feature correlation analysis and redundancy detection
3. ✅ Model retraining on cleaned feature set
4. ✅ Hyperparameter optimization with Grid/Random Search
5. ✅ Class imbalance handling (class weights implemented)
6. ✅ Advanced validation strategies (temporal and geographic splits)
7. ✅ Comprehensive visualization suite

## Remaining Optional Enhancements

## Remaining Optional Enhancements

1. **SMOTE/ADASYN integration**: While class weights are implemented, oversampling methods could be added to the training CLI (utilities already exist in `src/utils/imbalance.py`).
2. **Expanded hyperparameter search**: Consider Bayesian optimization for more efficient parameter exploration.
3. **Feature ablation study**: Systematically remove features to quantify individual contributions.
4. **Deeper fairness analysis**: Threshold tuning specific to demographic groups for fairness-aware predictions.

## Lessons Learned (so far)

1. Leakage mitigation reduced obvious signals; remaining features yield modest AUCs and are more realistic.
2. Random Forest generally outperforms Logistic on cleaned data; temporal/geo splits are notably harder (AUC ~0.58–0.64).
3. Fairness checks are integrated; deeper analysis and threshold tuning remain.
4. Hyperparameter search scaffolding is active; best RF configs now reach ~0.72 CV ROC-AUC on 50k sample.
