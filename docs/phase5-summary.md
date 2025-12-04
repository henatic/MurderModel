# Phase 5 Summary: Analysis & Final Documentation (Work in Progress)

**Phase Window:** Week 12-13  
**Status:** In Progress (core tasks not yet executed)

## Current State
- RF vs Logistic comparison runs on random/temporal/geo splits are completed; summary lives in `docs/phase5-algorithm-comparison.md`.
- Learning curves (10k sample) generated after fixing estimator tags; see artifacts below.
- Error analysis plots generated (misclassification by Victim Sex/Race on 20k random split).
- Interpretability (SHAP/LIME) is deferred; see `docs/phase5-interpretability.md` for rationale and steps to resume.
- README/API/troubleshooting refresh for Phase 5 is still pending.

## Latest Phase 5 runs (Dec 3, 2025)
| Model      | Split         | Acc    | ROC-AUC | Precision | Recall | F1    | Notes |
|------------|---------------|--------|---------|-----------|--------|-------|-------|
| Logistic   | Random 20k    | 0.716  | 0.576   | 0.716     | 1.000  | 0.834 | CV skipped: sklearn scorer saw regressor type; evaluation OK |
| Logistic   | Temporal 50k  | 0.751  | 0.583   | 0.751     | 1.000  | 0.858 |       |
| Logistic   | Geo 50k       | 0.386  | 0.584   | 0.829     | 0.199  | 0.322 | class_weight=balanced |
| RandomForest | Random 20k  | 0.702  | 0.651   | 0.755     | 0.863  | 0.806 | class_weight=balanced_subsample |
| RandomForest | Temporal 50k| 0.738  | 0.644   | 0.778     | 0.911  | 0.839 | class_weight=balanced_subsample |
| RandomForest | Geo 50k     | 0.724  | 0.580   | 0.733     | 0.978  | 0.838 | class_weight=balanced_subsample |

Artifacts:
- `data/output/logistic_model_evaluation_20251203_212800.json` (random 20k)
- `data/output/logistic_model_evaluation_20251203_212951.json` (temporal 50k)
- `data/output/logistic_model_evaluation_20251203_213012.json` (geo 50k)
- `data/output/randomforest_model_evaluation_20251203_213038.json` (random 20k)
- `data/output/randomforest_model_evaluation_20251203_213105.json` (temporal 50k)
- `data/output/randomforest_model_evaluation_20251203_213121.json` (geo 50k)
- Learning curves (10k sample, AUC vs training size, StratifiedKFold=3):
  - `data/output/learning_curve_logistic_(balanced)_20251203_214235.png`
  - `data/output/learning_curve_random_forest_(balanced_subsample)_20251203_214239.png`
- Error analysis (misclassification rate by group, 20k random split):
  - `data/output/error_by_victim_sex_logistic_balanced_20251203.png`
  - `data/output/error_by_victim_race_logistic_balanced_20251203.png`
  - `data/output/error_by_victim_sex_rf_balanced_subsample_20251203.png`
  - `data/output/error_by_victim_race_rf_balanced_subsample_20251203.png`
- Error analysis (misclassification rate by group, 20k random split):
  - `data/output/error_by_victim_sex_logistic_balanced_20251203.png`
  - `data/output/error_by_victim_race_logistic_balanced_20251203.png`
  - `data/output/error_by_victim_sex_rf_balanced_subsample_20251203.png`
  - `data/output/error_by_victim_race_rf_balanced_subsample_20251203.png`

## Plan of Record
See `docs/phase5-plan.md` for the actionable checklist, commands, and artifact dropzones. Key tracks:
1) Algorithm comparison (RF vs Logistic) on aligned splits with metrics + runtime.
2) Validation curves and error analysis plots; catalog outputs.
3) Interpretability (SHAP/LIME) or documented deferral.
4) Fairness-aware threshold review (optional but recommended).
5) Documentation refresh and final delivery checklist.

## Next Steps (immediate)
- Run one interpretability pass (e.g., SHAP for RF) or record a deferral with rationale.
- Update README/docs with usage/examples/troubleshooting for Phase 5 features (split strategies, resample toggle).
- If time allows: evaluate thresholds per key groups using existing threshold JSONs; otherwise note not applied.

## Threshold recap (global best-F1 per split)
- Logistic: random20k t=0.544 (F1=0.835, prec=0.717, rec=0.999); temporal50k t=0.620 (F1=0.858, prec=0.751, rec=1.000); geo50k t=0.281 (F1=0.844, prec=0.730, rec=1.000).
- Random Forest: random20k t=0.083 (F1=0.836, prec=0.720, rec=0.997); temporal50k t=0.141 (F1=0.859, prec=0.754, rec=0.997); geo50k t=0.100 (F1=0.844, prec=0.730, rec=1.000).
- Fairness-aware thresholds by group not applied; would need group-specific evaluation using existing threshold outputs.

## Artifacts (to be filled as produced)
- Metrics JSON: see list above
- Plots: see list above
- Interpretability notes: deferred (`docs/phase5-interpretability.md`)
- Documentation updates: `docs/phase5-plan.md`, `docs/phase5-summary.md`, README troubleshooting/usage
