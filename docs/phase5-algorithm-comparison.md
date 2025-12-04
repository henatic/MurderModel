# Phase 5 Algorithm Comparison (RF vs Logistic) — Dec 3, 2025 runs

## Scope
- Aligned runs on leakage-mitigated features with tuned params (where available) and class weights.
- Splits: random (20k), temporal (50k), geo (50k).
- Runtimes not captured in logs; commands completed within a few minutes per run on local machine. Re-run with `/usr/bin/time` or Powershell `Measure-Command` if needed.

## Results (test sets)
| Model | Split | Acc | ROC-AUC | Precision | Recall | F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic | Random 20k | 0.716 | 0.576 | 0.716 | 1.000 | 0.834 | CV failed (sklearn treated estimator as regressor; scorer fix needed) |
| Logistic | Temporal 50k | 0.751 | 0.583 | 0.751 | 1.000 | 0.858 |  |
| Logistic | Geo 50k | 0.386 | 0.584 | 0.829 | 0.199 | 0.322 | class_weight=balanced |
| Random Forest | Random 20k | 0.702 | 0.651 | 0.755 | 0.863 | 0.806 | class_weight=balanced_subsample |
| Random Forest | Temporal 50k | 0.738 | 0.644 | 0.778 | 0.911 | 0.839 | class_weight=balanced_subsample |
| Random Forest | Geo 50k | 0.724 | 0.580 | 0.733 | 0.978 | 0.838 | class_weight=balanced_subsample |

## Artifacts
- `data/output/logistic_model_evaluation_20251203_212800.json` (random 20k)
- `data/output/logistic_model_evaluation_20251203_212951.json` (temporal 50k)
- `data/output/logistic_model_evaluation_20251203_213012.json` (geo 50k)
- `data/output/randomforest_model_evaluation_20251203_213038.json` (random 20k)
- `data/output/randomforest_model_evaluation_20251203_213105.json` (temporal 50k)
- `data/output/randomforest_model_evaluation_20251203_213121.json` (geo 50k)

## Takeaways
- RF leads ROC-AUC on random/temporal splits; both models struggle on geo (~0.58).
- Logistic recall = 1.0 on random/temporal, but collapses on geo; RF recall remains very high with lower precision.
- CV/scorer bug: sklearn marks our estimators as regressors during CV/learning_curve; fix `_estimator_type` or scoring path before rerunning CV/learning curves.

## Follow-ups
- Capture runtimes formally (rerun with timing).
- Fix estimator/scorer so CV and learning curves report valid AUC (no NaNs).
- Consider oversampling toggle vs class weights once scorer issue is resolved.
