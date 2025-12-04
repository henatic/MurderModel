# Phase 5 Action Checklist (Analysis & Final Documentation)

Use this as the single place to track Phase 5 execution. Update the checkboxes as you complete tasks and drop links to artifacts (JSON/PNGs/notes) when available.

## 1) Algorithm Comparison (RF vs Logistic)
- [x] Random split: run both models with tuned params on the same split; capture metrics + runtime. (artifacts: `*evaluation_20251203_212800.json`, `*evaluation_20251203_213038.json`; CV skipped for Logistic due to scorer/type mismatch)
  - Command example: `python -m src.models.train --model random_forest --data data/raw/data.csv --split-strategy random --nrows 20000 --class-weight balanced_subsample --cv-folds 3`
  - Command example: `python -m src.models.train --model logistic --data data/raw/data.csv --split-strategy random --nrows 20000 --logreg-C 0.087 --logreg-penalty l2 --logreg-solver liblinear --class-weight balanced --cv-folds 3`
- [x] Temporal split: repeat both models; log deltas vs. random split. (artifacts: `*evaluation_20251203_212951.json`, `*evaluation_20251203_213105.json`)
- [x] Geographic split: repeat both models; log deltas vs. random split. (artifacts: `*evaluation_20251203_213012.json`, `*evaluation_20251203_213121.json`)
- [x] Summarize in a short RF vs Logistic comparison note (include metrics + runtime note). See `docs/phase5-algorithm-comparison.md` (runtimes not captured; rerun with timing if needed).

## 2) Validation Curves & Error Analysis
- [x] Learning curves generated (10k sample) after fixing estimator tags: see `data/output/learning_curve_logistic_(balanced)_20251203_214235.png`, `learning_curve_random_forest_(balanced_subsample)_20251203_214239.png`.
- [x] Error analysis visuals (misclassification by Victim Sex/Race, 20k random split): `data/output/error_by_victim_sex_logistic_balanced_20251203.png`, `data/output/error_by_victim_race_logistic_balanced_20251203.png`, `data/output/error_by_victim_sex_rf_balanced_subsample_20251203.png`, `data/output/error_by_victim_race_rf_balanced_subsample_20251203.png`.
- [x] Catalog plot paths in docs (Phase 5 summary + README section).

## 3) Interpretability
- [!] SHAP/LIME deferred for now; rationale and resume steps noted in `docs/phase5-interpretability.md`. Run later if time allows.

## 4) Fairness & Thresholds
- [!] Thresholds logged (global best-F1 per split in `docs/phase5-summary.md`); group-specific thresholds not applied.

## 5) Documentation & Packaging
- [x] Phase 5 summary doc updated with latest runs, learning curves, error analysis.
- [x] README expanded with Phase 5 flags/troubleshooting links.
- [ ] Prepare final delivery checklist (paper/presentation/package) per development-phases.md.

## Artifact Dropzones
- Metrics/JSON: `data/output/*evaluation*.json`, `data/output/cv_*.json`
- Plots: `data/output/*png`, `data/output/*pdf`
- Notes/Summaries: `docs/phase5-summary.md` (to be created), `docs/phase5-plan.md` (this file)
