# Final Paper Outline (ACM format) — Phase 6

Use this outline to draft the final paper. Align with ACM template in `reports/` and current metrics (random/temporal/geo splits) with leakage-mitigated features.

1. Introduction  
   - Problem statement (homicide case resolution prediction, 1980–2014)  
   - Motivation and ethical considerations (leakage, fairness)  
   - Contributions (leakage mitigation, multi-split evaluation, comparison RF vs Logistic)

2. Related Work  
   - Crime prediction, cold-case resolution, class imbalance handling  
   - Fairness considerations in criminal justice ML

3. Data & Preprocessing  
   - Dataset description, target distribution, splits (random/temporal/geo)  
   - Leakage mitigation (removed perpetrator fields, low-value features)  
   - Preprocessing pipeline (imputation/encoding/scaling), class weights and resample toggles

4. Methods  
   - Models: Logistic Regression (tuned C/penalty/solver) and Random Forest (tuned hyperparams, class_weight)  
   - Validation strategies: random, temporal, geo; CV via StratifiedKFold  
   - Thresholding and evaluation metrics (ROC-AUC, PR, F1, fairness signals)

5. Experiments  
   - Setup: sample sizes (20k random, 50k temporal/geo), seeds, hardware note  
   - Hyperparameter search summary (best params, CV AUC)  
   - Threshold analysis (best-F1 thresholds per split)

6. Results  
   - Random split: RF vs Logistic metrics (AUC ~0.65 vs 0.58 after leakage mitigation)  
   - Temporal split: AUC ~0.64 RF vs ~0.58 Logistic  
   - Geo split: AUC ~0.58 both; recall/precision trade-offs  
   - Learning curves and error analysis (misclassification by Victim Sex/Race)

7. Discussion  
   - Impact of leakage mitigation (drop from ~0.91 to ~0.58–0.65 AUC)  
   - Generalization challenges on geo/temporal splits  
   - Fairness considerations (group positive rates; no group-specific thresholds applied)  
   - Interpretability status (SHAP/LIME deferred) and implications

8. Conclusion & Future Work  
   - Summary of findings and recommended deployment stance (RF primary, LR for explanations)  
   - Future: richer features, SHAP/LIME, fairness-aware thresholds, additional models (GBM)

Appendices  
   - Command reproduction steps  
   - Full metrics tables (JSON refs)  
   - Plots: ROC/PR, learning curves, error-by-group
