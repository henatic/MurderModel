# Presentation Outline (10–15 min) — Phase 6

Slide flow:
1) Title & team, problem statement, data scope (1980–2014, solved/unsolved).  
2) Ethics & leakage: why perpetrator fields removed; fairness considerations.  
3) Data & preprocessing: pipeline, class balance, split strategies (random/temporal/geo), resample/class_weight options.  
4) Models & tuning: Logistic (tuned C/penalty/solver), Random Forest (tuned params, class_weight).  
5) Validation: random vs temporal vs geo; CV setup (StratifiedKFold).  
6) Results: key metrics by split (RF vs Logistic), threshold highlights, learning curves.  
7) Error analysis: misclassification by Victim Sex/Race; fairness signals (positive rates).  
8) Interpretability status: SHAP/LIME deferred; plan if time permits.  
9) Takeaways: leakage impact, generalization challenges, deployment stance (RF primary, LR for explanations).  
10) Next steps & risks: fairness-aware thresholds, richer features, model extensions, documentation/delivery checklist.
