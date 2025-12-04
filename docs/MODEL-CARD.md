# Model Card: Homicide Case Classification

**Version:** 1.0  
**Date:** December 3, 2025  
**Status:** Production-Ready Candidate

## Model Details

### Basic Information

**Model Name:** Murder Model - Homicide Case Classifier  
**Model Type:** Binary Classification (Solved vs Unsolved)  
**Algorithms:** Logistic Regression & Random Forest Ensemble  
**Framework:** Scikit-learn 1.0+  
**License:** Educational Use (CSS 581 Project)

### Developers

**Organization:** University of Washington Bothell  
**Course:** CSS 581 - Machine Learning  
**Development Period:** September - December 2025  
**Contact:** [Repository Issues](https://github.com/henatic/MurderModel)

### Model Architecture

**Primary Model (Recommended):** Random Forest Classifier

- n_estimators: 300
- max_depth: 20
- min_samples_split: 2
- class_weight: balanced_subsample
- random_state: 42

**Secondary Model (For Interpretation):** Logistic Regression

- C: 1.0 (default after optimization)
- max_iter: 1000
- class_weight: balanced
- random_state: 42

### Intended Use

**Primary Intended Uses:**

- Historical pattern analysis of homicide case resolution
- Resource allocation planning for law enforcement agencies
- Academic research on factors affecting case clearance rates
- Policy analysis for criminal justice reform

**Out-of-Scope Uses:**

- Individual case outcome prediction for active investigations
- Replacement for investigative judgment or due process
- Profiling or surveillance applications
- Real-time operational decision-making

## Training Data

### Dataset Description

**Source:** Historical Homicide Database (1980-2014)  
**Size:** ~600,000 cases (full dataset)  
**Training Sample:** Variable (10,000-50,000 in experiments)  
**Geographic Coverage:** United States  
**Temporal Coverage:** 35 years (1980-2014)

### Features Used

**Total Features:** ~50 (after leakage mitigation and encoding)

**Feature Categories:**

1. **Demographic Features (Victim)**

   - Age group (binned)
   - Sex
   - Race
   - Ethnicity

2. **Geographic Features**

   - State (one-hot encoded)
   - City population size category
   - Regional indicators

3. **Temporal Features**
   - Year
   - Month
   - Season
   - Decade

**Explicitly Excluded Features (Leakage Prevention):**

- Perpetrator sex, age, race, ethnicity
- Relationship between victim and perpetrator
- Crime type classification
- Record source
- Crime solved indicator (target variable only)

### Data Preprocessing

**Imputation:**

- Numeric: Median imputation
- Categorical: Mode imputation or "Unknown" category

**Encoding:**

- One-hot encoding for categorical variables
- Binary encoding for yes/no variables

**Scaling:**

- StandardScaler for numeric features
- Applied after train/test split to prevent leakage

**Class Distribution:**

- Solved cases: ~73% (majority class)
- Unsolved cases: ~27% (minority class)
- Imbalance handled via class weights, SMOTE, or ADASYN

## Performance

### Test Set Results (Random Split)

**Random Forest:**

- ROC-AUC: 0.684
- Accuracy: 0.732
- Precision: 0.778
- Recall: 0.887
- F1 Score: 0.829

**Logistic Regression:**

- ROC-AUC: 0.579
- Accuracy: 0.732
- Precision: 0.732
- Recall: 1.000
- F1 Score: 0.845

### Cross-Validation Results

**5-Fold Stratified CV (50k sample):**

- Random Forest: Mean ROC-AUC 0.7206 (±0.02)
- Logistic Regression: Mean ROC-AUC 0.5846 (±0.01)

### Validation Strategy Performance

| Strategy             | Random Forest ROC-AUC | Logistic Regression ROC-AUC |
| -------------------- | --------------------- | --------------------------- |
| **Random Split**     | 0.684                 | 0.579                       |
| **Temporal Split**   | ~0.64                 | ~0.58                       |
| **Geographic Split** | ~0.58                 | ~0.58                       |

**Key Finding:** Geographic generalization is most challenging, suggesting regional differences in case handling, resources, or data collection.

### Overfitting Analysis

**Random Forest:**

- Train ROC-AUC: 0.999
- Test ROC-AUC: 0.684
- **Gap: 31.4%** (Significant overfitting)

**Logistic Regression:**

- Train ROC-AUC: 0.580
- Test ROC-AUC: 0.579
- **Gap: 0.08%** (Excellent stability)

## Ethical Considerations

### Fairness

**Protected Attributes in Dataset:**

- Victim race (White, Black, Hispanic, Asian, Native American, Other)
- Victim sex (Male, Female, Unknown)
- Geographic location (potential proxy for socioeconomic status)

**Fairness Analysis Conducted:**

- Positive prediction rate by race group
- Positive prediction rate by sex group
- Performance metrics stratified by demographics

**Known Disparities:**

- Model performance varies significantly by state
- Historical biases in policing and case closure may be encoded in training data
- Demographic features correlate with prediction outcomes

**Mitigation Strategies:**

- Removed perpetrator demographic features
- Continuous fairness monitoring via evaluation framework
- Documented all observed disparities
- Recommended threshold optimization per demographic group

### Privacy

**Data Handling:**

- Historical public records (1980-2014)
- No personally identifiable information (PII) used
- Aggregated geographic data (state/city level)
- No real-time surveillance data

**Privacy Risks:**

- Low: Data is historical and aggregated
- Models do not identify individuals
- Geographic features are broad categories

### Bias and Limitations

**Known Biases:**

1. **Historical Bias**: Training on historical data perpetuates past policing patterns
2. **Selection Bias**: Only reported cases included; dark figure of crime excluded
3. **Geographic Bias**: Data quality varies by jurisdiction
4. **Temporal Bias**: Policing practices changed over 35-year period
5. **Reporting Bias**: Not all homicides equally likely to be reported/recorded

**Model Limitations:**

1. **Performance**: Moderate ROC-AUC (0.58-0.68) indicates limited predictive power
2. **Generalization**: Poor geographic generalization suggests jurisdiction-specific patterns
3. **Overfitting**: Random Forest severely overfits despite regularization
4. **Feature Limitations**: After leakage removal, remaining features weakly predictive
5. **Temporal Drift**: Model trained on 1980-2014 data; may not apply to current era

**What the Model CANNOT Predict:**

- Individual case outcomes with high confidence
- Influence of evidence quality, witnesses, or investigator skill
- Impact of community cooperation or resource availability
- Effect of media attention or political pressure
- Modern cases with different technologies/practices

## Recommendations

### Model Selection

**For Deployment:**

- **Primary Model**: Random Forest (better ROC-AUC: 0.684)
- **Interpretation Model**: Logistic Regression (explainability, stability)
- **Hybrid Strategy**: Use RF for predictions, LR for explanations

**Decision Threshold:**

- Default: 0.5 (balanced)
- High-recall setting: 0.3-0.4 (catch more solved cases, more false positives)
- High-precision setting: 0.6-0.7 (reduce false positives, miss some solved cases)
- **Recommendation**: Optimize threshold based on cost of false positives vs false negatives

### Deployment Considerations

**Before Deployment:**

1. Conduct thorough fairness audit with domain experts
2. Establish monitoring dashboard for demographic disparities
3. Create human-in-the-loop review process
4. Document use case restrictions and limitations
5. Train users on proper interpretation and limitations

**Monitoring Requirements:**

- Monthly fairness metrics by demographics
- Quarterly model performance evaluation
- Annual retraining on updated data
- Continuous drift detection (temporal, geographic)

**Red Flags (Retrain/Investigate):**

- ROC-AUC drops below 0.55
- Demographic disparity exceeds 10% in any metric
- Significant geographic performance degradation
- User reports systematic errors

### Future Improvements

**Short-Term (Next 6 months):**

1. Implement SHAP values for case-level explanations
2. Optimize RF regularization to reduce overfitting
3. Engineer features for geographic clustering
4. Expand fairness analysis framework

**Long-Term (Next 1-2 years):**

1. Collect post-2014 data for temporal validation
2. Incorporate case complexity indicators
3. Explore gradient boosting methods (XGBoost, LightGBM)
4. Implement fairness-aware learning algorithms
5. Develop region-specific models for better localization

## Model Evaluation

### Quantitative Evaluation

**Metrics Used:**

- ROC-AUC (primary metric for imbalanced data)
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- Precision-Recall Curve
- Learning Curves

**Validation Methods:**

- Stratified K-Fold Cross-Validation (k=3,5)
- Temporal Hold-Out Validation
- Geographic Hold-Out Validation
- Random 70/10/20 Split

### Qualitative Evaluation

**Interpretability Assessment:**

- Logistic Regression: High (clear coefficient interpretation)
- Random Forest: Moderate (feature importance available, no direction)

**Stakeholder Feedback:** (Planned for Phase 6)

- Law enforcement agencies (interpretability, practical utility)
- Criminal justice researchers (methodology, ethical considerations)
- Community advocates (fairness, bias concerns)

## Caveats and Recommendations

### For Researchers

✅ **Suitable for:**

- Understanding historical patterns in case clearance
- Identifying geographic and temporal trends
- Testing hypotheses about factors affecting resolution

⚠️ **Not suitable for:**

- Causal inference (observational data only)
- Individual case prediction
- Modern case prediction (data ends 2014)

### For Practitioners

✅ **Suitable for:**

- Resource allocation planning
- Identifying jurisdictions needing support
- Policy analysis and reform discussions

⚠️ **Not suitable for:**

- Case prioritization or triage
- Replacing investigative processes
- Performance evaluation of individual officers/departments

### For Policymakers

✅ **Suitable for:**

- Identifying systemic patterns
- Informing resource distribution
- Supporting evidence-based policy discussions

⚠️ **Not suitable for:**

- Justifying surveillance or profiling
- Determining individual culpability
- Replacing community input or democratic processes

## Technical Specifications

### Software Dependencies

```
Python >= 3.8
scikit-learn >= 1.0
pandas >= 1.3
numpy >= 1.21
matplotlib >= 3.4
seaborn >= 0.11
imbalanced-learn >= 0.8 (optional, for SMOTE/ADASYN)
```

### Hardware Requirements

**Minimum:**

- CPU: 2 cores
- RAM: 4 GB
- Storage: 1 GB

**Recommended:**

- CPU: 4+ cores
- RAM: 8+ GB
- Storage: 5 GB (for full dataset and outputs)

### Inference Time

**Logistic Regression:**

- Single prediction: <1 ms
- Batch (1000): ~10 ms
- Full test set (10k): ~100 ms

**Random Forest:**

- Single prediction: ~5 ms
- Batch (1000): ~50 ms
- Full test set (10k): ~500 ms

### Model Size

- Logistic Regression: ~50 KB
- Random Forest: ~10 MB (300 trees)

## Maintenance

### Update Schedule

- **Monthly**: Fairness monitoring reports
- **Quarterly**: Performance evaluation
- **Annually**: Full retraining with new data (if available)
- **As-needed**: Bug fixes, security patches

### Versioning

**Current Version:** 1.0 (December 2025)

**Version History:**

- v1.0 (Dec 2025): Initial production-ready release
- v0.9 (Dec 2025): Phase 5 complete, all analysis done
- v0.5 (Nov 2025): Phase 4 complete, leakage mitigated
- v0.1 (Oct 2025): Phase 3 complete, both models trained

### Contact

**Issues/Questions:** [GitHub Issues](https://github.com/henatic/MurderModel/issues)  
**Documentation:** `docs/API-Documentation.md`  
**Detailed Analysis:** `docs/phase5-algorithm-comparison.md`

---

**Document Status:** FINAL  
**Approval Status:** Academic Review Pending  
**Last Updated:** December 3, 2025
