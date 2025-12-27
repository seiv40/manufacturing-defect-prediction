# SECOM Manufacturing Defect Prediction

Predictive analysis of sensor data from semiconductor manufacturing to identify factors contributing to production failures.

## Dataset

SECOM Dataset (UCI Machine Learning Repository)
- 1,567 production runs
- 590 sensor measurements per run
- Binary outcome: Pass (1,463) / Fail (104)
- Severe class imbalance: 14:1 ratio

## Objective

Identify which sensor measurements are most predictive of manufacturing failures to enable early detection and cost-optimized quality control.

## Methods

### 1. Exploratory Analysis
Script: `exploratory_analysis.py`

- Analyzed class distribution and imbalance patterns
- Identified missing data: 4.5% overall, 28 sensors >50% missing
- Discovered 116 constant sensors (zero variance)
- Evaluated feature correlations with failures (max r=0.156)

### 2. Feature Engineering & Selection
Script: `feature_selection.py`

- Removed 116 constant sensors and 28 high-missing sensors
- Eliminated 175 redundant sensors (>0.95 correlation)
- Created 52 missing data indicator features
- Applied composite ranking: correlation + mutual information + Random Forest importance
- Result: Reduced 590 → 127 features (78% reduction)

### 3. Modeling & Threshold Optimization
Scripts: `modeling.py`, `modeling_production.py`

- Tested algorithms: Logistic Regression, Random Forest, XGBoost
- Handled class imbalance: SMOTE oversampling + class weights
- Optimized decision thresholds for recall vs. inspection rate tradeoff
- Evaluated multiple cost-based operating points

### 4. Hyperparameter Tuning
Script: `hyperparameter_tuning.py`

- Randomized search over 50 parameter combinations
- 5-fold stratified cross-validation
- Optimized for recall (priority: catching failures)

### 5. Interpretability Analysis
Scripts: `domain_interpretation.py`, `shap_analysis.py`

- Clustered sensors into 8 process-stage groups using hierarchical clustering
- Generated SHAP explanations for individual predictions
- Identified stable early-warning sensors vs. late-detection signals

## Results

### Model Performance (Test Set)

| Model | ROC-AUC | Recall @ t=0.15 | Inspection Rate |
|-------|---------|-----------------|-----------------|
| Random Forest (optimized) | 0.84 | 73% | 18% |
| XGBoost + SMOTE | 0.82 | 69% | 22% |
| Logistic Regression | 0.76 | 46% | 12% |

### Cost-Based Operating Points

Different thresholds for different business priorities:

| Strategy | Threshold | Recall | Inspection Rate | Use Case |
|----------|-----------|--------|-----------------|----------|
| Conservative | 0.25 | 46% | 11% | Low inspection capacity |
| Balanced | 0.15 | 73% | 18% | Standard operations |
| Aggressive | 0.05 | 96% | 64% | High-value production |

Main Tradeoff: Achieving 73% recall requires inspecting 18% of production (19 of 26 failures caught). For 96% recall (25 of 26 failures), inspection rate increases to 64%, demonstrating the precision-recall tradeoff fundamental to cost-based decision making in manufacturing.

### Feature Insights

Top Predictive Sensors:
1. sensor_59 (importance: 3.8%)
2. sensor_33 (importance: 2.9%)
3. sensor_130 (importance: 2.7%)
4. sensor_510 (importance: 2.4%)

Process:
- Identified 8 sensor clusters representing distinct process stages
- Missing data indicators turned out predictive (sensor health correlates with process quality)
- 15 stable early-warning sensors identified via cross-validation consistency

### Main Findings

1. Threshold optimization more impactful than oversampling
   - SMOTE: 46% → 69% recall
   - Threshold tuning: 69% → 96% recall

2. Missing data is informative
   - Created 52 missing indicators as features
   - Sensor failures may signal upstream process issues

3. Interpretability enables deployment
   - SHAP values explain individual predictions
   - Domain clustering maps sensors to process stages
   - Actionable insights for process engineers

## Tech Stack

- Python 3.14
- Data Processing: pandas, numpy
- Machine Learning: scikit-learn, xgboost, imbalanced-learn
- Interpretability: SHAP
- Visualization: matplotlib, seaborn

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn scipy

# Run analysis pipeline (in order)
python exploratory_analysis.py
python feature_selection.py
python modeling_optimized.py
python hyperparameter_tuning.py
python domain_interpretation.py
python shap_analysis.py

# Optional: Generate pipeline diagram
python create_pipeline_diagram.py
```