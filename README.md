# SECOM Manufacturing Defect Prediction

Can you predict semiconductor manufacturing failures from sensor data? Turns out yes, but the interesting part isn't the prediction - it's figuring out the right tradeoff between how many failures you catch vs. how much you're willing to inspect.

## Data

Using the SECOM dataset from UCI - 1,567 production runs with 590 sensor measurements each. Only 104 are failures, so you're dealing with severe class imbalance (14:1 ratio).

## What I Built

The data needed a lot of cleaning. Started with 590 sensors but 116 were constant (completely useless) and 28 were missing over half their values. After removing those and some highly correlated duplicates, ended up with 446 sensors.

Used mutual information on the training set to select the top 75 features. Also created missing data indicators for sensors that frequently dropped out - these actually turned out to be some of the most predictive features. Apparently when sensors fail to report, that's often a sign something's wrong with the process.

Tried Random Forest, XGBoost, and logistic regression. Random Forest with SMOTE worked best. Applied threshold optimization to balance recall vs. inspection cost.

Also used SHAP for interpretability so you can see exactly why the model flags specific production runs.

## Results

Test set: **AUC 0.77** (cross-validation: 0.62 ± 0.05)

Depending on how aggressive you want to be:
- Threshold 0.05: catches all 26 failures but you inspect 89% of production
- Threshold 0.15: catches 21/26 failures, inspect 45% (balanced)
- Threshold 0.25: catches 17/26 failures, inspect 24% (conservative)

So if missing a defect is really expensive, you can catch everything - but it costs you. If inspection is the bottleneck, you can dial it back and still catch most failures.

Top predictive features were sensor_247, sensor_519, and their missing indicators. Organized sensors into 8 clusters that probably represent different process stages.

## Main Findings

Missing data is informative. Four of the top 10 features were just "did this sensor fail to report" flags. Sensor failures correlate with process issues.

Threshold tuning matters way more than hyperparameter optimization. An extra 0.01 AUC doesn't help much, but getting the cost tradeoff right is critical for deployment.

## Running It

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn scipy

python exploratory_analysis.py
python feature_selection.py
python modeling.py
python hyperparameter_tuning.py
python domain_interpretation.py
python shap_analysis.py
```

The modeling.py script does the train/test split and feature selection. Other scripts load the pre-split data so everything uses the same partition.
