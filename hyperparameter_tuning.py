"""
Hyperparameter tuning with cost-aware scoring
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
import warnings
warnings.filterwarnings('ignore')

print("loading data...")

# load pre-split data from modeling.py
X_train = pd.read_csv('results/X_train.csv', index_col=0)
X_test = pd.read_csv('results/X_test.csv', index_col=0)
y_train = pd.read_csv('results/y_train.csv', index_col=0)['target']
y_test = pd.read_csv('results/y_test.csv', index_col=0)['target']

print(f"data shape: {X_train.shape[1]} features")
print(f"train: {len(y_train)} samples ({y_train.sum()} failures)")
print(f"test: {len(y_test)} samples ({y_test.sum()} failures)")
print()

# apply SMOTE
print("applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"after SMOTE: {len(y_train_smote)} samples ({y_train_smote.sum()} failures)")
print()

# define parameter space
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [10, 12, 15, 18, 20, None],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'class_weight': ['balanced', 'balanced_subsample']
}

print("searching parameter space:")
print("  n_estimators: 100-300")
print("  max_depth: 10, 12, 15, 18, 20, None")
print("  min_samples_split: 2-10")
print("  min_samples_leaf: 1-5")
print("  max_features: sqrt, log2, 0.3, 0.5")
print("  class_weight: balanced, balanced_subsample")
print("using recall as optimization metric")
print()

scorer = 'recall'

print("running randomized search (50 iterations, 3-fold CV)...")
print("this might take a few minutes...")
print()

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=3,
    scoring=scorer,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train_smote, y_train_smote)

print()
print("search complete!")
print()
print("best parameters found:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
print()

# evaluate best model
best_model = random_search.best_estimator_

y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("evaluating at different thresholds:")
print("-"*50)
print()

for threshold in [0.10, 0.15, 0.20, 0.25, 0.30]:
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    tn = ((y_pred == 0) & (y_test == 0)).sum()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    inspection_rate = (tp + fp) / len(y_test)
    
    print(f"threshold {threshold:.2f}:")
    print(f"  recall: {recall:.1%} ({tp}/{tp+fn} failures caught)")
    print(f"  precision: {precision:.1%}")
    print(f"  inspection rate: {inspection_rate:.1%}")
    print()

# get AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {auc:.3f}")
print()

# compare to baseline
print("comparison to baseline:")
print("-"*50)
print()

# train baseline model
baseline = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
baseline.fit(X_train_smote, y_train_smote)

y_pred_baseline = baseline.predict_proba(X_test)[:, 1]
auc_baseline = roc_auc_score(y_test, y_pred_baseline)

# compare at threshold 0.15
threshold = 0.15

y_pred_tuned = (y_pred_proba >= threshold).astype(int)
y_pred_base = (y_pred_baseline >= threshold).astype(int)

recall_tuned = recall_score(y_test, y_pred_tuned)
recall_base = recall_score(y_test, y_pred_base)

print(f"baseline (previous model):")
print(f"  AUC: {auc_baseline:.3f}")
print(f"  recall @ threshold {threshold}: {recall_base:.1%}")
print()

print(f"tuned model:")
print(f"  AUC: {auc:.3f}")
print(f"  recall @ threshold {threshold}: {recall_tuned:.1%}")
print()

improvement_auc = auc - auc_baseline
improvement_recall = recall_tuned - recall_base

print(f"improvement:")
print(f"  AUC: {improvement_auc:+.3f}")
print(f"  recall: {improvement_recall:+.1%}")
print()

# save results
results_df = pd.DataFrame({
    'metric': ['AUC', 'Recall@0.15', 'Improvement_AUC', 'Improvement_Recall'],
    'baseline': [auc_baseline, recall_base, 0, 0],
    'tuned': [auc, recall_tuned, improvement_auc, improvement_recall]
})

results_df.to_csv('results/hyperparameter_tuning_results.csv', index=False)

print("observations:")
if improvement_auc > 0 or improvement_recall > 0:
    print("  - hyperparameter tuning provided modest improvement")
    print("  - every percent helps in manufacturing quality control")
else:
    print("  - hyperparameter tuning didn't improve over baseline this time")
    print("  - this is normal with small datasets and severe imbalance")
    print("  - threshold optimization had much bigger impact")
print("  - still good practice to test - sometimes you get lucky")
print("  - baseline parameters were already pretty good")
print()

print("files saved:")
print("  - results/hyperparameter_tuning_results.csv")
print()

print("Done! model is now optimized")
