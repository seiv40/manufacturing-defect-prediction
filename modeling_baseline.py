"""
Initial modeling with threshold optimization
Building baseline models and tuning decision thresholds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, recall_score, 
                             precision_score, f1_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
import os
warnings.filterwarnings('ignore')

print("loading processed data...")

# load the cleaned dataset from feature selection
data = pd.read_csv('results/processed_data.csv')

# separate features and target
X = data.drop('target', axis=1)
y = data['target']

# convert target from -1/1 to 0/1 (needed for some models)
# -1 (pass) -> 0, 1 (fail) -> 1
y_binary = (y == 1).astype(int)

print(f"data shape: {X.shape}")
print(f"target distribution: {y.value_counts().to_dict()}")

# ----- train/test split -----

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, 
    test_size=0.25,
    stratify=y_binary,
    random_state=42
)

print(f"\ntrain size: {len(X_train)}, test size: {len(X_test)}")

# ----- feature scaling -----

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# ----- helper function -----

def evaluate_model(name, y_true, y_pred, y_proba=None):
    """evaluate and print metrics"""
    print(f"\n{name}:")
    print(f"  Recall: {recall_score(y_true, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  F1: {f1_score(y_true, y_pred):.3f}")
    
    if y_proba is not None:
        print(f"  ROC-AUC: {roc_auc_score(y_true, y_proba):.3f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    return {
        'model': name,
        'recall': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba) if y_proba is not None else None
    }

results = []

print("Building baseline models...")

# ----- logistic regression -----

print("\nLogistic Regression (baseline)")

lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

result = evaluate_model('Logistic Regression', y_test, y_pred_lr, y_proba_lr)
results.append(result)

# ----- random forest with SMOTE -----

print("\nRandom Forest (SMOTE)")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

rf_smote = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                   random_state=42, n_jobs=-1)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_rf_smote = rf_smote.predict(X_test)
y_proba_rf_smote = rf_smote.predict_proba(X_test)[:, 1]

result = evaluate_model('Random Forest (SMOTE)', y_test, y_pred_rf_smote, y_proba_rf_smote)
results.append(result)

# ----- random forest with class weights (for threshold tuning) -----

print("\nRandom Forest (weights) - default threshold")

n_negative = (y_train == 0).sum()
n_positive = (y_train == 1).sum()

rf_weights = RandomForestClassifier(n_estimators=100, max_depth=10,
                                     class_weight='balanced',
                                     random_state=42, n_jobs=-1)
rf_weights.fit(X_train, y_train)
y_pred_rf_weights = rf_weights.predict(X_test)
y_proba_rf_weights = rf_weights.predict_proba(X_test)[:, 1]

result = evaluate_model('RF (weights) default', y_test, y_pred_rf_weights, y_proba_rf_weights)
results.append(result)

# ----- threshold tuning on random forest -----

print("Optimizing decision threshold...")

# try different thresholds
thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

print("\nTesting thresholds on Random Forest (weights):")

threshold_results = []
for threshold in thresholds:
    y_pred_threshold = (y_proba_rf_weights >= threshold).astype(int)
    recall = recall_score(y_test, y_pred_threshold)
    precision = precision_score(y_test, y_pred_threshold, zero_division=0)
    f1 = f1_score(y_test, y_pred_threshold)
    
    # count predictions
    n_predicted_failures = y_pred_threshold.sum()
    pct_inspected = n_predicted_failures / len(y_test) * 100
    
    print(f"  Threshold {threshold:.2f}: Recall={recall:.3f}, Precision={precision:.3f}, "
          f"Inspect {pct_inspected:.1f}%")
    
    threshold_results.append({
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'pct_inspected': pct_inspected
    })

# find best threshold by F1 score
threshold_df = pd.DataFrame(threshold_results)
best_threshold_idx = threshold_df['f1'].idxmax()
best_threshold = threshold_df.iloc[best_threshold_idx]['threshold']

print(f"\nBest threshold by F1: {best_threshold}")

# apply best threshold
y_pred_rf_tuned = (y_proba_rf_weights >= best_threshold).astype(int)

result = evaluate_model(f'RF (weights) threshold={best_threshold}', 
                        y_test, y_pred_rf_tuned, y_proba_rf_weights)
results.append(result)

# also try threshold optimized for recall (more aggressive)
# choose threshold that gets ~70-80% recall
recall_target = 0.75
best_recall_threshold = None

for threshold in np.arange(0.05, 0.5, 0.01):
    y_pred_temp = (y_proba_rf_weights >= threshold).astype(int)
    recall = recall_score(y_test, y_pred_temp)
    if recall >= recall_target:
        best_recall_threshold = threshold
        break

if best_recall_threshold:
    y_pred_rf_recall = (y_proba_rf_weights >= best_recall_threshold).astype(int)
    result = evaluate_model(f'RF (weights) recall-optimized (t={best_recall_threshold:.2f})',
                           y_test, y_pred_rf_recall, y_proba_rf_weights)
    results.append(result)

# ----- xgboost with SMOTE -----

print("\nXGBoost (SMOTE)")

xgb_smote = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=42, eval_metric='logloss')
xgb_smote.fit(X_train_smote, y_train_smote)
y_pred_xgb = xgb_smote.predict(X_test)
y_proba_xgb = xgb_smote.predict_proba(X_test)[:, 1]

result = evaluate_model('XGBoost (SMOTE)', y_test, y_pred_xgb, y_proba_xgb)
results.append(result)

# ----- comparison -----

print("Model Comparison (sorted by recall)")

comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('recall', ascending=False)

print("\n" + comparison_df.to_string(index=False))

# save
os.makedirs('results', exist_ok=True)
comparison_df.to_csv('results/model_comparison.csv', index=False)

# save threshold tuning results
threshold_df.to_csv('results/threshold_tuning.csv', index=False)

# ----- visualizations -----

print("\ncreating visualizations...")

# threshold tuning plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(threshold_df['threshold'], threshold_df['recall'], 
        marker='o', label='Recall', linewidth=2, markersize=6)
ax.plot(threshold_df['threshold'], threshold_df['precision'], 
        marker='s', label='Precision', linewidth=2, markersize=6)
ax.plot(threshold_df['threshold'], threshold_df['f1'], 
        marker='^', label='F1-Score', linewidth=2, markersize=6)
ax.axvline(best_threshold, color='red', linestyle='--', alpha=0.5, 
           label=f'Best F1 threshold: {best_threshold}')
ax.set_xlabel('Decision Threshold', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Threshold Tuning: Recall vs Precision Trade-off', 
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(threshold_df['threshold'], threshold_df['pct_inspected'],
        marker='o', color='steelblue', linewidth=2, markersize=6)
ax.axvline(best_threshold, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Decision Threshold', fontsize=11)
ax.set_ylabel('% of Production Inspected', fontsize=11)
ax.set_title('Inspection Rate by Threshold', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/threshold_tuning.png', dpi=300)
print("saved threshold tuning plot")
plt.close()

# feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_weights.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\ntop 10 important features:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

importance_df.to_csv('results/feature_importance.csv', index=False)

# ----- summary -----

print("Final Results")

best_model = comparison_df.iloc[0]
print(f"\nBest model: {best_model['model']}")
print(f"  Recall: {best_model['recall']:.1%} (catches {best_model['recall']*26:.0f}/26 failures)")
print(f"  Precision: {best_model['precision']:.1%}")
print(f"  F1-Score: {best_model['f1']:.3f}")
print(f"  ROC-AUC: {best_model['roc_auc']:.3f}")

print("\nFiles saved:")
print("  - results/model_comparison.csv")
print("  - results/threshold_tuning.csv")
print("  - results/feature_importance.csv")
print("  - results/figures/threshold_tuning.png")

print("\ndone!")
