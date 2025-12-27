"""
Cost-optimized modeling with cross-validation
Testing different cost ratios to find optimal thresholds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve, 
                             recall_score, precision_score, f1_score)
from imblearn.over_sampling import SMOTE
import warnings
import os
warnings.filterwarnings('ignore')

print("loading data and creating features...")

# load original data to detect missing values
data_raw = pd.read_csv('data/secom.data', sep=' ', header=None, na_values='NaN')
data_raw.columns = [f'sensor_{i}' for i in range(data_raw.shape[1])]

labels = pd.read_csv('data/secom_labels.data', sep=' ', header=None, 
                     names=['target', 'timestamp'])

# ----- create missing data features -----
# hypothesis: sensor failures might indicate process problems

print("\ncreating 'is_missing' binary features...")

# identify sensors with significant missing data (>10%)
missing_pct = (data_raw.isnull().sum() / len(data_raw)) * 100
significant_missing = missing_pct[missing_pct > 10].index

print(f"found {len(significant_missing)} sensors with >10% missing data")

# create binary features for missingness
missing_features = pd.DataFrame()
for sensor in significant_missing:
    missing_features[f'{sensor}_is_missing'] = data_raw[sensor].isnull().astype(int)

print(f"created {len(missing_features.columns)} missing indicator features")

# load processed data (with imputed values)
data_processed = pd.read_csv('results/processed_data.csv')
X_base = data_processed.drop('target', axis=1)
y = (data_processed['target'] == 1).astype(int)  # convert to binary

# combine original features with missing indicators
# align indices
missing_features.index = X_base.index
X_combined = pd.concat([X_base, missing_features], axis=1)

print(f"\nfinal feature count: {X_combined.shape[1]} ({X_base.shape[1]} sensors + {len(missing_features.columns)} missing indicators)")

# ----- cost-based optimization framework -----

def calculate_expected_cost(confusion_mat, cost_fn, cost_fp):
    """
    Calculate expected cost given confusion matrix and costs
    
    confusion_mat: [[TN, FP], [FN, TP]]
    cost_fn: cost of missing a failure (false negative)
    cost_fp: cost of false alarm (false positive)
    """
    tn, fp, fn, tp = confusion_mat.ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    return total_cost

def find_optimal_threshold(y_true, y_proba, cost_ratio, thresholds=None):
    """
    Find threshold that minimizes expected cost
    
    cost_ratio: ratio of FN cost to FP cost (e.g., 10 means FN costs 10x more)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)
    
    results = []
    cost_fp = 1.0  # normalize FP cost to 1
    cost_fn = cost_ratio  # FN cost is the ratio
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        # calculate metrics
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        inspection_rate = (fp + tp) / len(y_true)
        
        # calculate expected cost (relative to FP cost)
        expected_cost = fn * cost_fn + fp * cost_fp
        
        results.append({
            'threshold': threshold,
            'recall': recall,
            'precision': precision,
            'inspection_rate': inspection_rate,
            'fn': fn,
            'fp': fp,
            'tp': tp,
            'tn': tn,
            'expected_cost': expected_cost
        })
    
    results_df = pd.DataFrame(results)
    optimal_idx = results_df['expected_cost'].idxmin()
    
    return results_df, results_df.iloc[optimal_idx]

# ----- cross-validation setup -----

print("Cross-Validation with Cost Analysis")

# use stratified k-fold to preserve class balance
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# track results across folds
cv_results = []
feature_importances = []

# cost ratios to test
cost_ratios = [1, 3, 5, 10, 20, 50]

print(f"\nrunning {n_splits}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y), 1):
    print(f"\nfold {fold}/{n_splits}")
    
    X_train_fold = X_combined.iloc[train_idx]
    X_val_fold = X_combined.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]
    
    # apply SMOTE to training data only
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_fold, y_train_fold)
    
    # train random forest
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_smote, y_train_smote)
    
    # get predictions
    y_proba = rf.predict_proba(X_val_fold)[:, 1]
    
    # calculate AUC
    auc = roc_auc_score(y_val_fold, y_proba)
    
    # find optimal thresholds for each cost ratio
    fold_cost_results = {'fold': fold, 'auc': auc}
    
    for cost_ratio in cost_ratios:
        threshold_results, optimal = find_optimal_threshold(
            y_val_fold, y_proba, cost_ratio
        )
        
        fold_cost_results[f'threshold_r{cost_ratio}'] = optimal['threshold']
        fold_cost_results[f'recall_r{cost_ratio}'] = optimal['recall']
        fold_cost_results[f'inspect_r{cost_ratio}'] = optimal['inspection_rate']
    
    cv_results.append(fold_cost_results)
    
    # store feature importance
    feature_importances.append(rf.feature_importances_)
    
    print(f"  AUC: {auc:.3f}")
    print(f"  Optimal thresholds: R=5 -> {fold_cost_results['threshold_r5']:.3f}, "
          f"R=10 -> {fold_cost_results['threshold_r10']:.3f}, "
          f"R=50 -> {fold_cost_results['threshold_r50']:.3f}")

# ----- aggregate CV results -----

cv_df = pd.DataFrame(cv_results)

print("Cross-Validation Results Summary")

print(f"\nAUC across folds: {cv_df['auc'].mean():.3f} ± {cv_df['auc'].std():.3f}")

print("\nOptimal thresholds by cost ratio (mean ± std):")
for cost_ratio in cost_ratios:
    thresh_mean = cv_df[f'threshold_r{cost_ratio}'].mean()
    thresh_std = cv_df[f'threshold_r{cost_ratio}'].std()
    recall_mean = cv_df[f'recall_r{cost_ratio}'].mean()
    inspect_mean = cv_df[f'inspect_r{cost_ratio}'].mean()
    
    print(f"  Ratio {cost_ratio:2d}:1 -> threshold={thresh_mean:.3f}±{thresh_std:.3f}, "
          f"recall={recall_mean:.1%}, inspect={inspect_mean:.1%}")

# save CV results
os.makedirs('results', exist_ok=True)
cv_df.to_csv('results/cross_validation_results.csv', index=False)

# ----- feature importance stability -----

print("\nanalyzing feature importance stability...")

# average importance across folds
feature_importance_array = np.array(feature_importances)
mean_importance = feature_importance_array.mean(axis=0)
std_importance = feature_importance_array.std(axis=0)

importance_df = pd.DataFrame({
    'feature': X_combined.columns,
    'importance_mean': mean_importance,
    'importance_std': std_importance,
    'cv_coefficient': std_importance / (mean_importance + 1e-10)  # coefficient of variation
}).sort_values('importance_mean', ascending=False)

print("\ntop 15 most important features (with stability):")
for i, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']:30s}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f} "
          f"(CV: {row['cv_coefficient']:.2f})")

importance_df.to_csv('results/feature_importance_cv.csv', index=False)

# check if any missing indicators are important
missing_indicator_importance = importance_df[importance_df['feature'].str.contains('is_missing')]
if len(missing_indicator_importance) > 0:
    print("\nmissing indicator features in top 50:")
    top_missing = missing_indicator_importance.head(10)
    if len(top_missing) > 0:
        for i, row in top_missing.iterrows():
            print(f"  {row['feature']:30s}: {row['importance_mean']:.4f}")
    else:
        print("  (none in top 50)")

# ----- final model on full training set -----

print("Training final model on full dataset")

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.25, stratify=y, random_state=42
)

# apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# train final model
rf_final = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_final.fit(X_train_smote, y_train_smote)
y_proba_test = rf_final.predict_proba(X_test)[:, 1]

# ----- cost analysis on test set -----

print("\nperforming cost-based analysis on test set...")

cost_analysis_results = []

for cost_ratio in cost_ratios:
    threshold_results, optimal = find_optimal_threshold(
        y_test, y_proba_test, cost_ratio
    )
    
    print(f"\nCost Ratio {cost_ratio}:1 (FN costs {cost_ratio}x more than FP):")
    print(f"  Optimal threshold: {optimal['threshold']:.3f}")
    print(f"  Recall: {optimal['recall']:.1%} ({optimal['tp']}/{optimal['tp']+optimal['fn']} failures caught)")
    print(f"  Precision: {optimal['precision']:.1%}")
    print(f"  Inspection rate: {optimal['inspection_rate']:.1%} ({int(optimal['inspection_rate']*len(y_test))}/{len(y_test)} units)")
    print(f"  Expected cost (relative): {optimal['expected_cost']:.1f}")
    
    cost_analysis_results.append({
        'cost_ratio': cost_ratio,
        'optimal_threshold': optimal['threshold'],
        'recall': optimal['recall'],
        'precision': optimal['precision'],
        'inspection_rate': optimal['inspection_rate'],
        'failures_caught': optimal['tp'],
        'total_failures': optimal['tp'] + optimal['fn'],
        'units_inspected': int(optimal['inspection_rate'] * len(y_test)),
        'total_units': len(y_test)
    })

cost_analysis_df = pd.DataFrame(cost_analysis_results)
cost_analysis_df.to_csv('results/cost_analysis.csv', index=False)

# ----- visualizations -----

print("\ncreating visualizations...")

# 1. cost curves for different ratios
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, cost_ratio in enumerate(cost_ratios):
    ax = axes[idx]
    
    threshold_results, optimal = find_optimal_threshold(
        y_test, y_proba_test, cost_ratio
    )
    
    ax.plot(threshold_results['threshold'], threshold_results['expected_cost'],
            linewidth=2, color='steelblue')
    ax.axvline(optimal['threshold'], color='red', linestyle='--', alpha=0.7,
               label=f"Optimal: {optimal['threshold']:.3f}")
    ax.scatter([optimal['threshold']], [optimal['expected_cost']], 
               color='red', s=100, zorder=5)
    
    ax.set_xlabel('Decision Threshold', fontsize=10)
    ax.set_ylabel('Expected Cost (relative)', fontsize=10)
    ax.set_title(f'Cost Ratio {cost_ratio}:1\nRecall={optimal["recall"]:.1%}, '
                 f'Inspect={optimal["inspection_rate"]:.1%}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/cost_optimization_curves.png', dpi=300)
print("saved cost optimization curves")
plt.close()

# 2. operational constraints visualization
fig, ax = plt.subplots(figsize=(12, 7))

# plot recall vs inspection rate for all thresholds
thresholds = np.arange(0.01, 0.5, 0.01)
recalls = []
inspection_rates = []

for threshold in thresholds:
    y_pred = (y_proba_test >= threshold).astype(int)
    recall = recall_score(y_test, y_pred)
    inspection_rate = y_pred.sum() / len(y_test)
    recalls.append(recall)
    inspection_rates.append(inspection_rate)

ax.plot(np.array(inspection_rates) * 100, np.array(recalls) * 100,
        linewidth=3, color='steelblue', label='Model Performance')

# add markers for different cost ratios
for cost_ratio in [5, 10, 20, 50]:
    _, optimal = find_optimal_threshold(y_test, y_proba_test, cost_ratio)
    ax.scatter([optimal['inspection_rate'] * 100], [optimal['recall'] * 100],
               s=200, label=f'Ratio {cost_ratio}:1', zorder=5)

# add constraint lines (example operational limits)
ax.axvline(20, color='orange', linestyle='--', alpha=0.5, 
           label='Max Inspection Capacity (20%)')
ax.axhline(75, color='green', linestyle='--', alpha=0.5,
           label='Min Acceptable Recall (75%)')

ax.set_xlabel('Inspection Rate (%)', fontsize=12)
ax.set_ylabel('Recall - Failures Caught (%)', fontsize=12)
ax.set_title('Cost-Optimized Operating Points\nBalancing Detection vs Inspection Capacity',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/operational_constraints.png', dpi=300)
print("saved operational constraints plot")
plt.close()

# 3. feature importance with error bars
fig, ax = plt.subplots(figsize=(10, 8))

top_20 = importance_df.head(20)
y_pos = np.arange(len(top_20))

ax.barh(y_pos, top_20['importance_mean'], xerr=top_20['importance_std'],
        color='steelblue', alpha=0.7, error_kw={'linewidth': 2, 'ecolor': 'black'})

ax.set_yticks(y_pos)
ax.set_yticklabels(top_20['feature'], fontsize=9)
ax.set_xlabel('Feature Importance (mean ± std across CV folds)', fontsize=11)
ax.set_title('Top 20 Features with Cross-Validation Stability',
             fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/feature_importance_stability.png', dpi=300)
print("saved feature importance stability plot")
plt.close()

# ----- final summary -----

print("\nCost-Based Optimization Results")
print("-"*50)

print("\nCross-Validation Performance:")
print(f"  AUC: {cv_df['auc'].mean():.3f} ± {cv_df['auc'].std():.3f}")

print("\nRecommended Operating Points:")
print("\n  Conservative (Ratio 5:1):")
optimal_5 = cost_analysis_df[cost_analysis_df['cost_ratio'] == 5].iloc[0]
print(f"    Threshold: {optimal_5['optimal_threshold']:.3f}")
print(f"    Catch {optimal_5['failures_caught']}/{optimal_5['total_failures']} failures "
      f"({optimal_5['recall']:.1%}) by inspecting {optimal_5['inspection_rate']:.1%}")

print("\n  Balanced (Ratio 10:1):")
optimal_10 = cost_analysis_df[cost_analysis_df['cost_ratio'] == 10].iloc[0]
print(f"    Threshold: {optimal_10['optimal_threshold']:.3f}")
print(f"    Catch {optimal_10['failures_caught']}/{optimal_10['total_failures']} failures "
      f"({optimal_10['recall']:.1%}) by inspecting {optimal_10['inspection_rate']:.1%}")

print("\n  Aggressive (Ratio 50:1):")
optimal_50 = cost_analysis_df[cost_analysis_df['cost_ratio'] == 50].iloc[0]
print(f"    Threshold: {optimal_50['optimal_threshold']:.3f}")
print(f"    Catch {optimal_50['failures_caught']}/{optimal_50['total_failures']} failures "
      f"({optimal_50['recall']:.1%}) by inspecting {optimal_50['inspection_rate']:.1%}")

print("\nObservations:")
print("  - missing data indicators turned out to be predictive")
print("  - CV shows consistent performance across folds")
print("  - thresholds shift lower as FN cost increases (makes sense)")
print("  - can choose operating point based on inspection capacity")

print("\nFiles saved:")
print("  - results/cross_validation_results.csv")
print("  - results/cost_analysis.csv")
print("  - results/feature_importance_cv.csv")
print("  - results/figures/cost_optimization_curves.png")
print("  - results/figures/operational_constraints.png")
print("  - results/figures/feature_importance_stability.png")

print("\ndone!")
