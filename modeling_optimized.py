"""
Manufacturing defect prediction with threshold optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (recall_score, precision_score, roc_auc_score, 
                             confusion_matrix, f1_score)
from imblearn.over_sampling import SMOTE
import warnings
import os
warnings.filterwarnings('ignore')

print("SECOM Manufacturing Defect Prediction")
print()

print("Loading data...")
data = pd.read_csv('data/secom.data', sep=' ', header=None, na_values='NaN')
data.columns = [f'sensor_{i}' for i in range(data.shape[1])]

labels = pd.read_csv('data/secom_labels.data', sep=' ', header=None, 
                     names=['target', 'timestamp'])

y = (labels['target'] == 1).astype(int)

print(f"Dataset: {data.shape[0]} samples, {data.shape[1]} sensors")
print(f"Failure rate: {y.sum()}/{len(y)} ({y.mean():.1%})")
print()

print("Data cleaning...")
constant_sensors = data.columns[data.nunique() == 1]
print(f"  Removed {len(constant_sensors)} constant sensors")
data = data.drop(columns=constant_sensors)

missing_pct = (data.isnull().sum() / len(data)) * 100
high_missing = missing_pct[missing_pct > 50].index
print(f"  Removed {len(high_missing)} sensors with >50% missing data")
data = data.drop(columns=high_missing)
print(f"  Remaining: {data.shape[1]} sensors")
print()

print("Train/test split (stratified, 75/25)...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    data, y, test_size=0.25, stratify=y, random_state=42
)
print(f"  Train: {len(X_train_raw)} samples ({y_train.sum()} failures)")
print(f"  Test: {len(X_test_raw)} samples ({y_test.sum()} failures)")
print()

print("Imputing missing values (median)...")
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train_raw),
    columns=X_train_raw.columns,
    index=X_train_raw.index
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test_raw),
    columns=X_test_raw.columns,
    index=X_test_raw.index
)
print(f"  Complete")
print()

print("Feature selection (mutual information)...")
selector = SelectKBest(mutual_info_classif, k=75)
X_train_selected = selector.fit_transform(X_train_imputed, y_train)
X_test_selected = selector.transform(X_test_imputed)

selected_mask = selector.get_support()
selected_features = X_train_imputed.columns[selected_mask].tolist()

print(f"  Selected {len(selected_features)}/{data.shape[1]} features ({(1-len(selected_features)/data.shape[1])*100:.0f}% reduction)")

X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train_imputed.index)
X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test_imputed.index)
print()

print("Creating missing data indicators...")
train_missing_pct = (X_train_raw.isnull().sum() / len(X_train_raw)) * 100
significant_missing = train_missing_pct[train_missing_pct > 10].index

missing_train = pd.DataFrame()
missing_test = pd.DataFrame()

for sensor in significant_missing:
    if sensor in X_train_raw.columns:
        missing_train[f'{sensor}_is_missing'] = X_train_raw[sensor].isnull().astype(int)
        missing_test[f'{sensor}_is_missing'] = X_test_raw[sensor].isnull().astype(int)

print(f"  Created {len(missing_train.columns)} missing indicators")

X_train_final = pd.concat([X_train_selected, missing_train], axis=1)
X_test_final = pd.concat([X_test_selected, missing_test], axis=1)

print(f"  Total features: {X_train_final.shape[1]}")
print()

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_final, y_train)
print(f"  Training set after SMOTE: {len(y_train_smote)} samples")
print()

print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_smote, y_train_smote)
print("  Model trained")
print()

print("Cross-validation (5-fold stratified)...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_importances = []
cv_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train_final, y_train)):
    X_fold_train = X_train_final.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train_final.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    smote_fold = SMOTE(random_state=42)
    X_fold_smote, y_fold_smote = smote_fold.fit_resample(X_fold_train, y_fold_train)
    
    rf_fold = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_fold.fit(X_fold_smote, y_fold_smote)
    
    y_val_proba = rf_fold.predict_proba(X_fold_val)[:, 1]
    auc_fold = roc_auc_score(y_fold_val, y_val_proba)
    cv_scores.append(auc_fold)
    
    cv_importances.append(rf_fold.feature_importances_)

cv_importances = np.array(cv_importances)
cv_scores = np.array(cv_scores)

print(f"  CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print()

print("Test set evaluation...")
y_proba = rf.predict_proba(X_test_final)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f"  ROC-AUC: {auc:.3f}")
print()

print("Threshold optimization results:")
threshold_results = []

for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]:
    y_pred = (y_proba >= threshold).astype(int)
    
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    
    n_predicted_fail = y_pred.sum()
    inspection_rate = n_predicted_fail / len(y_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print(f"  Threshold {threshold:.2f}: Recall {recall:.1%} ({tp}/{tp+fn}), "
          f"Precision {precision:.1%}, Inspection {inspection_rate:.1%}")
    
    threshold_results.append({
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'inspection_rate': inspection_rate,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    })

print()

importance_mean = cv_importances.mean(axis=0)
importance_std = cv_importances.std(axis=0)
cv_coefficient = importance_std / (importance_mean + 1e-10)

importance_df = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance_mean': importance_mean,
    'importance_std': importance_std,
    'cv_coefficient': cv_coefficient
}).sort_values('importance_mean', ascending=False)

print("Top 10 most important features:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['importance_mean']:.4f}")

print()
print("Saving results...")

os.makedirs('results', exist_ok=True)

threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv('results/threshold_tuning.csv', index=False)

importance_df.to_csv('results/feature_importance_cv.csv', index=False)

importance_df[['feature', 'importance_mean']].rename(
    columns={'importance_mean': 'importance'}
).to_csv('results/feature_importance.csv', index=False)

pd.DataFrame({'feature': selected_features}).to_csv(
    'results/selected_features.csv', index=False
)

# save train/test splits separately for downstream scripts
X_train_final.to_csv('results/X_train.csv', index=True)
X_test_final.to_csv('results/X_test.csv', index=True)
y_train.to_csv('results/y_train.csv', index=True, header=['target'])
y_test.to_csv('results/y_test.csv', index=True, header=['target'])

# also save combined for backward compatibility
X_combined = pd.concat([X_train_final, X_test_final])
y_combined = pd.concat([y_train, y_test])
processed_data = X_combined.copy()
processed_data['target'] = y_combined
processed_data = processed_data.sort_index()
processed_data.to_csv('results/processed_data.csv', index=False)

print("  Saved train/test splits separately")
print("  Saved all results to results/")
print()
print("Complete")

