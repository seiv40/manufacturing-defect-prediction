"""
Feature selection and engineering
Reducing 590 sensors to something manageable
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
import os
warnings.filterwarnings('ignore')

print("loading data...")

# load the cleaned data from day 1
data = pd.read_csv('data/secom.data', sep=' ', header=None, na_values='NaN')
data.columns = [f'sensor_{i}' for i in range(data.shape[1])]

labels = pd.read_csv('data/secom_labels.data', sep=' ', header=None, 
                     names=['target', 'timestamp'])

print(f"starting with {data.shape[1]} sensors")

# ----- remove constant sensors -----

# these have zero variance, completely useless
constant_sensors = data.columns[data.nunique() == 1]
print(f"\nremoving {len(constant_sensors)} constant sensors")

data = data.drop(columns=constant_sensors)
print(f"remaining: {data.shape[1]} sensors")

# ----- remove high-missing sensors -----

# if >50% missing, sensor is too unreliable
missing_pct = (data.isnull().sum() / len(data)) * 100
high_missing = missing_pct[missing_pct > 50].index

print(f"\nremoving {len(high_missing)} sensors with >50% missing data")
data = data.drop(columns=high_missing)
print(f"remaining: {data.shape[1]} sensors")

# also remove near-zero variance (std < 0.01)
# low_var = data.columns[data.std() < 0.01]
# print(f"removing {len(low_var)} low variance sensors")
# data = data.drop(columns=low_var)
# actually let's keep these for now, might be useful

# ----- correlation analysis -----

print("\nanalyzing sensor-sensor correlations...")

# this might take a minute with 400+ sensors
corr_matrix = data.corr()

# find highly correlated pairs (>0.95)
# these are probably measuring the same thing
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.95:
            high_corr_pairs.append({
                'sensor1': corr_matrix.columns[i],
                'sensor2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

print(f"found {len(high_corr_pairs)} highly correlated pairs (>0.95)")

# for highly correlated pairs, keep one and drop the other
# strategy: keep the one with less missing data
to_drop = set()
for pair in high_corr_pairs:
    s1, s2 = pair['sensor1'], pair['sensor2']
    missing_s1 = data[s1].isnull().sum()
    missing_s2 = data[s2].isnull().sum()
    
    # drop the one with more missing data
    if missing_s1 > missing_s2:
        to_drop.add(s1)
    else:
        to_drop.add(s2)

print(f"dropping {len(to_drop)} redundant sensors")
data = data.drop(columns=list(to_drop))
print(f"remaining: {data.shape[1]} sensors")

# save correlation matrix visualization
os.makedirs('results/figures', exist_ok=True)

# sample for visualization (can't plot 400+ sensors)
np.random.seed(42)
sample_sensors = np.random.choice(data.columns, size=min(50, len(data.columns)), replace=False)
sample_corr = data[sample_sensors].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(sample_corr, cmap='coolwarm', center=0, 
            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title(f'Sensor-Sensor Correlations (sample of {len(sample_sensors)})')
plt.tight_layout()
plt.savefig('results/figures/sensor_correlation_matrix.png', dpi=300)
print("saved correlation heatmap")
plt.close()

# ----- imputation strategy -----

print("\ntesting imputation methods...")

# compare a few approaches on a small subset
# using sensor_59 (top correlated from day 1) as test case
test_sensor = 'sensor_59'
test_data = data[[test_sensor]].copy()

# method 1: simple median
imp_median = SimpleImputer(strategy='median')
median_result = imp_median.fit_transform(test_data)

# method 2: mean
imp_mean = SimpleImputer(strategy='mean')
mean_result = imp_mean.fit_transform(test_data)

# method 3: KNN imputation
# imp_knn = KNNImputer(n_neighbors=5)
# knn_result = imp_knn.fit_transform(test_data)
# actually KNN is really slow with this many features, skip it

# for now, going with median imputation
# it's robust to outliers and works well with sensor data

print("using median imputation for remaining missing values")
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(
    imputer.fit_transform(data),
    columns=data.columns
)

print(f"missing values after imputation: {data_imputed.isnull().sum().sum()}")

# ----- feature selection with mutual information -----

print("\ncalculating mutual information scores...")

# mutual information captures non-linear relationships
# better than just correlation for complex data
mi_scores = mutual_info_classif(data_imputed, labels['target'], random_state=42)

mi_df = pd.DataFrame({
    'sensor': data_imputed.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\ntop 10 sensors by mutual information:")
for i, row in mi_df.head(10).iterrows():
    print(f"  {row['sensor']}: {row['mi_score']:.4f}")

# visualize
fig, ax = plt.subplots(figsize=(12, 8))
top_mi = mi_df.head(30)
ax.barh(range(len(top_mi)), top_mi['mi_score'], color='steelblue', alpha=0.7)
ax.set_yticks(range(len(top_mi)))
ax.set_yticklabels(top_mi['sensor'], fontsize=9)
ax.set_xlabel('Mutual Information Score')
ax.set_title('Top 30 Sensors by Mutual Information with Failure')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/mutual_information_scores.png', dpi=300)
print("saved mutual information plot")
plt.close()

# ----- feature selection with random forest -----

print("\ntraining random forest for feature importance...")

# quick random forest to get feature importances
# using class_weight to handle imbalance
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(data_imputed, labels['target'])

# get importances
rf_importance = pd.DataFrame({
    'sensor': data_imputed.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\ntop 10 sensors by random forest importance:")
for i, row in rf_importance.head(10).iterrows():
    print(f"  {row['sensor']}: {row['importance']:.4f}")

# visualize
fig, ax = plt.subplots(figsize=(12, 8))
top_rf = rf_importance.head(30)
ax.barh(range(len(top_rf)), top_rf['importance'], color='forestgreen', alpha=0.7)
ax.set_yticks(range(len(top_rf)))
ax.set_yticklabels(top_rf['sensor'], fontsize=9)
ax.set_xlabel('Random Forest Importance')
ax.set_title('Top 30 Sensors by Random Forest Feature Importance')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/random_forest_importance.png', dpi=300)
print("saved random forest importance plot")
plt.close()

# ----- combine selection methods -----

# merge all three ranking methods
# from day 1: correlation with target
corr_with_target = pd.read_csv('results/sensor_correlations.csv')

# combine rankings
combined = data_imputed.columns.to_frame(name='sensor')
combined = combined.merge(corr_with_target[['sensor', 'correlation']], on='sensor', how='left')
combined = combined.merge(mi_df[['sensor', 'mi_score']], on='sensor', how='left')
combined = combined.merge(rf_importance[['sensor', 'importance']], on='sensor', how='left')

# fill any missing values with 0
combined = combined.fillna(0)

# create composite score (average of normalized ranks)
# normalize each method to 0-1 scale
combined['corr_norm'] = combined['correlation'] / combined['correlation'].max()
combined['mi_norm'] = combined['mi_score'] / combined['mi_score'].max()
combined['rf_norm'] = combined['importance'] / combined['importance'].max()

# composite score (equal weight to each method)
combined['composite_score'] = (
    combined['corr_norm'] + 
    combined['mi_norm'] + 
    combined['rf_norm']
) / 3

combined = combined.sort_values('composite_score', ascending=False)

print("\ntop 20 sensors by composite score:")
for i, row in combined.head(20).iterrows():
    print(f"  {row['sensor']}: {row['composite_score']:.4f}")

# ----- select final feature set -----

# choosing top 75 features (good balance)
# could also try 50 or 100, but 75 seems reasonable
n_features = 75
selected_features = combined.head(n_features)['sensor'].tolist()

print(f"\nselected {len(selected_features)} final features")
print(f"reduced from 590 → {len(selected_features)} ({len(selected_features)/590*100:.1f}%)")

# create final dataset
final_data = data_imputed[selected_features].copy()
final_data['target'] = labels['target']

# save
final_data.to_csv('results/processed_data.csv', index=False)
print("\nsaved processed dataset to results/processed_data.csv")

# save feature rankings
combined.to_csv('results/feature_rankings.csv', index=False)
print("saved feature rankings to results/feature_rankings.csv")

# ----- summary stats -----

print("\nFeature Selection Complete!")
print(f"Original sensors: 590")
print(f"After removing constants: {590 - len(constant_sensors)}")
print(f"After removing high-missing: {590 - len(constant_sensors) - len(high_missing)}")
print(f"After removing redundant: {data.shape[1]}")
print(f"Final selected features: {len(selected_features)}")
print(f"\nReduction: {590} → {len(selected_features)} ({(1 - len(selected_features)/590)*100:.1f}% reduction)")

print("\ndone! ready for modeling")

# notes for next steps:
# - try different n_features values (50, 75, 100) and compare model performance
# - maybe create interaction features? (sensor_A * sensor_B)
# - could try PCA as alternative (but loses interpretability)
# - remember to use stratified split for imbalanced data
# - definitely need to handle class imbalance (SMOTE or class weights)
