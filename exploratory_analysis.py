"""
Exploratory data analysis for secom dataset
Trying to understand the class imbalance, missing data, and which sensors matter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

print("loading data...")

# ----- load the data files -----

# using relative paths so this works on any OS
data = pd.read_csv('data/secom.data', 
                   sep=' ',  # took me a while to figure out it's space delimited
                   header=None,
                   na_values='NaN')

# give columns actual names
data.columns = [f'sensor_{i}' for i in range(data.shape[1])]

labels = pd.read_csv('data/secom_labels.data',
                     sep=' ',
                     header=None,
                     names=['target', 'timestamp'])

print(f"data shape: {data.shape}")
print(f"labels shape: {labels.shape}")

# sanity check
assert len(data) == len(labels)

# ----- class distribution -----

# what's the breakdown of pass vs fail?
# -1 = pass, 1 = fail (according to the readme)
pass_count = (labels['target'] == -1).sum()
fail_count = (labels['target'] == 1).sum()

print(f"\npass: {pass_count}, fail: {fail_count}")
print(f"failure rate: {fail_count/len(labels):.2%}")

# wow that's really imbalanced (14:1)
# definitely gonna need to handle this somehow - maybe SMOTE?

# ----- missing data check -----

total_missing = data.isnull().sum().sum()
pct_missing = total_missing / data.size

print(f"\ntotal missing: {total_missing:,} ({pct_missing:.2%})")

# which sensors have the most missing?
missing_per_sensor = data.isnull().sum()
missing_pct = (missing_per_sensor / len(data)) * 100

print(f"sensors >50% missing: {(missing_pct > 50).sum()}")
print(f"sensors >10% missing: {(missing_pct > 10).sum()}")

# hmm 28 sensors are more than half empty
# probably should just drop those

# ----- visualizations -----

# create output directory if it doesn't exist
os.makedirs('results/figures', exist_ok=True)

# class distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
counts = pd.Series([pass_count, fail_count], index=['Pass', 'Fail'])
counts.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.8)
ax.set_title('Class Distribution')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# add counts on bars
for i, v in enumerate(counts.values):
    pct = v / len(labels) * 100
    ax.text(i, v + 30, f'{v:,}\n({pct:.1f}%)', ha='center', fontweight='bold')

# pie chart
ax = axes[1]
ax.pie([pass_count, fail_count], 
       labels=['Pass', 'Fail'],
       autopct='%1.1f%%',
       colors=['green', 'red'],
       startangle=90)
ax.set_title('Failure Rate')

plt.tight_layout()
plt.savefig('results/figures/class_distribution.png', dpi=300)
print("\nsaved class distribution plot")
plt.close()

# ----- missing data analysis -----

# make a df to track missing data
missing_df = pd.DataFrame({
    'sensor': missing_per_sensor.index,
    'missing_count': missing_per_sensor.values,
    'missing_pct': missing_pct.values
}).sort_values('missing_pct', ascending=False)

# plot top 50 worst offenders
fig, axes = plt.subplots(2, 1, figsize=(15, 11))

ax = axes[0]
top_missing = missing_df.head(50)

# color by severity
colors = []
for pct in top_missing['missing_pct']:
    if pct > 75:
        colors.append('darkred')
    elif pct > 50:
        colors.append('orangered')
    elif pct > 25:
        colors.append('orange')
    else:
        colors.append('steelblue')

ax.barh(range(len(top_missing)), top_missing['missing_pct'], color=colors, alpha=0.7)
ax.set_yticks(range(len(top_missing)))
ax.set_yticklabels(top_missing['sensor'], fontsize=8)
ax.set_xlabel('Missing Data (%)')
ax.set_title('Top 50 Sensors by Missing Data')
ax.axvline(50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
ax.invert_yaxis()
ax.legend()

# histogram
ax = axes[1]
ax.hist(missing_pct, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Missing Data (%)')
ax.set_ylabel('Number of Sensors')
ax.set_title('Distribution of Missing Data')
ax.axvline(missing_pct.median(), color='red', linestyle='--', 
           label=f'median: {missing_pct.median():.1f}%')
ax.axvline(missing_pct.mean(), color='orange', linestyle='--',
           label=f'mean: {missing_pct.mean():.1f}%')
ax.legend()

plt.tight_layout()
plt.savefig('results/figures/missing_data_overview.png', dpi=300)
print("saved missing data plots")
plt.close()

# ----- missing data heatmap -----

# can't plot all 590 sensors so sample some
np.random.seed(42)
sample_sensors = missing_df.head(100)['sensor'].values
sample_rows = np.random.choice(data.index, size=200, replace=False)
sample_data = data.loc[sample_rows, sample_sensors]

fig, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(sample_data.isnull(), 
            cmap=['white', 'black'],
            cbar_kws={'label': 'Missing'},
            ax=ax,
            yticklabels=False)
ax.set_title('Missing Data Pattern (sample)')
ax.set_xlabel('Sensor')

plt.tight_layout()
plt.savefig('results/figures/missing_data_heatmap.png', dpi=300)
print("saved heatmap")
plt.close()

# ----- correlations with target -----

print("\nchecking which sensors correlate with failures...")

# combine for easier analysis
data_with_target = data.copy()
data_with_target['target'] = labels['target']

# only use sensors with <50% missing (others are too unreliable)
usable_sensors = missing_df[missing_df['missing_pct'] < 50]['sensor'].values
print(f"analyzing {len(usable_sensors)} sensors...")

# calculate correlations
correlations = []
for sensor in usable_sensors:
    temp = data_with_target[[sensor, 'target']].dropna()
    
    if len(temp) > 100:  # need enough data
        corr = temp[sensor].corr(temp['target'])
        correlations.append({
            'sensor': sensor,
            'correlation': abs(corr),
            'raw_correlation': corr
        })

corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

print("\ntop 10 correlated sensors:")
for i, row in corr_df.head(10).iterrows():
    print(f"  {row['sensor']}: {row['raw_correlation']:+.4f}")

# interesting - correlations are pretty weak (max ~0.15)
# might need nonlinear models to capture relationships

# plot top correlations
fig, ax = plt.subplots(figsize=(12, 8))
top_corr = corr_df.head(30)

colors = ['red' if x < 0 else 'green' for x in top_corr['raw_correlation']]

ax.barh(range(len(top_corr)), top_corr['raw_correlation'], color=colors, alpha=0.7)
ax.set_yticks(range(len(top_corr)))
ax.set_yticklabels(top_corr['sensor'], fontsize=9)
ax.set_xlabel('Correlation with Failure')
ax.set_title('Top 30 Sensors by Correlation')
ax.axvline(0, color='black', linestyle='-', linewidth=1)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/feature_correlations.png', dpi=300)
print("saved correlation plot")
plt.close()

# ----- other checks -----

# constant sensors (useless)
constant_sensors = (data.nunique() == 1).sum()
print(f"\nconstant sensors: {constant_sensors}")

# near-zero variance (probably useless too)
low_variance = (data.std() < 0.01).sum()
print(f"low variance sensors: {low_variance}")

# ----- save results -----

print("\nsaving analysis files...")

missing_df.to_csv('results/missing_data_analysis.csv', index=False)
if len(corr_df) > 0:
    corr_df.to_csv('results/sensor_correlations.csv', index=False)

# write a quick summary
with open('results/eda_summary.txt', 'w') as f:
    f.write(f"SECOM Exploratory Analysis Summary\n\n")
    f.write(f"Dataset: {len(data)} samples, {data.shape[1]} sensors\n")
    f.write(f"Pass: {pass_count}, Fail: {fail_count} (imbalance: {pass_count/fail_count:.1f}:1)\n")
    f.write(f"Missing data: {pct_missing:.2%}\n")
    f.write(f"Sensors >50% missing: {(missing_pct > 50).sum()}\n")
    f.write(f"Constant sensors: {constant_sensors}\n")
    f.write(f"Low variance sensors: {low_variance}\n\n")
    f.write(f"Top correlated sensors:\n")
    for i, row in corr_df.head(10).iterrows():
        f.write(f"  {row['sensor']}: {row['raw_correlation']:+.4f}\n")

print("done!")

# notes for next steps:
# - need to handle class imbalance (smote? class weights?)
# - remove constant sensors (116 of them!)
# - decide what to do with high-missing sensors
# - feature selection to reduce dimensionality
# - check for multicollinearity between sensors
# - maybe try different imputation methods?

# things that didn't work:
# - tried seaborn pairplot but way too many features
# - attempted to look for time patterns but timestamps seem arbitrary
# - wanted to do PCA but should probably scale/clean data first
