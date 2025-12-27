"""
Sensor clustering and process stage analysis
Trying to group sensors by what they measure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

print("loading data for domain analysis...")

# load processed data
data = pd.read_csv('results/processed_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# load feature importance
importance = pd.read_csv('results/feature_importance_cv.csv')

print(f"\nanalyzing {X.shape[1]} sensors")

# ----- sensor correlation clustering -----

print("Sensor Correlation Clustering")

# compute correlation matrix
corr_matrix = X.corr()

# hierarchical clustering on correlation
# distance = 1 - |correlation|
dist_matrix = 1 - np.abs(corr_matrix)
dist_condensed = squareform(dist_matrix, checks=False)

# perform clustering
linkage = hierarchy.linkage(dist_condensed, method='ward')

# cut tree to get clusters
n_clusters = 8  # reasonable number for process stages
cluster_labels = hierarchy.fcluster(linkage, n_clusters, criterion='maxclust')

# create cluster dataframe
sensor_clusters = pd.DataFrame({
    'sensor': X.columns,
    'cluster': cluster_labels
})

# add importance info
sensor_clusters = sensor_clusters.merge(
    importance[['feature', 'importance_mean']], 
    left_on='sensor', 
    right_on='feature', 
    how='left'
).drop('feature', axis=1)

# sort by cluster and importance
sensor_clusters = sensor_clusters.sort_values(['cluster', 'importance_mean'], 
                                               ascending=[True, False])

print(f"\nidentified {n_clusters} sensor clusters based on correlation")

# analyze each cluster
print("\nCluster Summary:")
for cluster_id in range(1, n_clusters + 1):
    cluster_sensors = sensor_clusters[sensor_clusters['cluster'] == cluster_id]
    n_sensors = len(cluster_sensors)
    avg_importance = cluster_sensors['importance_mean'].mean()
    top_sensor = cluster_sensors.iloc[0]
    
    print(f"\nCluster {cluster_id}: {n_sensors} sensors")
    print(f"  Avg importance: {avg_importance:.4f}")
    print(f"  Top sensor: {top_sensor['sensor']} ({top_sensor['importance_mean']:.4f})")
    
    # show top 3 sensors in cluster
    if n_sensors > 1:
        print(f"  Key sensors: " + ", ".join(cluster_sensors['sensor'].head(3).values))

# save cluster assignments
sensor_clusters.to_csv('results/sensor_clusters.csv', index=False)

# ----- identify process stage hypotheses -----

print("Process Stage Hypothesis (based on sensor groupings)")

# high-importance clusters likely represent critical process stages
cluster_importance = sensor_clusters.groupby('cluster')['importance_mean'].agg(['mean', 'sum', 'count'])
cluster_importance = cluster_importance.sort_values('mean', ascending=False)

print("\nClusters ranked by average importance (likely process criticality):")
for idx, (cluster_id, row) in enumerate(cluster_importance.iterrows(), 1):
    print(f"  {idx}. Cluster {cluster_id}: avg={row['mean']:.4f}, "
          f"total={row['sum']:.4f}, n={int(row['count'])} sensors")

# hypothesis: high-importance clusters = critical process stages
print("\nHypothesized Process Stages:")
print("  Cluster with highest avg importance likely measures critical process")
print("  (e.g., etch depth, deposition uniformity, lithography alignment)")
print("  Clusters with many sensors may represent redundant measurements")
print("  Clusters with low importance may be routine monitoring")

# ----- early vs late detection analysis -----

print("Early Warning vs Late Detection Analysis")

# get top predictive sensors
top_sensors = importance.nlargest(20, 'importance_mean')

print("\nTop 20 predictive sensors (potential early warning signals):")
for i, row in top_sensors.iterrows():
    cluster_id = sensor_clusters[sensor_clusters['sensor'] == row['feature']]['cluster'].values
    cluster_id = cluster_id[0] if len(cluster_id) > 0 else 'N/A'
    print(f"  {row['feature']:25s}: importance={row['importance_mean']:.4f}, "
          f"stability(CV)={row['cv_coefficient']:.2f}, cluster={cluster_id}")

print("\nInterpretation:")
print("  - Low CV coefficient (<0.5) = stable importance = reliable early warning")
print("  - High CV coefficient (>1.0) = unstable = may be late/noisy detection")
print("  - Sensors in same cluster = measuring same process aspect")

# identify most stable predictors
stable_predictors = top_sensors[top_sensors['cv_coefficient'] < 0.5]
print(f"\nMost stable early warning sensors ({len(stable_predictors)} found):")
for i, row in stable_predictors.iterrows():
    print(f"  {row['feature']}: {row['importance_mean']:.4f}")

# ----- missing data pattern analysis -----

print("Missing Data Pattern Analysis")

# check if missing indicators are predictive
missing_features = importance[importance['feature'].str.contains('is_missing')]

if len(missing_features) > 0:
    print(f"\nFound {len(missing_features)} missing indicator features")
    print("\nTop missing indicators by importance:")
    
    for i, row in missing_features.nlargest(10, 'importance_mean').iterrows():
        base_sensor = row['feature'].replace('_is_missing', '')
        print(f"  {row['feature']:30s}: {row['importance_mean']:.4f}")
    
    print("\nInterpretation:")
    print("  Missing data from these sensors is PREDICTIVE of failures!")
    print("  Hypothesis: sensor failures may indicate process instability")
    print("  Recommendation: monitor these sensors for communication/measurement issues")
else:
    print("No missing indicator features found important")

# ----- visualizations -----

print("\ncreating domain interpretation visualizations...")

# 1. dendrogram
fig, ax = plt.subplots(figsize=(14, 8))

# only show top 50 sensors for readability (excluding missing indicators)
top_50_sensors = importance[~importance['feature'].str.contains('is_missing')].nlargest(50, 'importance_mean')['feature'].values
sensor_idx = [i for i, col in enumerate(X.columns) if col in top_50_sensors]

if len(sensor_idx) >= 2:
    # create subset correlation matrix
    sub_corr = corr_matrix.iloc[sensor_idx, sensor_idx]
    sub_dist = 1 - np.abs(sub_corr)
    sub_dist_condensed = squareform(sub_dist, checks=False)
    
    sub_linkage = hierarchy.linkage(sub_dist_condensed, method='ward')
    dendro = hierarchy.dendrogram(
        sub_linkage,
        labels=[X.columns[i] for i in sensor_idx],
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8
    )
    
    ax.set_title('Sensor Hierarchical Clustering (Top 50 by Importance)\nSimilar sensors cluster together', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Sensor', fontsize=10)
    ax.set_ylabel('Distance (1 - |correlation|)', fontsize=10)

plt.tight_layout()
plt.savefig('results/figures/sensor_clustering_dendrogram.png', dpi=300, bbox_inches='tight')
print("saved sensor clustering dendrogram")
plt.close()

# 2. cluster importance heatmap
fig, ax = plt.subplots(figsize=(12, 8))

# prepare data: top 5 sensors per cluster
cluster_data = []
for cluster_id in range(1, n_clusters + 1):
    cluster_sensors = sensor_clusters[sensor_clusters['cluster'] == cluster_id]
    top_5 = cluster_sensors.nlargest(5, 'importance_mean')
    for _, row in top_5.iterrows():
        cluster_data.append({
            'sensor': row['sensor'],
            'cluster': f'Cluster {cluster_id}',
            'importance': row['importance_mean']
        })

cluster_df = pd.DataFrame(cluster_data)

# pivot for heatmap
pivot_data = cluster_df.pivot_table(
    index='cluster', 
    columns='sensor', 
    values='importance', 
    fill_value=0
)

sns.heatmap(pivot_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Importance'})
ax.set_title('Sensor Importance by Cluster (Top 5 per Cluster)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Sensor', fontsize=10)
ax.set_ylabel('Cluster (Hypothesized Process Stage)', fontsize=10)

plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('results/figures/cluster_importance_heatmap.png', dpi=300, bbox_inches='tight')
print("saved cluster importance heatmap")
plt.close()

# 3. early warning timeline concept
fig, ax = plt.subplots(figsize=(14, 6))

# categorize sensors by stability and importance
early_warning = top_sensors[top_sensors['cv_coefficient'] < 0.5]['feature'].values
late_detection = top_sensors[top_sensors['cv_coefficient'] >= 0.5]['feature'].values

categories = []
for sensor in top_sensors['feature'].values[:15]:
    if sensor in early_warning:
        categories.append(('Early Warning', sensor, 
                          top_sensors[top_sensors['feature']==sensor]['importance_mean'].values[0]))
    else:
        categories.append(('Late Detection', sensor,
                          top_sensors[top_sensors['feature']==sensor]['importance_mean'].values[0]))

# plot
y_pos = np.arange(len(categories))
colors = ['green' if c[0]=='Early Warning' else 'orange' for c in categories]
importances = [c[2] for c in categories]

ax.barh(y_pos, importances, color=colors, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([c[1] for c in categories], fontsize=9)
ax.set_xlabel('Feature Importance', fontsize=11)
ax.set_title('Early Warning vs Late Detection Sensors\n(Green = stable predictors, Orange = variable predictors)', 
             fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label='Early Warning (stable)'),
    Patch(facecolor='orange', alpha=0.7, label='Late Detection (variable)')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('results/figures/early_vs_late_detection.png', dpi=300, bbox_inches='tight')
print("saved early vs late detection analysis")
plt.close()

# ----- summary report -----

print("\nDomain Analysis Summary")
print("-"*50)

print(f"\nSensor Organization:")
print(f"  - found {n_clusters} sensor clusters based on correlation")
print(f"  - clusters probably represent different process stages")
print(f"  - high-importance clusters = critical control points")

print(f"\nEarly Warning:")
print(f"  - {len(stable_predictors)} stable predictors identified")
print(f"  - these are consistent across CV folds")
print(f"  - can use for proactive monitoring")

if len(missing_features) > 0:
    print(f"\nMissing Data Findings:")
    print(f"  - {len(missing_features)} missing indicators turned out predictive")
    print(f"  - sensor failures might indicate process issues")
    print(f"  - worth monitoring sensor health as quality signal")

print("\nWhat this suggests:")
print("  1. focus maintenance on high-importance clusters")
print("  2. use stable predictors for real-time monitoring")
print("  3. investigate why sensors fail during problem runs")
print("  4. might need redundancy for critical single sensors")

print("\nFiles saved:")
print("  - results/sensor_clusters.csv")
print("  - results/figures/sensor_clustering_dendrogram.png")
print("  - results/figures/cluster_importance_heatmap.png")
print("  - results/figures/early_vs_late_detection.png")

print("\ndone! domain analysis complete")
