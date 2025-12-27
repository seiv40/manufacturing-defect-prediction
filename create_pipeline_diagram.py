"""
Create visual pipeline diagram for the project
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# colors
color_data = '#E3F2FD'
color_process = '#FFF3E0'
color_model = '#E8F5E9'
color_output = '#FCE4EC'

def add_box(ax, x, y, width, height, text, color, fontsize=10):
    """Add a rounded box with text"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=color,
        linewidth=2
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            wrap=True)

def add_arrow(ax, x1, y1, x2, y2):
    """Add arrow between boxes"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.8',
        color='black',
        linewidth=2,
        zorder=1
    )
    ax.add_patch(arrow)

# title
ax.text(5, 13.5, 'Semiconductor Failure Prediction Pipeline',
        ha='center', fontsize=18, fontweight='bold')
ax.text(5, 13, 'Cost-Optimized Manufacturing Quality Control',
        ha='center', fontsize=12, style='italic')

# row 1: data input
add_box(ax, 0.5, 11, 2, 1, 'Raw Data\n1567 samples\n590 sensors\n14:1 imbalance', color_data)
add_arrow(ax, 2.5, 11.5, 3.5, 11.5)

# row 1: EDA
add_box(ax, 3.5, 11, 3, 1, 'Exploratory Analysis\n• Missing data: 4.5%\n• 116 constant sensors\n• Weak correlations (<0.16)', color_process)
add_arrow(ax, 6.5, 11.5, 7.5, 11.5)

# row 1: insights
add_box(ax, 7.5, 11, 2, 1, 'Key Findings\n• Class imbalance critical\n• High dimensionality\n• Complex relationships', color_output, fontsize=9)

# row 2: feature engineering
add_arrow(ax, 2.5, 11, 2.5, 10)
add_box(ax, 0.5, 9, 4, 1, 'Feature Engineering\n• Remove 116 constant + 28 high-missing sensors\n• Create missing data indicators (52 features)\n• Remove redundant correlations (>0.95)', color_process, fontsize=9)
add_arrow(ax, 4.5, 9.5, 5.5, 9.5)

# row 2: output
add_box(ax, 5.5, 9, 2, 1, 'Reduced Features\n590 -> 127 features\n(75 sensors +\n52 missing flags)', color_output, fontsize=9)

# row 3: feature selection
add_arrow(ax, 2.5, 9, 2.5, 8)
add_box(ax, 0.5, 7, 4, 1, 'Feature Selection\n• Mutual Information ranking\n• Random Forest importance\n• Correlation with target\n• Composite scoring', color_process, fontsize=9)
add_arrow(ax, 4.5, 7.5, 5.5, 7.5)

# row 3: output
add_box(ax, 5.5, 7, 2, 1, 'Final Features\n127 predictive\nfeatures selected\n(validated by CV)', color_output, fontsize=9)

# row 4: modeling approaches
add_arrow(ax, 2.5, 7, 2.5, 6)
add_box(ax, 0.25, 5, 2.25, 1, 'Random Forest\n+ SMOTE\nClass weights\nbalanced', color_model, fontsize=9)

add_box(ax, 2.75, 5, 2.25, 1, 'XGBoost\n+ SMOTE\nScale pos weight\n= 14.06', color_model, fontsize=9)

add_box(ax, 5.25, 5, 2.25, 1, 'Logistic Reg.\nClass weights\nL2 regularization', color_model, fontsize=9)

add_box(ax, 7.75, 5, 1.75, 1, 'Ensemble\nVoting\nStacking', color_model, fontsize=9)

# row 5: cross-validation
add_arrow(ax, 1.4, 5, 1.4, 4)
add_arrow(ax, 3.9, 5, 3.9, 4)
add_arrow(ax, 6.4, 5, 6.4, 4)

add_box(ax, 0.5, 3, 7, 1, '5-Fold Stratified Cross-Validation\nAUC: 0.736 +/- 0.041 | Feature importance stability analysis',
        color_process, fontsize=10)
add_arrow(ax, 4, 3, 4, 2)

# row 6: threshold optimization
add_box(ax, 1, 1, 7.5, 1, 'Cost-Based Threshold Optimization\nSweep cost ratios: 1:1, 3:1, 5:1, 10:1, 20:1, 50:1',
        color_process, fontsize=10)
add_arrow(ax, 4.75, 1, 4.75, 0)

# row 7: final outputs (3 columns)
add_box(ax, 0.25, -1.5, 3, 1.2, 'Conservative\nRatio 5:1\n46% recall\n11% inspect', color_output, fontsize=9)

add_box(ax, 3.5, -1.5, 3, 1.2, 'Balanced\nRatio 10:1\n69% recall\n25% inspect', color_output, fontsize=9)

add_box(ax, 6.75, -1.5, 3, 1.2, 'Aggressive\nRatio 50:1\n92% recall\n44% inspect', color_output, fontsize=9)

# row 8: business decision
add_arrow(ax, 1.75, -1.5, 1.75, -2.5)
add_arrow(ax, 5, -1.5, 5, -2.5)
add_arrow(ax, 8.25, -1.5, 8.25, -2.5)

add_box(ax, 0.5, -3.5, 9, 1, 'Business Decision: Choose Operating Point Based On\nInspection capacity | Cost ratio | Quality targets | Throughput requirements',
        color_output, fontsize=10)

# legend
legend_elements = [
    mpatches.Patch(color=color_data, label='Data'),
    mpatches.Patch(color=color_process, label='Processing'),
    mpatches.Patch(color=color_model, label='Modeling'),
    mpatches.Patch(color=color_output, label='Results')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig('results/figures/pipeline_diagram.png', dpi=300, bbox_inches='tight')
print("Pipeline diagram saved!")
plt.close()

print("\nPipeline Summary:")
print("1. Raw Data: 1567 samples, 590 sensors, severe imbalance (14:1)")
print("2. EDA: Identify quality issues, missing data patterns, correlations")
print("3. Feature Engineering: Add missing indicators, remove redundant/constant")
print("4. Feature Selection: Reduce 590 -> 127 using multiple methods")
print("5. Modeling: Test RF, XGBoost, LR with SMOTE and class weights")
print("6. Cross-Validation: Validate stability (5-fold stratified)")
print("7. Threshold Optimization: Find cost-optimal operating points")
print("8. Business Decision: Choose based on operational constraints")
