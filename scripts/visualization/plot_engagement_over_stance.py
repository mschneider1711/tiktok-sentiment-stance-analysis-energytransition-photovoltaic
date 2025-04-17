"""
plot_engagement_over_stance.py

This script visualizes average or median engagement statistics (views, likes, comments, shares) 
grouped by predicted stance classes in a TikTok dataset related to energy topics. The analysis 
helps understand how engagement metrics differ across videos with various stances (e.g., positive, negative, neutral).

The script includes:
- Loading the dataset
- Selecting relevant columns
- Aggregating engagement statistics (mean or median)
- Visualizing the statistics in a horizontal bar chart
- Saving the plot and the aggregated statistics as a CSV

IMPORTANT:
- Input data file path must be set accordingly.
- Ensure that the dataset contains the columns: 'predicted_stance', 'view_count', 'like_count', 'comment_count', 'share_count'.
- Choose between 'mean' or 'median' for aggregation.

Author: Marc Schneider [mschneider1711]
Date: 2025-04-17
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# === Configuration ===

# Relative input file path (adjust if needed)
input_file = os.path.join("data", "PV_CONGRESS_DATASET.xlsx")

# Output paths
output_plot_path = os.path.join("output", "engagement_by_stance.png")
output_csv_path = os.path.join("output", "engagement_by_stance.csv")

# Statistic to use: 'mean' or 'median'
aggregation_method = 'mean'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

# === Load data ===
df = pd.read_excel(input_file)

# === Select relevant columns ===
columns_of_interest = ['predicted_stance', 'view_count', 'like_count', 'comment_count', 'share_count']
df_filtered = df[columns_of_interest].dropna()

# === Aggregate data ===
if aggregation_method == 'mean':
    engagement_stats = df_filtered.groupby('predicted_stance').mean()
elif aggregation_method == 'median':
    engagement_stats = df_filtered.groupby('predicted_stance').median()
else:
    raise ValueError("aggregation_method must be either 'mean' or 'median'")

# === Plot ===
plt.figure(figsize=(10, 5))
engagement_stats.plot(kind='barh', ax=plt.gca(), colormap='viridis')
plt.title(f"{aggregation_method.capitalize()} Engagement per Predicted Stance")
plt.xlabel(f"{aggregation_method.capitalize()} Engagement")
plt.ylabel("Predicted Stance")
plt.legend(title='Metric', loc='lower right')
plt.tight_layout()

# === Save outputs ===
plt.savefig(output_plot_path, dpi=300)
plt.show()

engagement_stats.to_csv(output_csv_path)

print(f"Plot saved to: {output_plot_path}")
print(f"CSV saved to: {output_csv_path}")
