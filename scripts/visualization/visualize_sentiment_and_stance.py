import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np

# Define output directory for plots
output_dir = r"C:\Users\marc\OneDrive - TH Köln\Forschungsseminar\PV_Congress_Plots"
os.makedirs(output_dir, exist_ok=True)

# Load the unified DataFrame
excel_path = r"C:\Users\marc\OneDrive - TH Köln\Forschungsseminar\PV_CONGRESS_FINAL_NO_DUPLICATES.xlsx"
df = pd.read_excel(excel_path)

# Check if the DataFrame is loaded successfully
print(f"DataFrame loaded with {len(df)} rows.")

# Map incorrect stance labels to correct ones
stance_mapping = {'positive': 'favour', 'negative': 'against', 'neutral': 'neutral'}
df['predicted_stance_label'] = df['predicted_stance_label'].map(stance_mapping)

# --- Sentiment Analysis ---
sentiment_distribution = df['predicted_sentiment'].value_counts(normalize=True)
stance_distribution = df['predicted_stance_label'].value_counts(normalize=True)

# --- Process JSON Comments ---
json_dir = r"C:\Users\marc\OneDrive - TH Köln\Forschungsseminar\Photovoltaik_Kommentare"
comment_data = []

# Read JSON files
for file in os.listdir(json_dir):
    if file.endswith(".json"):
        with open(os.path.join(json_dir, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            comment_data.extend(data)

# Convert JSON data to DataFrame
comments_df = pd.DataFrame(comment_data)

# Ensure required columns are present
required_columns = ['Comment', 'Sentiment', 'Predicted_Stance_Label']
if not all(col in comments_df.columns for col in required_columns):
    raise ValueError("JSON data is missing required columns!")

# Map incorrect stance labels in comments
comments_df['Predicted_Stance_Label'] = comments_df['Predicted_Stance_Label'].map(stance_mapping)

# Sentiment & Stance Distributions from Comments
comment_sentiment_dist = comments_df['Sentiment'].value_counts(normalize=True)
comment_stance_dist = comments_df['Predicted_Stance_Label'].value_counts(normalize=True)

# --- Sort the categories ---
stance_order = ['favour', 'neutral', 'against']
sentiment_order = ['positive', 'neutral', 'negative']

# Reindex to ensure correct order
stance_distribution = stance_distribution.reindex(stance_order, fill_value=0)
comment_stance_dist = comment_stance_dist.reindex(stance_order, fill_value=0)
sentiment_distribution = sentiment_distribution.reindex(sentiment_order, fill_value=0)
comment_sentiment_dist = comment_sentiment_dist.reindex(sentiment_order, fill_value=0)

# Define colours
video_color = "#003366"   # Sehr dunkles Blau (Navy)
comment_color = "#a3c9f1"  # Sehr helles Blau (Pastellblau)

# --- Plot Stance Distribution (Side-by-Side) ---
x = np.arange(len(stance_order))
bar_width = 0.35

plt.figure(figsize=(8, 6))
plt.bar(x - bar_width / 2, stance_distribution.values, width=bar_width, label='Videos', alpha=0.7, color=video_color)
plt.bar(x + bar_width / 2, comment_stance_dist.values, width=bar_width, label='Comments', alpha=0.7, color=comment_color)

plt.ylabel('Percentage', fontsize=14)
plt.xlabel('Stance', fontsize=14)
plt.title('Stance Distribution', fontsize=16)
plt.xticks(x, stance_order, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "stance_distribution_side_by_side.png"))
plt.close()

# --- Plot Sentiment Distribution (Side-by-Side) ---
x_sentiment = np.arange(len(sentiment_order))

plt.figure(figsize=(8, 6))
plt.bar(x_sentiment - bar_width / 2, sentiment_distribution.values, width=bar_width, label='Videos', alpha=0.7, color=video_color)
plt.bar(x_sentiment + bar_width / 2, comment_sentiment_dist.values, width=bar_width, label='Comments', alpha=0.7, color=comment_color)

plt.ylabel('Percentage', fontsize=14)
plt.xlabel('Sentiment', fontsize=14)
plt.title('Sentiment Distribution', fontsize=16)
plt.xticks(x_sentiment, sentiment_order, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sentiment_distribution_side_by_side.png"))
plt.close()

print(f"Plots have been updated and saved in the directory: {output_dir}")

# --- Calculate and Print Combinations of Sentiment and Stance ---
# First, create a crosstab to see how many combinations of Sentiment and Stance we have in the videos DataFrame
combination_crosstab = pd.crosstab(df['predicted_sentiment'], df['predicted_stance_label'])

# Total number of videos for percentage calculation
total_videos = len(df)

# --- Print values used in bar plots (Stance + Sentiment) ---

print("\n--- Stance Distribution (Videos) ---")
for label in stance_order:
    abs_count = df['predicted_stance_label'].value_counts().get(label, 0)
    rel_pct = stance_distribution[label] * 100
    print(f"{label.capitalize()}: {abs_count} ({rel_pct:.2f}%)")

print("\n--- Stance Distribution (Comments) ---")
for label in stance_order:
    abs_count = comments_df['Predicted_Stance_Label'].value_counts().get(label, 0)
    rel_pct = comment_stance_dist[label] * 100
    print(f"{label.capitalize()}: {abs_count} ({rel_pct:.2f}%)")

print("\n--- Sentiment Distribution (Videos) ---")
for label in sentiment_order:
    abs_count = df['predicted_sentiment'].value_counts().get(label, 0)
    rel_pct = sentiment_distribution[label] * 100
    print(f"{label.capitalize()}: {abs_count} ({rel_pct:.2f}%)")

print("\n--- Sentiment Distribution (Comments) ---")
for label in sentiment_order:
    abs_count = comments_df['Sentiment'].value_counts().get(label, 0)
    rel_pct = comment_sentiment_dist[label] * 100
    print(f"{label.capitalize()}: {abs_count} ({rel_pct:.2f}%)")


# Now calculate specific combinations
# Videos with neutral sentiment and positive stance
neutral_positive = combination_crosstab.loc['neutral', 'favour']
neutral_positive_pct = (neutral_positive / total_videos) * 100

# Videos with negative sentiment and positive stance
negative_positive = combination_crosstab.loc['negative', 'favour']
negative_positive_pct = (negative_positive / total_videos) * 100

# Videos with different sentiment and stance
diff_sentiment_stance = len(df[df['predicted_sentiment'] != df['predicted_stance_label']])
diff_sentiment_stance_pct = (diff_sentiment_stance / total_videos) * 100

# Print the results
print("\nSentiment and Stance Combinations in Videos:")
print(f"Videos with neutral sentiment and positive stance: {neutral_positive} ({neutral_positive_pct:.2f}%)")
print(f"Videos with negative sentiment and positive stance: {negative_positive} ({negative_positive_pct:.2f}%)")
print(f"Videos with different sentiment and stance: {diff_sentiment_stance} ({diff_sentiment_stance_pct:.2f}%)")
