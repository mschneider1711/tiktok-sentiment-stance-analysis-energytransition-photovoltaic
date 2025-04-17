"""
This script enriches a TikTok video metadata Excel file with comment-level sentiment and stance statistics.

It loads video metadata from an Excel file and corresponding comment-level JSON files (named by video ID).
Each JSON contains predicted sentiment and stance labels for each comment.
The script counts how many comments per video fall into each sentiment/stance category and appends those stats
as new columns in the original Excel sheet.

Expected structure of each comment in the JSON:
{
    "comment_text": "...",
    "Predicted_Stance_Label": "positive" / "neutral" / "negative",
    "Sentiment": "positive" / "neutral" / "negative"
}

Make sure the sentiment and stance classification has been applied to each comment before running this script.
"""

import os
import pandas as pd
import json
from tqdm import tqdm

# File paths (replace with your own paths)
excel_path = r"path_to_your_excel\TikTok_dataset_sentiment_and_stance_analysis.xlsx"  # Excel file with metadata
comment_dir = r"path_to_comment_jsons\filtered_comments"  # Folder containing comment JSONs by video ID

# Load the Excel file
df = pd.read_excel(excel_path)

# Initialize new columns for sentiment and stance counts
df["negative_stance_count"] = 0
df["neutral_stance_count"] = 0
df["positive_stance_count"] = 0
df["negative_sentiment_count"] = 0
df["neutral_sentiment_count"] = 0
df["positive_sentiment_count"] = 0

# Loop through each video ID in the dataset
for video_id in tqdm(df["id"], desc="Processing JSON files"):
    json_file = os.path.join(comment_dir, f"{video_id}.json")

    if not os.path.exists(json_file):
        print(f"Warning: JSON file for video ID {video_id} not found.")
        continue

    # Load comment data from JSON
    with open(json_file, "r", encoding="utf-8") as file:
        comments = json.load(file)

    # Initialize counters
    stance_counts = {"negative": 0, "neutral": 0, "positive": 0}
    sentiment_counts = {"negative": 0, "neutral": 0, "positive": 0}

    # Count stance and sentiment values
    for comment in comments:
        stance = comment.get("Predicted_Stance_Label", "unknown")
        if stance in stance_counts:
            stance_counts[stance] += 1

        sentiment = comment.get("Sentiment", "unknown")
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1

    # Assign counts back to the main dataframe
    df.loc[df["id"] == video_id, "negative_stance_count"] = stance_counts["negative"]
    df.loc[df["id"] == video_id, "neutral_stance_count"] = stance_counts["neutral"]
    df.loc[df["id"] == video_id, "positive_stance_count"] = stance_counts["positive"]
    df.loc[df["id"] == video_id, "negative_sentiment_count"] = sentiment_counts["negative"]
    df.loc[df["id"] == video_id, "neutral_sentiment_count"] = sentiment_counts["neutral"]
    df.loc[df["id"] == video_id, "positive_sentiment_count"] = sentiment_counts["positive"]

# Ensure 'id' and 'create_time' are strings
df['id'] = df['id'].astype(str)
if 'create_time' in df.columns:
    df['create_time'] = df['create_time'].astype(str)

# Save the enriched Excel file
df.to_excel(excel_path, index=False)
print(f"Analysis completed. Results saved to: {excel_path}")
