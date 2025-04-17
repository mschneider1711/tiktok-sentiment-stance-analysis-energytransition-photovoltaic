"""
sentiment_analysis.py

This script performs sentiment analysis on TikTok video transcriptions using the
pretrained `GermanSentiment` model by the GermanSentiment library.

- Input: Excel file with 'preprocessed_voice_to_text' column
- Output: Excel file with additional column 'predicted_sentiment'

Supports long texts by slicing into overlapping windows and aggregating results
via majority voting.

Model: https://huggingface.co/oliverguhr/german-sentiment-bert
"""

import pandas as pd
import os
from tqdm import tqdm
from germansentiment import SentimentModel

# Load pretrained German sentiment model
model = SentimentModel()

def slice_text(text, max_length=350, stride=175):
    words = text.split()
    slices = []
    for i in range(0, len(words), stride):
        slice_ = " ".join(words[i:i + max_length])
        slices.append(slice_)
        if i + max_length >= len(words):
            break
    return slices

def predict_sentiment_with_slicing(text, max_length=350):
    text_slices = slice_text(text, max_length=max_length, stride=int(max_length / 2))
    predictions = [model.predict_sentiment([s])[0] for s in text_slices]
    return max(set(predictions), key=predictions.count)

def main():
    # Define path to input file
    dataset_path = "path/to/your/input_file.xlsx"  # Replace with actual file
    df = pd.read_excel(dataset_path)

    # Filter out rows with invalid text
    df = df[df['preprocessed_voice_to_text'].apply(lambda x: isinstance(x, str))]

    # Set optimal slicing length (based on model limits or prior testing)
    best_max_length = 300

    # Apply sentiment analysis
    tqdm.pandas(desc="Predicting sentiment")
    df['predicted_sentiment'] = df['preprocessed_voice_to_text'].progress_apply(
        lambda x: predict_sentiment_with_slicing(x, best_max_length)
    )

    # Ensure ID and create_time are strings
    df["id"] = df["id"].astype(str)
    df["create_time"] = df["create_time"].astype(str)

    # Save results
    output_path = os.path.join(
        os.path.dirname(dataset_path),
        "TikTok_dataset_sentiment_analysis.xlsx"
    )
    df.to_excel(output_path, index=False)
    print(f"Sentiment analysis completed. Results saved to: {output_path}")

if __name__ == "__main__":
    main()
