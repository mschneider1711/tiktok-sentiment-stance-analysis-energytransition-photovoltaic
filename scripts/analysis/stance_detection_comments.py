"""""
stance_detection_comments.py

This script processes TikTok comment JSON files and adds predicted sentiment and stance labels
for each comment. It uses the `GermanSentiment` model for sentiment prediction and a fine-tuned
BERT model from Hugging Face for stance classification.

Since TikTok comments are usually short, no sliding window or text slicing is applied during inference.

- Input: JSON comment files and an Excel file with parent post metadata including sentiment and stance scores
- Output: Updated JSON files with predicted sentiment and stance labels

Sentiment Model: https://huggingface.co/oliverguhr/german-sentiment-bert
Stance Model: https://huggingface.co/mrcschndr/bert-german-stance-detection-energytransition

Author: Marc Schneider [mschneider1711]
Date: 2025-04-17
"""

import os
import json
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from germansentiment import SentimentModel
import numpy as np

# Generic paths (replace with actual paths when running)
comment_path = "path/to/json/comment_files"
excel_path = "path/to/excel_file_with_sentiment_and_stance.xlsx"

# Load stance detection model and tokenizer from Hugging Face
print("Loading stance detection model from Hugging Face...")
tokenizer = BertTokenizer.from_pretrained("mrcschndr/bert-german-stance-detection-energytransition")
stance_model = BertForSequenceClassification.from_pretrained("mrcschndr/bert-german-stance-detection-energytransition")
stance_model.eval()

# Load sentiment model
sentiment_model = SentimentModel()

# Load Excel metadata
df_excel = pd.read_excel(excel_path)
df_excel = df_excel.set_index('id')

# JSON serialization helper
def convert_json_serializable(obj):
    if isinstance(obj, (np.int64, pd.Int64Dtype)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_json_serializable(i) for i in obj]
    return obj

# Stance prediction function (no slicing necessary for short comments)
def predict_stance(text, max_length=512):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        outputs = stance_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    return torch.argmax(outputs.logits, dim=1).item()

# Process all comment JSON files
print("Processing JSON files with sentiment and stance prediction...")
for filename in tqdm(os.listdir(comment_path), desc="Processing files"):
    if filename.endswith(".json"):
        file_path = os.path.join(comment_path, filename)
        parent_id = filename.replace(".json", "")

        # Load comment JSON
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                print(f"Warning: Empty file detected: {filename}")
                continue
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file {filename}: {e}")
                continue

        # Annotate each comment
        for comment in data:
            comment_id = comment.get("Comment_ID")
            parent_comment_id = str(comment.get("Parent_Comment_ID"))
            text = comment.get("Comment")

            predicted_stance = predict_stance(text)
            sentiment_result = sentiment_model.predict_sentiment([text])[0]
            sentiment_numeric = {'positive': 0, 'neutral': 1, 'negative': 2}.get(sentiment_result, 'unknown')

            parent_stance_label, parent_stance_numeric = "unknown", None
            parent_sentiment_label, parent_sentiment_numeric = "unknown", None

            if parent_comment_id == parent_id:
                if int(parent_id) in df_excel.index:
                    parent_stance_label = df_excel.loc[int(parent_id), 'predicted_stance']
                    parent_stance_numeric = {'positive': 0, 'neutral': 1, 'negative': 2}.get(parent_stance_label)
                    parent_sentiment_label = df_excel.loc[int(parent_id), 'predicted_sentiment']
                    parent_sentiment_numeric = {'positive': 0, 'neutral': 1, 'negative': 2}.get(parent_sentiment_label)
                else:
                    print(f"Warning: Parent ID {parent_id} not found in Excel.")
            else:
                parent_comment = next((c for c in data if str(c.get("Comment_ID")) == parent_comment_id), None)
                if parent_comment:
                    parent_stance_numeric = parent_comment.get("Predicted_Stance", None)
                    parent_stance_label = parent_comment.get("Predicted_Stance_Label", "unknown")
                    parent_sentiment_numeric = parent_comment.get("Sentiment_Numeric", None)
                    parent_sentiment_label = parent_comment.get("Sentiment", "unknown")
                else:
                    print(f"Warning: Parent Comment ID {parent_comment_id} not found in JSON {filename}.")

            comment['Predicted_Stance'] = predicted_stance
            comment['Predicted_Stance_Label'] = {0: 'positive', 1: 'neutral', 2: 'negative'}.get(predicted_stance, 'unknown')
            comment['Parent_Stance'] = parent_stance_numeric
            comment['Parent_Stance_Label'] = parent_stance_label
            comment['Sentiment'] = sentiment_result
            comment['Sentiment_Numeric'] = sentiment_numeric
            comment['Parent_Sentiment'] = parent_sentiment_numeric
            comment['Parent_Sentiment_Label'] = parent_sentiment_label

        # Save updated JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            try:
                json.dump(convert_json_serializable(data), f, ensure_ascii=False, indent=4)
            except TypeError as e:
                print(f"JSON serialization error in {filename}: {e}")
                continue

print("All comment JSON files have been successfully updated.")