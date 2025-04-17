"""
stance_detection.py

This script performs stance detection on TikTok video transcriptions using a
pretrained BERT model for stance detection.

- Input: Excel file with 'preprocessed_voice_to_text' column (TikTok transcriptions)
- Output: Excel file with additional columns 'predicted_stance' and 'predicted_stance_label'

The sliding window technique is used to handle long texts by slicing them into smaller, overlapping windows,
predicting stance for each window, and aggregating the results via majority voting.

Model: https://huggingface.co/mrcschndr/bert-german-stance-detection-energytransition

Author: Marc Schneider [mschneider1711]
Date: 2025-04-17
"""

import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

# Load input file with preprocessed 'preprocessed_voice_to_text' column
input_excel_path = "path/to/your/TikTok_dataset_sentiment_analysis.xlsx"  # Replace with actual file path
df = pd.read_excel(input_excel_path)

# Load pretrained model from Hugging Face
model_name = "mrcschndr/bert-german-stance-detection-energytransition"
print(f"Loading pretrained model from: {model_name}")
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()

# Sliding window technique for stance prediction
def slice_text_for_stance(text, max_length=256, stride=128):
    tokens = tokenizer(text, truncation=False, padding=False, return_tensors="pt")['input_ids'][0]
    slices = []
    for i in range(0, len(tokens), stride):
        slice_ = tokens[i:i + max_length]
        if len(slice_) == 0:
            continue
        
        input_text = tokenizer.decode(slice_)
        slices.append(input_text)
        
        if i + max_length >= len(tokens):
            break
    return slices

def predict_stance_with_slicing(text, max_length=350):
    text_slices = slice_text_for_stance(text, max_length=max_length, stride=int(max_length / 2))
    predictions = [predict_stance_with_slicing_window(s) for s in text_slices]
    return max(set(predictions), key=predictions.count)

def predict_stance_with_slicing_window(text):
    input_tokens = tokenizer(text, truncation=True, padding=True, max_length=350, return_tensors="pt")
    
    # Model inference
    with torch.no_grad():
        output = model(input_ids=input_tokens['input_ids'], attention_mask=input_tokens['attention_mask'])
    
    # Return the predicted stance label (0, 1, 2)
    return torch.argmax(output.logits, dim=1).item()

# Classify the entire dataset using the preprocessed 'voice_to_text' column
print("\nStarting stance classification on the dataset...")
df['predicted_stance'] = [predict_stance_with_slicing(text) for text in tqdm(df['preprocessed_voice_to_text'], desc="Classifying dataset")]

# Map stance predictions to labels
df['predicted_stance_label'] = df['predicted_stance'].map({0: 'positive', 1: 'neutral', 2: 'negative'})

# Ensure ID and time are stored as strings
df['id'] = df['id'].astype(str)
df['create_time'] = df['create_time'].astype(str)

# Save the predictions to a new Excel file
output_excel_path = "path/to/your/output/TikTok_dataset_sentiment_and_stance_analysis.xlsx"  # Replace with actual file path
df.to_excel(output_excel_path, index=False)
print(f"Predictions saved to: {output_excel_path}")
