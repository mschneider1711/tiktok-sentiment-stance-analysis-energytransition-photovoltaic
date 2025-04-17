"""
preprocess_texts.py

This script cleans and preprocesses German text data from TikTok's 'voice_to_text' field
using SpaCy and NLTK. It performs:
- Lowercasing and umlaut replacement
- Special character and number removal
- Stopword filtering (German)
- Lemmatization using SpaCy

Input: Excel file with a column 'voice_to_text'
Output: Excel file with additional column 'preprocessed_voice_to_text'

Author: Marc Schneider [mschneider1711]
Date: 2025-04-17
"""

import pandas as pd
import re
import spacy
import nltk
import os
from nltk.corpus import stopwords

# Download stopwords once (if not already done)
nltk.download('stopwords')

# Load German language tools
nlp = spacy.load("de_core_news_sm")
german_stopwords = set(stopwords.words('german'))

def replace_umlauts(text):
    umlaut_map = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss'}
    for umlaut, replacement in umlaut_map.items():
        text = text.replace(umlaut, replacement)
    return text.lower()

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    
    text = replace_umlauts(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-letters

    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc 
        if token.text not in german_stopwords and not token.is_punct and token.is_alpha
    ]
    return ' '.join(tokens).lower()

def main():
    # Define path to input dataset
    dataset_path = "path/to/your/excel_file.xlsx" # Replace with actual path
    df = pd.read_excel(dataset_path)

    # Remove rows with known unusable transcriptions
    filter_values = [
        "could not extract text from audio track",
        "video has no audio track",
        "could not detect text",
        "music",
        "background music"
    ]
    df = df[~df['voice_to_text'].str.lower().isin([v.lower() for v in filter_values])]

    # Apply preprocessing
    df['preprocessed_voice_to_text'] = df['voice_to_text'].apply(preprocess_text)

    # Ensure 'id' and 'create_time' are strings
    df["id"] = df["id"].astype(str)
    df["create_time"] = df["create_time"].astype(str)

    # Save the cleaned output
    output_path = os.path.join(
        os.path.dirname(dataset_path),
        "tik_tok_dataset_solarenergy_preprocessed.xlsx"
    )
    df.to_excel(output_path, index=False)
    print(f"Preprocessing complete. File saved to: {output_path}")

if __name__ == "__main__":
    main()
