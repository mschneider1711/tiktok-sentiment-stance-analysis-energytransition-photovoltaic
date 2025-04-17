
"""
This script performs a topic-related analysis of TikTok critics who express a negative stance.
The input data must contain a 'voice_to_text' column with transcribed speech from TikTok videos
and a 'predicted_stance_label' column with stance labels. Only rows with 'negative' stance are analyzed.

The script includes:
- Noun extraction and lemmatization using spaCy (German model)
- Stopword removal (NLTK)
- Translation of nouns into English (GoogleTranslator API)
- Generation of a WordCloud
- Basic engagement statistics

IMPORTANT:
- Input Excel path and output path must be set accordingly.
- Translation uses the free GoogleTranslator API, which may have request limitations.
"""

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import os
import time
from deep_translator import GoogleTranslator
import random

# Load SpaCy German model
nlp = spacy.load("de_core_news_sm")

# Download and load German stopwords
nltk.download('stopwords')
german_stopwords = set(stopwords.words('german'))

# Define output directory for plots
output_dir = r"PATH/TO/OUTPUT/FOLDER"
os.makedirs(output_dir, exist_ok=True)

# Load the Excel file after sentiment and stance detection
excel_path = r"PATH/TO/DATASET.xlsx"
df = pd.read_excel(excel_path)

# Check if the file is loaded successfully
print(f"DataFrame loaded with {len(df)} rows and {df.shape[1]} columns.")

# Filter critics
critics_df = df[(df['predicted_stance_label'] == 'negative')]

# Function for extracting and normalizing nouns
def extract_and_normalize_nouns(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if token.pos_ == "NOUN" and token.text.lower() not in german_stopwords]
    return " ".join(lemmas)

# Extract and normalize nouns
print("Extracting and normalizing nouns...")
critics_df['nouns_only'] = critics_df['voice_to_text'].dropna().apply(extract_and_normalize_nouns)

# Translate extracted nouns to English
print("Translating words to English...")
all_nouns_text = " ".join(critics_df['nouns_only'].dropna())
unique_words = list(set(all_nouns_text.split()))

# Translate in batches
batch_size = 100
translated_words = {}
for i in range(0, len(unique_words), batch_size):
    batch = unique_words[i:i + batch_size]
    try:
        translations = {word: GoogleTranslator(source='de', target='en').translate(word) for word in batch}
        translated_words.update(translations)
    except Exception as e:
        print(f"Translation error in batch {i // batch_size + 1}: {e}")
    time.sleep(1)  # Avoid rate limits

# Combine translated words
translated_text = " ".join([
    translated_words.get(word, word) for word in all_nouns_text.split()
])

# WordCloud colors
colors = ["#C90C0F", "#EA5B0C", "#B82585"]  # Red, orange, purple

def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return random.choice(colors)

# Generate WordCloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Blues'
).generate(translated_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Save WordCloud
wordcloud_path = os.path.join(output_dir, "wordcloud_critics_nouns_translated.png")
wordcloud.to_file(wordcloud_path)
print(f"WordCloud saved to: {wordcloud_path}")

# Engagement metrics
engagement_columns = ['like_count', 'share_count', 'comment_count']
critic_engagement = critics_df[engagement_columns].mean()
print("\nAverage Engagement of Critics:")
print(critic_engagement)
