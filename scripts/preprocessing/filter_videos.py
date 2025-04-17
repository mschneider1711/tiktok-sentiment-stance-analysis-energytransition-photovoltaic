"""
This script is used to filter TikTok videos based on their relevance to specific energy-related topics.

Requirements:
- Manually labeled data for training: Add labels in the 'manual_rating' column of the Excel training file.
  Use the classes: 'relevant' and 'irrelevant'.
- A list of domain-specific keywords must be defined by the user to reflect the relevant context (e.g., energy, climate, solar, etc.).

The model combines TF-IDF text features and keyword matching to train an ensemble classifier
(Logistic Regression, Random Forest, and SVM) for predicting relevance.

It will generate:
- A prediction file for the entire test dataset.
- A filtered file with only relevant videos.
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

# === Paths (adjust as needed) ===
train_dataset_path = "path/to/your/training_data.xlsx"
test_dataset_path = "path/to/your/test_data.xlsx"

# === User-defined keywords that are considered relevant ===
keywords = ["energy", "climate", "heating law", "wind energy", "coal", "electricity", "photovoltaics", "solar energy"]

# === Helper Function: Keyword Matching ===
def keyword_match(text, keywords):
    text = text.lower()
    for keyword in keywords:
        if re.search(rf'\b{re.escape(keyword)}', text):
            return 1
    return 0

# === Load and preprocess training data ===
df_train = pd.read_excel(train_dataset_path)
df_train = df_train[df_train['voice_to_text'].apply(lambda x: isinstance(x, str))]
df_train['keyword_match'] = df_train['voice_to_text'].apply(lambda x: keyword_match(x, keywords))
df_train = df_train[['voice_to_text', 'keyword_match', 'manual_rating']].dropna()

# Map labels to binary
label_mapping = {'irrelevant': 0, 'relevant': 1}
df_train['manual_rating'] = df_train['manual_rating'].map(label_mapping)
df_train = df_train.dropna(subset=['manual_rating'])

# Features and Labels
train_texts = df_train['voice_to_text']
train_labels = df_train['manual_rating']

# TF-IDF Features
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
X_train = vectorizer.fit_transform(train_texts)
train_keyword_match = np.array(df_train['keyword_match']).reshape(-1, 1)
X_train_combined = hstack([X_train, train_keyword_match])

# Ensemble Classifier
logistic_model = LogisticRegression(class_weight='balanced', C=10, penalty='l1', solver='liblinear', max_iter=1000, random_state=42)
random_forest_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=10, random_state=42)
svm_model = SVC(class_weight='balanced', C=10, kernel='linear', gamma='scale', probability=True, random_state=42)

ensemble_model = VotingClassifier(
    estimators=[
        ('Logistic Regression', logistic_model),
        ('Random Forest', random_forest_model),
        ('SVM', svm_model)
    ],
    voting='hard'
)

# Train the model
ensemble_model.fit(X_train_combined, train_labels)

# === Load and preprocess test data ===
df_test = pd.read_excel(test_dataset_path)
df_test['voice_to_text'] = df_test['voice_to_text'].astype(str)
test_keyword_match = np.array(df_test['voice_to_text'].apply(lambda x: keyword_match(x, keywords))).reshape(-1, 1)
X_test = vectorizer.transform(df_test['voice_to_text'])
X_test_combined = hstack([X_test, test_keyword_match])

# === Prediction ===
df_test['predicted_class'] = ensemble_model.predict(X_test_combined)
class_mapping = {0: 'irrelevant', 1: 'relevant'}
df_test['predicted_class'] = df_test['predicted_class'].map(class_mapping)
df_test["id"] = df_test["id"].astype(str)
df_test["create_time"] = df_test["create_time"].astype(str)

# === Save Outputs ===
output_path = os.path.join(os.path.dirname(test_dataset_path), "predicted_dataset.xlsx")
df_test.to_excel(output_path, index=False)

# Save only relevant videos
df_relevant = df_test[df_test['predicted_class'] == 'relevant']
relevant_output_path = os.path.join(os.path.dirname(test_dataset_path), "filtered_relevant_dataset.xlsx")
df_relevant.to_excel(relevant_output_path, index=False)

# === Summary ===
print("=" * 50)
print(f"Predictions saved to: {output_path}")
print(f"Relevant videos saved to: {relevant_output_path}")
print("=" * 50)
