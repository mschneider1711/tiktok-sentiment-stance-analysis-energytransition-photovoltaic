# TikTok Sentiment and Stance Analysis

This project analyzes TikTok videos related to energytransition and photovoltaics, specifically focusing on stance and sentiment analysis of comments and videos.

![image](https://github.com/user-attachments/assets/e6734aee-57e5-40ad-93db-486b8c3cefa3)

## How to use

1. Clone the repository.
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the scripts in the following order:

### Data Collection
1. `fetch_tiktok_videos.py`
2. `fetch_tiktok_comments.py`

### Preprocessing
1. `preprocess_texts.py`
2. `filter_videos.py`

### Analysis
1. `sentiment_analysis.py`
2. `stance_detection.py`
3. `stance_detection_comments.py`
4. `extract_comments_stats.py`
5. `analyse_critics.py`

### Visualization
1. `visualize_sentiment_and_stance.py`
2. `plot_engagement_over_stance.py`
