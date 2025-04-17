# TikTok Sentiment and Stance Analysis

This project analyzes TikTok videos related to energy topics, specifically focusing on stance and sentiment analysis of comments and videos.

## Directory Structure

- **data/**
  - raw video and comment data
- **output/**
  - plots and statistics
- **scripts/**
  - **data_collection/**
    - Scripts to fetch TikTok data
  - **preprocessing/**
    - Data preprocessing scripts
  - **analysis/**
    - Sentiment and stance analysis scripts
  - **visualization/**
    - Scripts for data visualization

## How to use

1. Clone the repository.
2. Install the dependencies: `pip install -r requirements.txt`
3. Run the scripts in the following order:
   1. fetch_tiktok_videos.py
   2. fetch_tiktok_comments.py
   3. preprocess_texts.py
   4. filter_videos.py
   5. sentiment_analysis.py
   6. stance_detection.py
   7. stance_detection_comments.py
   8. extract_comments_stats.py
   9. analyse_critics.py
   10. visualize_sentiment_and_stance.py
   11. plot_engagement_over_stance.py

