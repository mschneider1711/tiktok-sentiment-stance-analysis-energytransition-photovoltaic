"""
fetch_video_comments.py

This script retrieves comments for TikTok videos using the Research API and saves each video’s
comments as a separate JSON file. The video IDs are read from an Excel file.

Before running, create a `.env` file with:
    TIKTOK_CLIENT_KEY=your_client_key
    TIKTOK_CLIENT_SECRET=your_client_secret
    
Author: Marc Schneider [mschneider1711]
Date: 2025-04-17
"""

import requests
import os
import json
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load API credentials from .env
load_dotenv()
CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY")
CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET")

def get_access_token(client_key, client_secret):
    url = "https://open.tiktokapis.com/v2/oauth/token/"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'client_key': client_key,
        'client_secret': client_secret,
        'grant_type': 'client_credentials',
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        print("Failed to get access token:", response.status_code, response.text)
        return None

def fetch_comments(access_token, video_id):
    url = "https://open.tiktokapis.com/v2/research/video/comment/list/"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    query_params = {
        "fields": "id,text,like_count,create_time,video_id,parent_comment_id",
    }
    all_comments = []
    cursor = 0
    retries = 5
    wait_time = 1

    while True:
        payload = {
            "video_id": video_id,
            "cursor": cursor,
            "max_count": 50
        }
        response = requests.post(url, headers=headers, json=payload, params=query_params)
        if response.status_code == 200:
            data = response.json().get('data', {})
            comments = data.get('comments', [])
            all_comments.extend(comments)
            if not data.get('has_more', False):
                break
            cursor = data.get('cursor', 0)
        else:
            print(f"Error fetching comments: {response.status_code} {response.text}")
            if retries > 0:
                time.sleep(wait_time)
                wait_time *= 2
                retries -= 1
                continue
            else:
                break

    return all_comments

def main():
    access_token = get_access_token(CLIENT_KEY, CLIENT_SECRET)
    if not access_token:
        print("Access token could not be retrieved. Exiting.")
        return

    # Define the input Excel path
    excel_path = "path/to/your/excel_file.xlsx"  # Replace this with your actual path
    df = pd.read_excel(excel_path)
    video_ids = df['id'].astype(str).tolist()

    output_dir = "tiktok_comments"
    os.makedirs(output_dir, exist_ok=True)

    for video_id in tqdm(video_ids, desc="Fetching comments"):
        output_file = os.path.join(output_dir, f"{video_id}.json")
        if os.path.exists(output_file):
            print(f"Skipping {video_id}, file already exists.")
            continue

        comments = fetch_comments(access_token, video_id)
        comments_data = [
            {
                "ID": video_id,
                "Comment_ID": comment['id'],
                "Parent_Comment_ID": comment['parent_comment_id'],
                "Create_Time": comment['create_time'],
                "Comment": comment['text'],
                "Like_Count": comment['like_count']
            }
            for comment in comments
        ]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comments_data, f, ensure_ascii=False, indent=4)

        print(f"Saved comments for video ID {video_id} to {output_file}")

if __name__ == "__main__":
    main()
