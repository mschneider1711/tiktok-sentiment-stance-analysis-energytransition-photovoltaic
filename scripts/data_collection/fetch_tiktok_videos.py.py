"""
fetch_tiktok_videos.py

This script uses the TikTok Research API to fetch videos based on hashtag search queries
over a defined date range. Results are exported to an Excel file.

Before running, create a `.env` file with the following content:
    TIKTOK_CLIENT_KEY=your_client_key
    TIKTOK_CLIENT_SECRET=your_client_secret
    
Author: Marc Schneider [mschneider1711]
Date: 2025-04-17
"""

import requests
import pandas as pd
import os
import time
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv

# Load API credentials from .env
load_dotenv()
CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY")
CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def exponential_backoff(retry_attempts):
    return min(60, 5 * (2 ** retry_attempts))

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
        logging.error("Failed to retrieve access token: %s", response.json())
        return None

def fetch_videos(session, headers, base_url, payload, query_params, delay=1):
    results = []
    retry_count = 0
    max_retries = 1
    page_count = 0

    while True:
        try:
            response = session.post(base_url, headers=headers, json=payload, params=query_params)
            if response.status_code == 200:
                data = response.json().get('data', {})
                videos = data.get('videos', [])
                results.extend(videos)
                page_count += 1
                logging.info(f"Fetched page {page_count}, total videos: {len(results)}")
                retry_count = 0
            else:
                raise requests.HTTPError(f"Status {response.status_code}: {response.text}")
        except requests.RequestException as e:
            logging.warning(f"Request failed: {str(e)}")
            retry_count += 1
            if retry_count > max_retries:
                logging.error("Max retries exceeded")
                break
            sleep_time = exponential_backoff(retry_count)
            logging.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
            continue

        time.sleep(delay)

        cursor = data.get('cursor')
        search_id = data.get('search_id')
        if not data.get('has_more', False) or (payload.get('cursor') == cursor and payload.get('search_id') == search_id):
            break

        payload['cursor'] = cursor
        if search_id:
            payload['search_id'] = search_id

    return results

def main():
    access_token = get_access_token(CLIENT_KEY, CLIENT_SECRET)
    if not access_token:
        logging.error("Access token retrieval failed. Exiting.")
        return

    start_date = None # e.g. datetime(2022, 1, 1)
    end_date = None # e.g. datetime(2025, 1, 1)
    delta = timedelta(days=30)

    all_videos_df = pd.DataFrame()
    session = requests.Session()
    total_days = (end_date - start_date).days

    with tqdm(total=total_days, desc="Fetching TikTok videos") as pbar:
        while start_date < end_date:
            start_str = start_date.strftime("%Y%m%d")
            next_date = min(start_date + delta, end_date)
            end_str = next_date.strftime("%Y%m%d")

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            query_params = {
                "fields": "id,video_description,create_time,share_count,view_count,like_count,comment_count,username,voice_to_text",
                "start_date": start_str,
                "end_date": end_str
            }

            payload = {
                "query": {
                    "and": [
                        {
                            "operation": "IN",
                            "field_name": "hashtag_name",
                            "field_values": ["photovoltaik", "solarenergie", "energiewende"]
                        }
                    ]
                },
                "cursor": 0,
                "max_count": 100,
                "start_date": start_str,
                "end_date": end_str,
            }

            videos = fetch_videos(session, headers, "https://open.tiktokapis.com/v2/research/video/query/", payload, query_params)
            df = pd.DataFrame(videos).astype(str)

            for index, row in df.iterrows():
                video_link = f"https://www.tiktok.com/@{row['username']}/video/{row['id']}"
                df.loc[index, 'video_link'] = video_link

            all_videos_df = pd.concat([all_videos_df, df], ignore_index=True)
            pbar.update((next_date - start_date).days)
            start_date = next_date

    all_videos_df.drop_duplicates(subset=['id'], inplace=True)
    filename = f"tiktok_videos_{datetime.now().strftime('%Y%m%d-%H%M%S')}.xlsx"
    all_videos_df.to_excel(filename, index=False, engine='openpyxl')
    logging.info(f"Data saved to {filename}")

if __name__ == "__main__":
    main()
