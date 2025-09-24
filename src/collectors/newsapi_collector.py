
import os
import requests
import time
import json
from datetime import datetime
import logging
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

API_KEY = os.getenv('NEWS_API_KEY')

tickers = [
    "RELIANCE",
]

last_fetched_titles = set() 

def fetch_articles_from_newsapi(ticker):
    """Fetch latest news articles from NewsAPI for a given ticker."""
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={API_KEY}&pageSize=5"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        logger.error(f"Failed to fetch news for {ticker}. Status code: {response.status_code}")
        return []

def save_to_excel(data, folder_path="../../data/raw/", filename="articles.xlsx"):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)
    df = pd.DataFrame(data)

    
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_excel(file_path, index=False)
    logger.info(f"Saved {filename} to {folder_path}")

def fetch_and_store_news():
    all_articles = []

    for ticker in tickers:
        articles = fetch_articles_from_newsapi(ticker)
        for article in articles:
            title = article.get("title")
            if title in last_fetched_titles:
                continue 

            last_fetched_titles.add(title)

            published_utc = article.get("publishedAt")
            dt = datetime.strptime(published_utc, "%Y-%m-%dT%H:%M:%SZ") if published_utc else None

            article_data = {
                "ticker": ticker,
                "title": title,
                "description": article.get("description"),
                "content": article.get("content"),
                "url": article.get("url"),
                "published_utc": published_utc,
            }
            all_articles.append(article_data)
            logger.info(f"Fetched article for {ticker}: {title}")

    if all_articles:
        save_to_excel(all_articles)

if __name__ == "__main__":
    while True:
        fetch_and_store_news()
        logger.info("Fetched and stored news articles. Waiting 10 minutes before next fetch...")
        time.sleep(60)  