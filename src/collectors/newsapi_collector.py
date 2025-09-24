'''import os
import requests
from bs4 import BeautifulSoup
import time
import json
from datetime import datetime
import logging
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


API_KEY = os.getenv('NEWS_API_KEY')


tickers = [          
    "RELIANCE.NS",            
]

last_fetched_ids = {}

def get_article_content(url):
    """Scrape the full article content from the given URL."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            content = "\n".join([para.get_text() for para in paragraphs])
            return content
        else:
            print(f"Failed to retrieve content from {url}")
            return None
    except Exception as e:
        logger.error(f"Error retrieving content from {url}: {e}")
        return None

def save_locally(data, folder_path="../../data/raw/", filename="article.xlsx"):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)
    
    df = pd.DataFrame([data])
    
    # If the file exists, append; else, create new
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_excel(file_path, index=False)
    logger.info(f"Saved {filename} to {folder_path}")

def fetch_and_store_news():
    base_url = "https://api.polygon.io/v2/reference/news"
    batch_size = 5  # Fetch 5 articles per batch

    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]

        for ticker in batch_tickers:
            params = {
                "ticker": ticker,
                "limit": 1,
                "apiKey": API_KEY
            }
            response = requests.get(base_url, params=params)
            print(response.json())

            if response.status_code == 200:
                data = response.json()
                if data["results"]:
                    article = data["results"][0]
                    article_id = article["id"]

                    # Skip if already fetched
                    if last_fetched_ids.get(ticker) == article_id:
                        logger.info(f"No new article for {ticker}. Skipping...")
                        continue

                    last_fetched_ids[ticker] = article_id
                    article_url = article["article_url"]
                    logger.info(f"Fetching new content for {ticker} from {article_url}")

                    content = get_article_content(article_url)
                    if content:
                        published_utc = article["published_utc"]
                        dt = datetime.strptime(published_utc, "%Y-%m-%dT%H:%M:%SZ")

                        
                        folder_path = os.path.join(
                            "data", "news_articles", ticker,
                            dt.strftime("%Y"), f"Month={dt.strftime('%m')}",
                            f"Day={dt.strftime('%d')}", f"Hour={dt.strftime('%H')}",
                            f"Minute={dt.strftime('%M')}"
                        )
                        filename = f"{ticker}_{dt.strftime('%S')}.json"

                        article_data = {
                            "ticker": ticker,
                            "title": article["title"],
                            "summary": article["description"],
                            "content": content,
                            "published_utc": published_utc
                        }

                        save_locally(article_data, folder_path, filename)
                        logger.info(f"Stored article for {ticker} in {folder_path}/{filename}")

            else:
                logger.error(f"Failed to retrieve news for {ticker}. Status Code: {response.status_code}")

            


        print("Waiting for 1 minute before fetching the next batch of articles...")
        time.sleep(60)

if __name__ == "__main__":
    while True:
        fetch_and_store_news()
        print("fetched and stored news articles. Waiting for 10 minutes before the next fetch...") 
        time.sleep(60)  '''
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