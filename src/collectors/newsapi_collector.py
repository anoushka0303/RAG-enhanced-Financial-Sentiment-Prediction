import os
import pandas as pd
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWS_API_KEY") 
client = NewsApiClient(api_key=NEWSAPI_KEY)

QUERIES = [
    "Nifty 50 performance",
    "Sensex today",
    "Indian stock market news",
    "Dow Jones news",
    "Nasdaq performance",
    "RBI monetary policy",
    "Repo rate India",
    "Indian banking sector",
    "Non-performing assets India",
    "Fintech India",
    "Digital payments UPI",
    "US Federal Reserve interest rates",
    "China economy slowdown",
    "Crude oil prices India impact",
    "Gold price news",
    "Rupee vs dollar",
    "Global recession fears",
    "Mutual funds India",
    "Best IPOs in India",
    "ETFs India",
    "Long term stock picks",
    "AI in stock trading",
    "Blockchain in banking",
    "Crypto regulations India",
    "Fintech funding India",
    "Stock market crash India",
    "Banking fraud news",
    "Indian housing market",
    "Corporate layoffs news"
]

def fetch_news(queries, language="en", page_size=20, save_path="../../data/raw/newsapi_data.xlsx"):
    all_articles = []

    for query in queries:
        response = client.get_everything(
            q=query,
            language=language,
            sort_by="relevancy",
            page_size=page_size
        )
        articles = response.get("articles", [])
        for a in articles:
            all_articles.append({
                "query": query,
                "headline": a.get("title"),
                "description": a.get("description"),
                "publishedAt": a.get("publishedAt"),
                "url": a.get("url")
            })

    df = pd.DataFrame(all_articles)
    df.to_excel(save_path, index=False)
    print(f"Saved news data → {save_path}")

if __name__ == "__main__":
    fetch_news(QUERIES)