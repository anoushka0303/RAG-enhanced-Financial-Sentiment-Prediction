import pickle
import numpy as np
from chromadb import Client
from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")

with open(r"../data/raw/news/embeddings/reliance_embeddings.pkl", "rb") as f:
    news_texts, news_embeddings = pickle.load(f)

with open(r"../data/raw/stock/reliance_stock_embeddings.pkl", "rb") as f:
    stock_embeddings = pickle.load(f) 

news_embeddings = np.array(news_embeddings, dtype=np.float32)
stock_embeddings = np.array(stock_embeddings, dtype=np.float32)

client = Client()
news_collection = client.get_or_create_collection(name="reliance_news")
stock_collection = client.get_or_create_collection(name="reliance_stock")

if news_collection.count() == 0:
    news_collection.add(
        documents=news_texts,
        embeddings=news_embeddings.tolist(),
        metadatas=[{"source": "newsapi"}]*len(news_texts),
        ids=[f"news_{i}" for i in range(len(news_texts))]
    )

if stock_collection.count() == 0:
    stock_collection.add(
        documents=[f"Stock day {i}" for i in range(len(stock_embeddings))],
        embeddings=stock_embeddings.tolist(),
        metadatas=[{"type": "OHLCV+MA"}]*len(stock_embeddings),
        ids=[f"stock_{i}" for i in range(len(stock_embeddings))]
    )

print("News and stock embeddings loaded into ChromaDB")

tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = TFBertModel.from_pretrained("yiyanghkust/finbert-tone")

def embed_text(text):
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=512)
    emb = model(**inputs).last_hidden_state
    return tf.reduce_mean(emb, axis=1).numpy()[0]

def retrieve_top_docs(query, stock_query_vector=None, top_k=3):
    query_emb = embed_text(query).tolist()
    news_results = news_collection.query(query_embeddings=[query_emb], n_results=top_k)
    top_news = news_results['documents'][0]
    if stock_query_vector is None:
        stock_query_vector = stock_embeddings[-1].tolist()
    stock_results = stock_collection.query(query_embeddings=[stock_query_vector], n_results=top_k)
    top_stock_days = stock_results['documents'][0]
    return top_news, top_stock_days

queries = [
    "Reliance stock performance",
    "Recent news affecting Reliance share price",
    "Reliance quarterly results"
]

for q in queries:
    news_docs, stock_docs = retrieve_top_docs(q)
    pprint(f"\nQuery: {q}")
    pprint("Top news docs:")
    pprint(news_docs)
    pprint("Top stock days:")
    pprint(stock_docs)