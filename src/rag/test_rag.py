import os
import pickle
import numpy as np
from pprint import pprint
from chromadb import Client
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensorflow.keras.models import load_model # type : ignore
from sentence_transformers import SentenceTransformer

# Load news and stock documents
with open(r"../data/raw/news/embeddings/reliance_embeddings.pkl", "rb") as f:
    news_texts, _ = pickle.load(f)

with open(r"../data/raw/stock/reliance_stock_embeddings.pkl", "rb") as f:
    stock_texts = pickle.load(f)

# Initialize sentence embeddings
embed_model = SentenceTransformer('all-mpnet-base-v2')  # 768-dim
news_embeddings = np.array([embed_model.encode([text])[0] for text in news_texts], dtype=np.float32)
stock_embeddings = np.array([embed_model.encode([text])[0] for text in stock_texts], dtype=np.float32)

# Initialize ChromaDB
client = Client()
news_collection = client.get_or_create_collection("news_new")
stock_collection = client.get_or_create_collection("stock_new")

# Add documents to collections if empty
if news_collection.count() == 0:
    news_collection.add(
        documents=news_texts,
        embeddings=news_embeddings.tolist(),
        metadatas=[{"source": "news"}]*len(news_texts),
        ids=[f"news_{i}" for i in range(len(news_texts))]
    )

if stock_collection.count() == 0:
    stock_collection.add(
        documents=stock_texts,
        embeddings=stock_embeddings.tolist(),
        metadatas=[{"type": "OHLCV+MA"}]*len(stock_texts),
        ids=[f"stock_{i}" for i in range(len(stock_texts))]
    )

# Load LSTM model
lstm_model = load_model(r"C:\Users\Dell\OneDrive\Desktop\dump\RAG-based financial sentiment analysis\notebooks\lstm_model_reliance_1.h5")

# Load LLM
llm_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
llm_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

# Embedding function
def embed_text(text):
    return embed_model.encode([text])[0]

# Predict stock trend using LSTM (expects proper sequence input)
def predict_stock_trend(stock_input):
    preds = lstm_model.predict(stock_input)
    return "Up" if preds[0][0] > 0.5 else "Down"

# Check if query is company-related
company_keywords = ["Reliance", "stock", "share", "price", "quarterly", "results"]
def is_company_query(query):
    return any(kw.lower() in query.lower() for kw in company_keywords)

# Retrieve top docs from ChromaDB
def retrieve_top_docs(query, top_k=3):
    query_emb = embed_text(query).tolist()
    news_results = news_collection.query(query_embeddings=[query_emb], n_results=top_k)
    stock_results = stock_collection.query(query_embeddings=[query_emb], n_results=top_k)
    news_docs = news_results['documents'][0]
    stock_indices = [int(doc.split(" ")[-1]) for doc in stock_results['documents'][0]]
    return news_docs, stock_indices

# Generate answer with optional stock trend
def generate_answer(query, news_docs, stock_trend=None):
    context = "News:\n" + "\n".join(news_docs)
    if stock_trend:
        context += f"\nPredicted Stock Trend: {stock_trend}"
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = llm_tokenizer(input_text, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_new_tokens=150)
    return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
queries = [
    "Reliance stock performance",
    "Recent news affecting Reliance share price",
    "Reliance quarterly results",
    "Global oil prices update"
]

for q in queries:
    news_docs, stock_indices = retrieve_top_docs(q)

    if is_company_query(q):
        # Example: use the latest stock data for prediction
        # stock_input must match the LSTM's expected 3D shape (1, timesteps, features)
        stock_input = np.expand_dims(stock_embeddings[-7:], axis=0)  # adjust 7 to your LSTM timesteps
        stock_trend = predict_stock_trend(stock_input)
    else:
        stock_trend = None

    answer = generate_answer(q, news_docs, stock_trend)
    pprint("\nQuery: " + q)
    pprint("Top news docs: " + str(news_docs))
    pprint("Stock trend: " + str(stock_trend))
    pprint("Generated answer: " + str(answer))