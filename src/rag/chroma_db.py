import pickle
from chromadb import Client
from chromadb.config import Settings

with open(r"C:\Users\Dell\OneDrive\Desktop\dump\RAG-based financial sentiment analysis\src\data\raw\news\embeddings\reliance_embeddings.pkl", "rb") as f:
    texts, embeddings = pickle.load(f)

client = Client()  
collection = client.get_or_create_collection(name="reliance_news")

collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=[{"source": "newsapi"}]*len(texts),
    ids=[str(i) for i in range(len(texts))]
)

print("News embeddings loaded into ChromaDB")