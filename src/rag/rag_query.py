import pickle
from chromadb import Client
from chromadb.config import Settings
with open('../../data/raw/news/embeddings/reliance_embeddings.pkl', 'rb') as f:
    texts, embeddings = pickle.load(f)

client = Client()
collection = client.get_or_create_collection(name="reliance_news")

def retrieve_news(query, k=3):
    from transformers import TFBertModel, BertTokenizer
    import tensorflow as tf

    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = TFBertModel.from_pretrained("yiyanghkust/finbert-tone")

    inputs = tokenizer(query, return_tensors='tf', truncation=True, padding=True, max_length=512)
    emb = model(**inputs).last_hidden_state
    query_emb = tf.reduce_mean(emb, axis=1).numpy()

    results = collection.query(query_embeddings=query_emb.tolist(), n_results=k)
    return results['documents'][0]


# news = retrieve_news("Reliance quarterly performance")
# print(news)