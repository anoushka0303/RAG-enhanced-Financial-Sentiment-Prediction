import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

class FinBertSentiment:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        self.labels = ["negative", "neutral", "positive"]

    def predict(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            output = self.model(**tokens)
        scores = torch.nn.functional.softmax(output.logits, dim=-1)[0]
        sentiment = self.labels[ scores.argmax().item() ]
        return sentiment, scores.tolist()

if __name__ == "__main__":
    finbert = FinBertSentiment()
    
    df = pd.read_excel("../../data/raw/newsapi_data.xlsx")
    
    df["sentiment"] = df["description"].astype(str).apply(lambda x: finbert.predict(x)[0])
    df.to_excel("../../data/processed/newsapi_with_sentiment.xlsx", index=False)
    print("Saved sentiment-annotated data")