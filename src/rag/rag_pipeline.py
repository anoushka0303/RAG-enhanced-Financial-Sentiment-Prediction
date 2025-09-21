from rag_query import retrieve_news
from lstm_predictor import predict_stock
import pandas as pd

def rag_stock_query(stock_df, user_query):
    stock_pred = predict_stock(stock_df)

    news_docs = retrieve_news(user_query)

    return {
        "stock_prediction": stock_pred,
        "news_summary": news_docs
    }

# df = pd.read_csv('../../data/raw/stock/reliance_stock.csv')
# query = "Reliance quarterly revenue update"
# result = rag_stock_query(df, query)
# print(result)