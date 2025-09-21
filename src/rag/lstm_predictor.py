import tensorflow as tf
import pandas as pd
import numpy as np

model = tf.keras.models.load_model('../../models/lstm_model_reliance_1.h5')

def predict_stock(df):
    X = df.values.reshape((1, df.shape[0], df.shape[1]))
    prediction = model.predict(X)
    return prediction[0][0]

# df = pd.read_csv('../../data/raw/stock/reliance_stock.csv')
# print(predict_stock(df))