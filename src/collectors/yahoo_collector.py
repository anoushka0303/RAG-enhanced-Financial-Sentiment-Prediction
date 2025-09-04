import yfinance as yf
import pandas as pd
import os

TICKERS = [
    "AAPL",           
    "RELIANCE.NS",    
    "HDFCBANK.NS",    
    "BHARTIARTL.NS",  
    "TCS.NS",         
    "ICICIBANK.NS",   
    "SBIN.NS",        
    "HINDUNILVR.NS",  
    "INFY.NS",        
    "BAJFINANCE.NS",  
    "LICI.NS"         
]

def fetch_yahoo_data(tickers, period="1y", interval="1d", save_path="../../data/raw/yfinance_data.xlsx"):
    all_data = []

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        df.reset_index(inplace=True)
        
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        #print(df["Date"])
        df["company"] = ticker
        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_excel(save_path, index=False)  
    print(f"Saved yfinance data → {save_path}")

if __name__ == "__main__":
    fetch_yahoo_data(TICKERS)