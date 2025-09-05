from dotenv import load_dotenv
import os
import pandas as pd
import requests

load_dotenv()
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

def fetch_alpha_vantage(symbol, function="TIME_SERIES_DAILY", outputsize="compact"):
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={outputsize}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if "Time Series (Daily)" not in data:
        raise Exception(f"Error fetching data for {symbol}: {data}")
    
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

if __name__ == "__main__":
    companies = [
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
    
    save_dir = "../../data/raw"
    os.makedirs(save_dir, exist_ok=True)

    for c in companies:
        try:
            df = fetch_alpha_vantage(c)
            save_path = os.path.join(save_dir, f"{c}_alpha_vantage.xlsx")
            df.to_excel(save_path)
            print(f"Saved {c} → {save_path}")
        except Exception as e:
            print(f"Failed for {c}: {e}")
