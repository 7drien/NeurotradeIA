import yfinance as yf
import pandas as pd
import os
import time
from src.config import TICKER, START_DATE_STR, END_DATE_STR, INTERVAL

DATA_CACHED_PATH = "data/cached"

def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)
    return df

def _download_with_retry(ticker, start_date, end_date, interval, retries=3, initial_delay=60):
    delay = initial_delay
    for i in range(retries):
        try:
            print(f"Downloading data for {ticker} ({interval}) (Attempt {i + 1}/{retries})...")
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            if not data.empty:
                return data
            print(f"Warning: No data downloaded for {ticker}.")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            if "RateLimitError" in str(e) or "Too Many Requests" in str(e):
                print(f"Rate limited. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"An unexpected error occurred: {e}")
                return pd.DataFrame()
    print("All download attempts failed.")
    return pd.DataFrame()

def get_data(ticker=TICKER, start_date=START_DATE_STR, end_date=END_DATE_STR, interval=INTERVAL, use_cache=True):
    os.makedirs(DATA_CACHED_PATH, exist_ok=True)
    cache_file = os.path.join(DATA_CACHED_PATH, f"{ticker}_{interval}.pkl")

    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached data from: {cache_file}")
        try:
            data = pd.read_pickle(cache_file)
            data = _validate_dataframe(data)
            if not data.empty:
                return data
        except Exception as e:
            print(f"Could not read cache file: {e}. Re-downloading...")

    data = _download_with_retry(ticker, start_date, end_date, interval)
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    data.index.name = 'Datetime'
    data = _validate_dataframe(data)

    if not data.empty:
        print(f"Saving validated data to cache: {cache_file}")
        data.to_pickle(cache_file)
    
    return data
