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

def _download_with_retry(ticker, start_date, end_date, interval, retries=3, initial_delay=10):
    delay = initial_delay
    for i in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
            if not data.empty:
                return data
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            if "RateLimitError" in str(e) or "Too Many Requests" in str(e):
                time.sleep(delay)
                delay *= 2
            else:
                return pd.DataFrame()
    return pd.DataFrame()

def _download_in_blocks(ticker, start_date, end_date, interval, block_size_days=30):
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    current_start = start_dt
    all_data = []
    
    print(f"Downloading data for {ticker} in blocks of {block_size_days} days to prevent rate limits...")
    
    while current_start < end_dt:
        current_end = min(current_start + pd.Timedelta(days=block_size_days), end_dt)
        print(f"  -> Fetching {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...")
        
        df_block = _download_with_retry(
            ticker, 
            current_start.strftime('%Y-%m-%d'), 
            current_end.strftime('%Y-%m-%d'), 
            interval
        )
        
        if not df_block.empty:
            all_data.append(df_block)
            
        current_start = current_end
        time.sleep(1.5) # Anti rate-limit sleep
        
    if all_data:
        combined_df = pd.concat(all_data)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df.sort_index(inplace=True)
        return combined_df
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

    data = _download_in_blocks(ticker, start_date, end_date, interval)
    if data.empty:
        print("Error: Could not retrieve any data.")
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    data.index.name = 'Datetime'
    data = _validate_dataframe(data)

    if not data.empty:
        print(f"Saving validated data to cache: {cache_file}")
        data.to_pickle(cache_file)
    
    return data
