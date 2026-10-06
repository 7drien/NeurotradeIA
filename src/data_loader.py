import os
import io
import time
import datetime
import requests
import pandas as pd
import yfinance as yf
from src.config import TICKER, START_DATE_STR, END_DATE_STR, INTERVAL, TICKER_START_DATES

DATA_CACHED_PATH = "data/cached"

def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)
    return df

def _download_crypto_cdd(ticker: str) -> pd.DataFrame:
    """
    Downloads complete hourly historical crypto data from CryptoDataDownload (Gemini archive).
    Covers BTC from 2015 to current and ETH from 2016 to current.
    """
    sym = 'BTCUSD' if 'BTC' in ticker.upper() else 'ETHUSD'
    url = f'https://www.cryptodatadownload.com/cdd/Gemini_{sym}_1h.csv'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    print(f"Downloading historical {sym} 1h data from CryptoDataDownload...")
    res = requests.get(url, headers=headers, timeout=30)
    res.raise_for_status()

    df = pd.read_csv(io.StringIO(res.text), skiprows=1)
    col_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
    df.rename(columns=col_map, inplace=True)

    if 'Volume USD' in df.columns:
        df['Volume'] = df['Volume USD']
    elif f'Volume {sym[:3]}' in df.columns:
        df['Volume'] = df[f'Volume {sym[:3]}']
    else:
        vol_candidates = [c for c in df.columns if 'volume' in c.lower()]
        if vol_candidates:
            df['Volume'] = df[vol_candidates[0]]

    df['Datetime'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def _append_recent_binance(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Fetches recent 1h candles from Binance API to update the dataset up to the latest hour.
    """
    try:
        binance_symbol = 'BTCUSDT' if 'BTC' in ticker.upper() else 'ETHUSDT'
        last_dt = df.index.max()
        start_ts = int(last_dt.timestamp() * 1000)
        url = f'https://api.binance.com/api/v3/klines?symbol={binance_symbol}&interval=1h&startTime={start_ts}&limit=1000'
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            klines = res.json()
            if klines:
                rows = []
                for k in klines:
                    rows.append({
                        'Datetime': pd.to_datetime(k[0], unit='ms', utc=True),
                        'Open': float(k[1]),
                        'High': float(k[2]),
                        'Low': float(k[3]),
                        'Close': float(k[4]),
                        'Volume': float(k[7])
                    })
                df_new = pd.DataFrame(rows).set_index('Datetime')
                combined = pd.concat([df, df_new])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.sort_index(inplace=True)
                return combined
    except Exception as e:
        print(f"Note: Binance recent candles update skipped ({e})")
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
        time.sleep(1.5)

    if all_data:
        combined_df = pd.concat(all_data)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df.sort_index(inplace=True)
        return combined_df
    return pd.DataFrame()

def get_data(ticker=TICKER, start_date=None, end_date=None, interval=INTERVAL, use_cache=True, days_to_load=None):
    """
    Retrieves market data for a given ticker and interval.
    Automatically downloads from historical crypto archives (Gemini/Binance) or Yahoo Finance,
    and caches validated outputs locally in data/cached/.
    """
    if start_date is None:
        start_date = TICKER_START_DATES.get(ticker, START_DATE_STR)
    if end_date is None:
        end_date = END_DATE_STR

    if days_to_load is not None:
        start_dt = datetime.datetime.now() - datetime.timedelta(days=days_to_load)
        start_date = start_dt.strftime('%Y-%m-%d')

    os.makedirs(DATA_CACHED_PATH, exist_ok=True)
    cache_file = os.path.join(DATA_CACHED_PATH, f"{ticker}_{interval}.pkl")

    req_start_ts = pd.to_datetime(start_date, utc=True) if start_date else None
    req_end_ts = pd.to_datetime(end_date, utc=True) if end_date else None

    # Check cache coverage
    if use_cache and os.path.exists(cache_file):
        try:
            data = pd.read_pickle(cache_file)
            data = _validate_dataframe(data)
            if not data.empty:
                cache_min = data.index.min()
                # Verify cache covers requested start date (with 7 days tolerance)
                if req_start_ts is None or cache_min <= req_start_ts + pd.Timedelta(days=7):
                    print(f"Loading cached data from: {cache_file} ({len(data)} candles from {cache_min} to {data.index.max()})")
                    if req_start_ts:
                        data = data.loc[data.index >= req_start_ts]
                    if req_end_ts:
                        data = data.loc[data.index <= req_end_ts]
                    return data
                else:
                    print(f"Cached data starts at {cache_min}, but {start_date} is required. Re-downloading full history...")
        except Exception as e:
            print(f"Could not read cache file: {e}. Re-downloading...")

    # Download from historical archive for 1h crypto
    data = pd.DataFrame()
    is_crypto = any(c in ticker.upper() for c in ['BTC', 'ETH'])
    if interval == '1h' and is_crypto:
        try:
            data = _download_crypto_cdd(ticker)
            data = _append_recent_binance(data, ticker)
        except Exception as e:
            print(f"CryptoDataDownload failed: {e}. Falling back to Yahoo Finance...")
            data = pd.DataFrame()

    if data.empty:
        data = _download_in_blocks(ticker, start_date, end_date, interval)

    if data.empty:
        print(f"Error: Could not retrieve any data for {ticker}.")
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    data.index.name = 'Datetime'
    data = _validate_dataframe(data)

    if not data.empty:
        print(f"Saving validated data to cache: {cache_file} ({len(data)} candles)")
        data.to_pickle(cache_file)

    if req_start_ts:
        data = data.loc[data.index >= req_start_ts]
    if req_end_ts:
        data = data.loc[data.index <= req_end_ts]

    return data
