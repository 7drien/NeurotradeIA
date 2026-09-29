import numpy as np
import pandas as pd

def get_weights_ffd(d, threshold=1e-5):
    """
    Computes weights for the Fractional Differentiation using the Fixed-Width Window (FFD) method.
    """
    weights = [1.]
    k = 1
    while True:
        w_ = -weights[-1] * (d - k + 1) / k
        if abs(w_) < threshold:
            break
        weights.append(w_)
        k += 1
    return np.array(weights[::-1])

def frac_diff_ffd(series, d, threshold=1e-5):
    """
    Applies Fractional Differentiation to a pandas Series.
    """
    weights = get_weights_ffd(d, threshold)
    width = len(weights)
    
    # We use a rolling window dot product to apply the weights
    df_ = series.ffill().dropna()
    out = pd.Series(index=df_.index, dtype=float)
    
    for i in range(width - 1, len(df_)):
        out.iloc[i] = np.dot(weights, df_.iloc[i - width + 1 : i + 1])
        
    return out

def garman_klass_volatility(df, window=14):
    """
    Computes Garman-Klass Volatility which is more efficient than standard deviation of close prices.
    """
    log_hl = np.log(df['High'] / df['Low']) ** 2
    log_co = np.log(df['Close'] / df['Open']) ** 2
    
    gk_vol = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk_vol.rolling(window).mean())

def compute_vwap(df):
    """
    Computes the Volume Weighted Average Price (VWAP) assuming intraday data is continuous.
    To be fully correct, VWAP is usually anchored to a session (e.g. daily), 
    but a rolling VWAP is also useful. Here we use a rolling window of 1 day (96 periods for 15m).
    """
    window = 96 # 24h * 4 (15m periods)
    q = df['Volume']
    p = (df['High'] + df['Low'] + df['Close']) / 3
    
    rolling_vwap = (p * q).rolling(window=window).sum() / q.rolling(window=window).sum()
    return rolling_vwap

def add_advanced_features(df):
    """
    Main function to augment the dataframe with advanced quantitative features.
    """
    print("Computing advanced features (Fractional Differentiation, Volatility, VWAP)...")
    
    # 1. Fractional Differentiation (d=0.4 is often a good starting point to achieve stationarity while keeping memory)
    df['Close_FracDiff'] = frac_diff_ffd(df['Close'], d=0.4, threshold=1e-4)
    
    # 2. Garman-Klass Volatility
    df['GK_Vol_14'] = garman_klass_volatility(df, window=14)
    
    # 3. Rolling VWAP
    df['Rolling_VWAP'] = compute_vwap(df)
    
def compute_rsi(series, period=14):
    """
    Computes standard Relative Strength Index (RSI).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))

def compute_z_score(series, period=100):
    """
    Computes rolling Z-score: (x - mean) / std.
    """
    mean = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return (series - mean) / (std + 1e-8)

def compute_williams_r(df, period=14):
    """
    Computes Williams %R oscillator: (Highest_High - Close) / (Highest_High - Lowest_Low) * -100.
    Values range from -100 to 0.
    """
    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    return ((highest_high - df['Close']) / (highest_high - lowest_low + 1e-8)) * -100.0

def add_advanced_features(df):
    """
    Augments the dataframe with quantitative features:
    RSI, Z-Score, Williams %R, and legacy indicators.
    """
    print("Computing technical features (RSI, Z-Score, Williams %R)...")
    
    # 1. RSI (Relative Strength Index)
    df['RSI'] = compute_rsi(df['Close'], period=14)
    df['RSI_14'] = df['RSI'] # Backward compatibility
    
    # 2. Z-Score (Rolling 100-period price z-score)
    df['Z_Score'] = compute_z_score(df['Close'], period=100)
    
    # 3. Williams %R (14 periods)
    df['Williams_R'] = compute_williams_r(df, period=14)
    
    # Drop rows with NaN (from rolling windows)
    df.dropna(inplace=True)
    return df
