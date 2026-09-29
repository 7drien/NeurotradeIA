import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from src.config import N_STEPS, K_STEPS, PROFIT_TAKE_FACTOR, STOP_LOSS_FACTOR, MAX_HOLDING_PERIOD

FEATURE_COLS = ['Open', 'Close', 'Low', 'High', 'Volume', 'RSI', 'Z_Score', 'Williams_R']

def compute_rsi(series, period=14):
    """Computes RSI for a pandas Series."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))

def create_sequences(df: pd.DataFrame, n_steps: int = 128, k_steps: int = 16):
    """
    Creates 1024-dimensional input feature vectors and regression targets.
    
    Input vector per sample:
    - 128 consecutive candles of 8 features:
      (Open, Close, Low, High, Volume, RSI, Z-Score, Williams %R)
    - 128 * 8 = 1024 values.
    
    Normalization:
    - Prices (Open, Close, Low, High) are normalized using the SAME mean and std
      across the 128-candle segment, preserving relative price levels and candle structures.
    - Volume is normalized by its mean and std in the 128-candle segment.
    - RSI is scaled to [-1, 1] centered at 50.
    - Z-Score is clipped to [-5, 5].
    - Williams %R is scaled to [-1, 1] centered at -50.
    
    Target:
    - Future price ratio: Close[t + 16] / Close[t], where t is the 128th candle.
    - Values in (0, 2), where 1.0 represents unchanged price.
    """
    # Ensure all required features are present
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: {col}")

    features_data = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    close_prices = df['Close'].to_numpy(dtype=np.float32)

    X, y = [], []
    num_samples = len(df) - n_steps - k_steps + 1

    for i in range(num_samples):
        # Extract 128 candles of 8 features
        seq = features_data[i : i + n_steps].copy() # shape: (128, 8)
        
        # 1. Price normalization (Open=0, Close=1, Low=2, High=3)
        # All 4 price series are normalized by the SAME mean and standard deviation
        price_slice = seq[:, 0:4]
        price_mean = np.mean(price_slice)
        price_std = np.std(price_slice)
        if price_std < 1e-8:
            price_std = 1e-8
        seq[:, 0:4] = (price_slice - price_mean) / price_std

        # 2. Volume normalization (Volume=4)
        vol_slice = seq[:, 4]
        vol_mean = np.mean(vol_slice)
        vol_std = np.std(vol_slice)
        if vol_std < 1e-8:
            vol_std = 1e-8
        seq[:, 4] = (vol_slice - vol_mean) / vol_std

        # 3. RSI normalization (RSI=5) -> scale [0, 100] to [-1, 1]
        seq[:, 5] = (seq[:, 5] - 50.0) / 50.0

        # 4. Z-Score normalization (Z_Score=6) -> clip to [-5, 5]
        seq[:, 6] = np.clip(seq[:, 6], -5.0, 5.0)

        # 5. Williams %R normalization (Williams_R=7) -> scale [-100, 0] to [-1, 1]
        seq[:, 7] = (seq[:, 7] + 50.0) / 50.0

        # Target calculation: price in 16 candles / price of 128th candle
        current_price = close_prices[i + n_steps - 1]
        future_price = close_prices[i + n_steps + k_steps - 1]
        target_ratio = future_price / (current_price + 1e-8)

        # Flatten 128 candles * 8 features into a single 1024-dimensional vector
        X.append(seq.flatten())
        y.append(target_ratio)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

def create_sequences_triple_label(features: np.ndarray, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray, atrs_for_labeling: np.ndarray, n_steps: int, k_steps: int):
    """
    Legacy sequence generator for Meta-Labeling backward compatibility.
    """
    X, y, sample_weights = [], [], []
    prices_series = pd.Series(prices)
    z_score = (prices_series - prices_series.rolling(100).mean()) / prices_series.rolling(100).std()
    rsi = compute_rsi(prices_series, 14)
    
    cond_long = (z_score < -2) & (rsi < 30)
    cond_short = (z_score > 2) & (rsi > 70)
    cross_long = cond_long & ~cond_long.shift(1).fillna(False)
    cross_short = cond_short & ~cond_short.shift(1).fillna(False)
    primary_signals = np.where(cross_long, 2, np.where(cross_short, 0, 1))
    
    for i in range(100, len(features) - n_steps - k_steps + 1):
        seq = features[i:(i + n_steps)].copy()
        seq_mean = np.mean(seq, axis=0)
        seq_std = np.std(seq, axis=0) + 1e-8
        seq = (seq - seq_mean) / seq_std
        X.append(seq)
        
        current_price = prices[i + n_steps - 1]
        current_atr = atrs_for_labeling[i + n_steps - 1]
        primary_signal = primary_signals[i + n_steps - 1]
        
        if pd.isna(current_atr) or pd.isna(primary_signal) or primary_signal == 1: 
            y.append(0)
            sample_weights.append(0.0)
            continue

        profit_barrier = current_price + (PROFIT_TAKE_FACTOR * current_atr) if primary_signal == 2 else current_price - (PROFIT_TAKE_FACTOR * current_atr)
        loss_barrier = current_price - (STOP_LOSS_FACTOR * current_atr) if primary_signal == 2 else current_price + (STOP_LOSS_FACTOR * current_atr)
        
        future_slice_start = i + n_steps
        future_slice_end = min(i + n_steps + MAX_HOLDING_PERIOD, len(prices))
        
        if future_slice_start >= future_slice_end:
            y.append(0)
            sample_weights.append(1.0)
            continue

        future_highs_slice = highs[future_slice_start : future_slice_end]
        future_lows_slice = lows[future_slice_start : future_slice_end]

        meta_label = 0
        for j in range(len(future_highs_slice)):
            if primary_signal == 2:
                if future_highs_slice[j] >= profit_barrier:
                    meta_label = 1
                    break
                elif future_lows_slice[j] <= loss_barrier:
                    meta_label = 0
                    break
            else:
                if future_lows_slice[j] <= profit_barrier:
                    meta_label = 1
                    break
                elif future_highs_slice[j] >= loss_barrier:
                    meta_label = 0
                    break
        
        y.append(meta_label)
        sample_weights.append(1.0)
            
    return np.array(X), to_categorical(np.array(y), num_classes=2), np.array(sample_weights)
