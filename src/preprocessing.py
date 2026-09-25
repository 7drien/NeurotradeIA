import numpy as np
import pandas as pd # Keep pandas for pd.isna check
from tensorflow.keras.utils import to_categorical
from src.config import N_STEPS, K_STEPS, PROFIT_TAKE_FACTOR, STOP_LOSS_FACTOR, MAX_HOLDING_PERIOD

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_sequences_triple_label(features: np.ndarray, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray, atrs_for_labeling: np.ndarray, n_steps: int, k_steps: int):
    """
    Creates sequences with Meta-Labeling.
    Primary Model: RSI + Z-Score Mean Reversion.
    Secondary Model (ML): Predicts 1 if Primary Model is correct (hits TP), 0 otherwise.
    """
    X, y, sample_weights = [], [], []
    
    # Calculate primary signals (RSI + Z-Score)
    prices_series = pd.Series(prices)
    z_score = (prices_series - prices_series.rolling(100).mean()) / prices_series.rolling(100).std()
    rsi = compute_rsi(prices_series, 14)
    
    # Vectorized signal generation (2 for LONG, 0 for SHORT, 1 for HOLD)
    # Trigger when both conditions are met (Mean Reversion setup)
    cond_long = (z_score < -2) & (rsi < 30)
    cond_short = (z_score > 2) & (rsi > 70)
    
    # Only trigger on the initial crossover of the condition
    cross_long = cond_long & ~cond_long.shift(1).fillna(False)
    cross_short = cond_short & ~cond_short.shift(1).fillna(False)
    primary_signals = np.where(cross_long, 2, np.where(cross_short, 0, 1))
    
    for i in range(100, len(features) - n_steps - k_steps + 1):
        # Input sequence of features
        seq = features[i:(i + n_steps)].copy()
        
        # SEQUENCE-WISE NORMALIZATION (Z-score per sequence)
        seq_mean = np.mean(seq, axis=0)
        seq_std = np.std(seq, axis=0) + 1e-8
        seq = (seq - seq_mean) / seq_std
        X.append(seq)
        
        current_price = prices[i + n_steps - 1]
        current_atr = atrs_for_labeling[i + n_steps - 1]
        primary_signal = primary_signals[i + n_steps - 1]
        
        if pd.isna(current_atr) or pd.isna(primary_signal) or primary_signal == 1: 
            y.append(0) # Default to failure/ignore so index mapping is preserved
            sample_weights.append(0.0) # IGNORE IN TRAINING
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

        meta_label = 0 # Default to failure
        
        for j in range(len(future_highs_slice)):
            if primary_signal == 2: # LONG
                if future_highs_slice[j] >= profit_barrier:
                    meta_label = 1 # Success
                    break
                elif future_lows_slice[j] <= loss_barrier:
                    meta_label = 0 # Failure
                    break
            else: # SHORT
                if future_lows_slice[j] <= profit_barrier:
                    meta_label = 1 # Success
                    break
                elif future_highs_slice[j] >= loss_barrier:
                    meta_label = 0 # Failure
                    break
        
        y.append(meta_label)
        sample_weights.append(1.0) # TRAIN ON THIS
            
    # Binary classification now!
    return np.array(X), to_categorical(np.array(y), num_classes=2), np.array(sample_weights)
