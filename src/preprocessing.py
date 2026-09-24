import numpy as np
import pandas as pd # Keep pandas for pd.isna check
from tensorflow.keras.utils import to_categorical
from src.config import N_STEPS, K_STEPS, PROFIT_TAKE_FACTOR, STOP_LOSS_FACTOR, MAX_HOLDING_PERIOD

def create_sequences_triple_label(features: np.ndarray, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray, atrs_for_labeling: np.ndarray, n_steps: int, k_steps: int):
    """
    Creates sequences with Triple Barrier Method labels.
    - features: The NORMALIZED input data for the model (X).
    - prices: The RAW Close price data to generate labels (y).
    - highs: The RAW High price data for barriers.
    - lows: The RAW Low price data for barriers.
    - atrs_for_labeling: The RAW ATR data specifically for barrier calculation.
    All input arrays (features, prices, highs, lows, atrs_for_labeling) must be aligned.
    """
    X, y = [], []
    
    for i in range(len(features) - n_steps - k_steps + 1):
        # Input sequence of features
        X.append(features[i:(i + n_steps)])
        
        current_price = prices[i + n_steps - 1]
        current_atr = atrs_for_labeling[i + n_steps - 1]
        
        if pd.isna(current_atr): 
            y.append(1) # Default to Hold if ATR is missing
            continue

        # Define barriers
        profit_barrier = current_price + (PROFIT_TAKE_FACTOR * current_atr)
        loss_barrier = current_price - (STOP_LOSS_FACTOR * current_atr)
        
        # Look into the future (up to MAX_HOLDING_PERIOD)
        future_slice_start = i + n_steps
        future_slice_end = min(i + n_steps + MAX_HOLDING_PERIOD, len(prices))
        
        if future_slice_start >= future_slice_end:
            y.append(1)
            continue

        future_highs_slice = highs[future_slice_start : future_slice_end]
        future_lows_slice = lows[future_slice_start : future_slice_end]

        label = 1 # Default to Hold
        
        # Check for profit or loss barrier hit
        # This loop needs to iterate over each element of the slice
        for j in range(len(future_highs_slice)):
            if future_highs_slice[j] >= profit_barrier:
                label = 2 # Buy signal (hit profit barrier first)
                break
            elif future_lows_slice[j] <= loss_barrier:
                label = 0 # Sell signal (hit loss barrier first)
                break
        
        y.append(label)
            
    return np.array(X), to_categorical(np.array(y), num_classes=3)
