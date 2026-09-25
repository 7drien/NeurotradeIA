import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
import numpy as np
from src.data_loader import get_data
from src.preprocessing import create_sequences_triple_label
from src.model import create_lstm_model
from src.callbacks import UILoggerCallback, BacktestOnEpochEnd
from src.config import TICKER, INTERVAL, N_STEPS, K_STEPS, EPOCHS, BATCH_SIZE, PROFIT_TAKE_FACTOR, STOP_LOSS_FACTOR, MAX_HOLDING_PERIOD

# Helper function to calculate ATR for labeling (moved from preprocessing)
def _calculate_atr_for_labeling(df_original, period=14):
    high_low = df_original['High'] - df_original['Low']
    high_close = np.abs(df_original['High'] - df_original['Close'].shift())
    low_close = np.abs(df_original['Low'] - df_original['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df_original['ATR_label'] = true_range.rolling(period).mean()
    return df_original

def train_model_with_callback(queue):
    """
    Trains the multi-class classification model with Triple Barrier labeling.
    """
    print("Starting training process...")
    data = get_data(ticker=TICKER, interval=INTERVAL)
    if data.empty:
        queue.put({'type': 'train_finished'})
        return

    print("Preparing raw OHLCV data and advanced features...")
    from src.features import add_advanced_features
    
    # Calculate ATR_label on the original data first for labeling
    data = _calculate_atr_for_labeling(data.copy())
    
    # Add advanced features
    data_with_features = add_advanced_features(data.copy())

    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Close_FracDiff', 'GK_Vol_14', 'Rolling_VWAP', 'RSI_14']
    
    # We need 'High', 'Low', 'Close', 'ATR_label' for labeling, and the features for training
    all_needed_cols = list(set(feature_cols + ['ATR_label', 'High', 'Low', 'Close']))
    df_processed = data_with_features[all_needed_cols].copy()
    df_processed.dropna(inplace=True)

    # Now df_processed is aligned and clean
    prices_for_labels = df_processed['Close'].values
    highs_for_labels = df_processed['High'].values
    lows_for_labels = df_processed['Low'].values
    atrs_for_labeling = df_processed['ATR_label'].values
    
    scaler = StandardScaler()
    # Scale only the OHLCV features, ensuring the order is consistent
    scaled_features = scaler.fit_transform(df_processed[feature_cols])
    joblib.dump(scaler, 'scaler.joblib')
    print("Scaler for raw features has been saved.")
    
    # Pass scaled features and the ALIGNED raw data (for labels) to the sequence function
    X, y, sample_weights = create_sequences_triple_label(
        scaled_features, 
        prices_for_labels, 
        highs_for_labels,
        lows_for_labels, 
        atrs_for_labeling, 
        N_STEPS, 
        K_STEPS
    )

    if len(X) == 0:
        queue.put({'type': 'train_finished'})
        return

    # --- Class Weight Calculation (Only on actual signals) ---
    # We only compute class weights where sample_weights == 1
    y_integers = np.argmax(y, axis=1)
    valid_indices = np.where(sample_weights == 1.0)[0]
    
    if len(valid_indices) > 0:
        valid_y = y_integers[valid_indices]
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(valid_y), y=valid_y)
        class_weights_dict = dict(enumerate(weights))
        print(f"Class Weights (on actual signals): Failure (0)={class_weights_dict.get(0, 1.0):.2f}, Success (1)={class_weights_dict.get(1, 1.0):.2f}")
        
        # Incorporate class weights into sample_weights directly
        for i in valid_indices:
            sample_weights[i] *= class_weights_dict.get(y_integers[i], 1.0)
    else:
        print("No valid signals found for training.")

    X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(X, y, sample_weights, test_size=0.2, shuffle=False)

    if len(X_train) == 0 or len(X_val) == 0:
        queue.put({'type': 'train_finished'})
        return

    model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # --- Callbacks ---
    ui_callback = UILoggerCallback(queue)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    # Must be after model_checkpoint so the .keras file exists!
    backtest_callback = BacktestOnEpochEnd(queue, frequency=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

    print("Starting model fitting with Meta-Labeling and sample weights...")
    model.fit(
        X_train, y_train,
        sample_weight=sw_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val, sw_val),
        callbacks=[ui_callback, model_checkpoint, backtest_callback, early_stopping, reduce_lr],
        verbose=1
    )
