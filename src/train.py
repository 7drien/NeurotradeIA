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

    # --- Filter Data (Dense Batches for Generalization) ---
    # To prevent 98% of the batch being empty (sample_weight=0), which causes extremely noisy gradients,
    # we filter the dataset to ONLY contain actual trading signals before training.
    valid_indices = np.where(sample_weights == 1.0)[0]
    
    if len(valid_indices) == 0:
        print("No valid signals found for training.")
        queue.put({'type': 'train_finished'})
        return
        
    X_dense = X[valid_indices]
    y_dense = y[valid_indices]
    y_integers_dense = np.argmax(y_dense, axis=1)
    
    # Compute class weights on the dense targets
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_integers_dense), y=y_integers_dense)
    class_weights_dict = dict(enumerate(weights))
    
    # --- PENALIZE NOT TRADING ---
    # The user requested to penalize the model if it doesn't trade.
    # We heavily increase the weight of class 1 (Approve Trade) so the network is biased towards taking trades.
    class_weights_dict[1] = class_weights_dict.get(1, 1.0) * 2.5
    
    print(f"Class Weights (on actual signals): Failure/Hold (0)={class_weights_dict.get(0, 1.0):.2f}, Success/Trade (1)={class_weights_dict.get(1, 1.0):.2f}")

    # Temporal split on dense trades
    X_train, X_val, y_train, y_val = train_test_split(X_dense, y_dense, test_size=0.2, shuffle=False)

    if len(X_train) == 0 or len(X_val) == 0:
        queue.put({'type': 'train_finished'})
        return

    model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # --- Callbacks ---
    ui_callback = UILoggerCallback(queue)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    # Must be after model_checkpoint so the .keras file exists!
    backtest_callback = BacktestOnEpochEnd(queue, frequency=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001, verbose=1)

    print("Starting model fitting on DENSE Meta-Labeling signals...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[ui_callback, model_checkpoint, backtest_callback, early_stopping, reduce_lr],
        class_weight=class_weights_dict,
        verbose=1
    )
