import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.data_loader import get_data
from src.features import add_advanced_features
from src.preprocessing import create_sequences
from src.model import create_dense_model, two_sigmoid, directional_accuracy
from src.callbacks import UILoggerCallback, BacktestOnEpochEnd
import src.config as config
from src.config import TICKER, INTERVAL, N_STEPS, K_STEPS, EPOCHS, BATCH_SIZE, LEARNING_RATE

def train_model_with_callback(queue, params=None):
    """
    Trains the 4-layer MLP (256-64-16-8) with 1024-dimensional inputs and 2*sigmoid output.
    Predicts the price ratio 16 candles ahead (Close[t+16] / Close[t]).
    Accepts a dictionary of parameters from the UI to dynamically override configuration.
    """
    if params is None:
        params = {}
    
    epochs = params.get('epochs', EPOCHS)
    batch_size = params.get('batch_size', BATCH_SIZE)
    n_steps = params.get('n_steps', N_STEPS)
    k_steps = params.get('k_steps', K_STEPS)
    days_to_load = params.get('days_to_load', None)
    learning_rate = params.get('learning_rate', LEARNING_RATE)
    
    print(f"Starting training process: Epochs={epochs}, Batch={batch_size}, N_Steps={n_steps}, K_Steps={k_steps}, Days={days_to_load}, LR={learning_rate}")
    
    start_date_str = config.START_DATE_STR
    if days_to_load is not None:
        start_date = datetime.datetime.now() - datetime.timedelta(days=days_to_load)
        start_date_str = start_date.strftime('%Y-%m-%d')
        
    data = get_data(ticker=TICKER, interval=INTERVAL, start_date=start_date_str)
    if data.empty:
        queue.put({'type': 'train_finished'})
        return

    print("Computing technical indicators (RSI, Z-Score, Williams %R)...")
    df_processed = add_advanced_features(data.copy())

    print("Generating 1024-dimensional normalized input vectors and 16-candle target ratios...")
    X, y = create_sequences(df_processed, n_steps=n_steps, k_steps=k_steps)

    if len(X) == 0:
        print("Error: No sequences created.")
        queue.put({'type': 'train_finished'})
        return

    print(f"Dataset generated: X shape = {X.shape}, y shape = {y.shape}")

    # Strict chronological split: 80% train, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    if len(X_train) == 0 or len(X_val) == 0:
        queue.put({'type': 'train_finished'})
        return

    print(f"Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
    print(f"y_train statistics: Mean={np.mean(y_train):.4f}, Min={np.min(y_train):.4f}, Max={np.max(y_train):.4f}")

    # Create 4-layer MLP model: 1024 -> 256 -> 64 -> 16 -> 8 -> 1 (2*sigmoid)
    model = create_dense_model(input_dim=X_train.shape[1], lr=learning_rate)
    model.summary()

    # Callbacks
    ui_callback = UILoggerCallback(queue)
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    backtest_callback = BacktestOnEpochEnd(queue, frequency=1, params=params)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

    callbacks_list = [ui_callback, model_checkpoint, backtest_callback, reduce_lr]

    # Only enable EarlyStopping if explicitly requested in params
    use_early_stopping = params.get('early_stopping', False)
    if use_early_stopping:
        patience = params.get('patience', 15)
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
        callbacks_list.append(early_stopping)
        print(f"EarlyStopping enabled (patience={patience}).")
    else:
        print(f"EarlyStopping disabled: training will run for the full {epochs} epochs (best weights saved via ModelCheckpoint).")

    print(f"Fitting MLP model for {epochs} epochs...")
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
