import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.data_loader import get_data
from src.features import add_advanced_features
from src.preprocessing import create_sequences, create_train_test_split
from src.model import create_dense_model, two_sigmoid, directional_accuracy
from src.callbacks import UILoggerCallback, BacktestOnEpochEnd
import src.config as config
from src.config import TICKER, INTERVAL, N_STEPS, K_STEPS, EPOCHS, BATCH_SIZE, LEARNING_RATE

def train_model_with_callback(queue, params=None):
    """
    Trains the 6-layer MLP (1024-512-256-64-16-8) with 2048-dimensional inputs and 2*sigmoid output.
    Predicts the price ratio k_steps candles ahead (Close[t+k_steps] / Close[t]).
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
    tickers_config = params.get('tickers_config')
    if tickers_config is None:
        single_ticker = params.get('ticker')
        if single_ticker:
            tickers_config = {single_ticker: getattr(config, 'TICKER_START_DATES', {}).get(single_ticker, config.START_DATE_STR)}
        else:
            tickers_config = getattr(config, 'TICKER_START_DATES', {
                'BTC-USD': '2016-01-01',
                'ETH-USD': '2018-01-01'
            })

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    btc_test_data = None

    for tkr, default_start in tickers_config.items():
        curr_start = default_start
        if days_to_load is not None:
            start_date = datetime.datetime.now() - datetime.timedelta(days=days_to_load)
            curr_start = start_date.strftime('%Y-%m-%d')
            
        print(f"Loading data for {tkr} starting from {curr_start}...")
        data = get_data(ticker=tkr, interval=INTERVAL, start_date=curr_start)
        if data.empty:
            print(f"Warning: No data retrieved for {tkr}.")
            continue

        print(f"Computing technical indicators for {tkr} ({len(data)} candles)...")
        df_processed = add_advanced_features(data.copy())

        print(f"Generating leak-free train/test split for {tkr} (80% train / 20% test)...")
        X_tr, y_tr, X_v, y_v, test_info = create_train_test_split(
            df_processed, n_steps=n_steps, k_steps=k_steps, train_ratio=0.8
        )
        if len(X_tr) == 0:
            print(f"Warning: No training sequences created for {tkr}.")
            continue

        X_train_list.append(X_tr)
        y_train_list.append(y_tr)
        if len(X_v) > 0:
            X_val_list.append(X_v)
            y_val_list.append(y_v)

        print(f"  -> {tkr}: {len(X_tr)} train samples (strictly before {test_info['split_date']}), {len(test_info['X_test'])} test candles")

        if 'BTC' in tkr.upper() or btc_test_data is None:
            btc_test_data = test_info
            btc_test_data['ticker'] = tkr

    if not X_train_list or not X_val_list:
        print("Error: No training sequences created.")
        queue.put({'type': 'train_finished'})
        return

    X_train = np.vstack(X_train_list)
    y_train = np.vstack(y_train_list)
    X_val = np.vstack(X_val_list)
    y_val = np.vstack(y_val_list)

    # Shuffle training set so mini-batches blend BTC and ETH samples
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    print(f"Combined dataset: Training={len(X_train)} samples, Validation={len(X_val)} samples")
    print(f"y_train statistics: Mean={np.mean(y_train):.4f}, Min={np.min(y_train):.4f}, Max={np.max(y_train):.4f}")

    # Create MLP model: 2048 -> 1024 -> 512 -> 256 -> 64 -> 16 -> 8 -> 1 (2*sigmoid)
    model = create_dense_model(input_dim=X_train.shape[1], lr=learning_rate)
    model.summary()

    # Callbacks
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='loss', save_best_only=True, verbose=1)
    backtest_callback = BacktestOnEpochEnd(queue, frequency=1, params=params, test_data=btc_test_data)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=0.00001, verbose=1)
    ui_callback = UILoggerCallback(queue)

    callbacks_list = [model_checkpoint, backtest_callback, reduce_lr]

    # Only enable EarlyStopping if explicitly requested in params
    use_early_stopping = params.get('early_stopping', False)
    if use_early_stopping:
        patience = params.get('patience', 15)
        early_stopping = EarlyStopping(
            monitor='loss',
            min_delta=1e-6,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        print(f"EarlyStopping enabled (monitoring 'loss', min_delta=1e-6, patience={patience}).")
    else:
        print(f"EarlyStopping disabled: training will run for the full {epochs} epochs (best weights saved via ModelCheckpoint).")

    # Put UI logger callback last so on_train_end fires after backtest_callback.on_train_end
    callbacks_list.append(ui_callback)

    print(f"Fitting MLP model for {epochs} epochs...")
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )

    # Explicitly save final restored model weights to best_model.keras
    model.save('best_model.keras')
    print("Model training complete. Best model weights saved to best_model.keras.")

