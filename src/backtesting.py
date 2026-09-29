import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from src.data_loader import get_data
from src.features import add_advanced_features
from src.preprocessing import create_sequences
from src.model import two_sigmoid, directional_accuracy, opportunity_cost_loss
from src.config import (
    TICKER, INTERVAL, N_STEPS, K_STEPS,
    INITIAL_CAPITAL, TRANSACTION_COST,
    START_DATE_STR
)

PREDICTIONS_CACHE_FILE = "data/cached/model_predictions.pkl"
_CACHED_PREDICTIONS = None

def get_or_compute_predictions(params=None, force_recompute=False):
    """
    Retrieves or calculates model predictions for each candle.
    Saves outputs to disk (data/cached/model_predictions.pkl) so backtests can be
    recalculated with different thresholds instantaneously without retraining.
    """
    global _CACHED_PREDICTIONS
    
    if not force_recompute:
        if _CACHED_PREDICTIONS is not None:
            return _CACHED_PREDICTIONS
        if os.path.exists(PREDICTIONS_CACHE_FILE):
            try:
                print(f"Loading cached predictions from: {PREDICTIONS_CACHE_FILE}")
                _CACHED_PREDICTIONS = pd.read_pickle(PREDICTIONS_CACHE_FILE)
                return _CACHED_PREDICTIONS
            except Exception as e:
                print(f"Could not read cached predictions: {e}. Recomputing...")

    if params is None:
        params = {}
    n_steps = params.get('n_steps', N_STEPS)
    k_steps = params.get('k_steps', K_STEPS)
    days_to_load = params.get('days_to_load', None)
    
    start_date_str = START_DATE_STR
    if days_to_load is not None:
        start_date = datetime.datetime.now() - datetime.timedelta(days=days_to_load)
        start_date_str = start_date.strftime('%Y-%m-%d')
        
    data = get_data(ticker=TICKER, interval=INTERVAL, start_date=start_date_str)
    if data.empty:
        return None

    df_processed = add_advanced_features(data.copy())
    X, y_true = create_sequences(df_processed, n_steps=n_steps, k_steps=k_steps)
    if len(X) == 0:
        return None

    print("Loading best model for inference...")
    try:
        custom_objects = {
            'two_sigmoid': two_sigmoid,
            'directional_accuracy': directional_accuracy,
            'opportunity_cost_loss': opportunity_cost_loss
        }
        model = load_model('best_model.keras', custom_objects=custom_objects)
    except Exception as e:
        print(f"Backtesting failed: Model file could not be loaded ({e})")
        return None

    train_size = int(len(X) * 0.8)
    X_test = X[train_size:]
    
    test_start_index = train_size + n_steps - 1
    end_slice = test_start_index + len(X_test)
    test_indices = df_processed.index[test_start_index:end_slice]
    
    print(f"Computing model output oscillator for {len(X_test)} candles...")
    preds = model.predict(X_test, verbose=0).flatten()

    price = df_processed['Close'].loc[test_indices]
    pred_series = pd.Series(preds, index=test_indices)

    cache_data = {
        "price": price,
        "preds": pred_series,
        "test_indices": test_indices,
        "y_true": y_true[train_size:],
        "n_steps": n_steps,
        "k_steps": k_steps
    }

    _CACHED_PREDICTIONS = cache_data
    os.makedirs(os.path.dirname(PREDICTIONS_CACHE_FILE), exist_ok=True)
    pd.to_pickle(cache_data, PREDICTIONS_CACHE_FILE)
    print(f"Saved {len(preds)} candle predictions to {PREDICTIONS_CACHE_FILE}")

    return cache_data

def run_backtest_with_threshold(threshold_x=0.002, predictions_data=None, k_steps=None):
    """
    Runs a fast VectorBT backtest based on precomputed model outputs:
    - BUY (Long) when output > 1.0 + x
    - SELL (Short) when output < 1.0 - x
    - EXIT after k_steps (16 candles)
    Takes < 50ms without retraining.
    """
    if predictions_data is None:
        predictions_data = get_or_compute_predictions()
    if predictions_data is None:
        return None

    import vectorbt as vbt

    price = predictions_data["price"]
    preds = predictions_data["preds"]
    test_indices = predictions_data["test_indices"]
    if k_steps is None:
        k_steps = predictions_data.get("k_steps", K_STEPS)

    buy_threshold = 1.0 + threshold_x
    sell_threshold = 1.0 - threshold_x

    entries = preds > buy_threshold
    short_entries = preds < sell_threshold

    # Hold duration = k_steps candles
    exits = entries.vbt.signals.fshift(k_steps)
    short_exits = short_entries.vbt.signals.fshift(k_steps)

    pf = vbt.Portfolio.from_signals(
        price,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        fees=TRANSACTION_COST,
        init_cash=INITIAL_CAPITAL,
        freq='1h'
    )
    
    return {
        "portfolio": pf,
        "initial_capital": INITIAL_CAPITAL,
        "price": price,
        "preds": preds,
        "test_indices": test_indices,
        "threshold_x": threshold_x,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "k_steps": k_steps
    }

def simulate_backtest(params=None):
    """
    Executes backtest with latest model checkpoint (called during training or manually).
    """
    if params is None:
        params = {}
    threshold_x = params.get('threshold_x', 0.002)
    predictions_data = get_or_compute_predictions(params=params, force_recompute=True)
    if predictions_data is None:
        return None
    return run_backtest_with_threshold(threshold_x=threshold_x, predictions_data=predictions_data)

def plot_backtest_results(results, fig=None, ax1=None, ax2=None, ax3=None):
    """
    Plots Price with buy/sell signals, the Model Oscillator with 1±x thresholds,
    and the Portfolio Equity curve.
    """
    if not results:
        return
    
    pf = results["portfolio"]
    initial_capital = results["initial_capital"]
    preds_series = results.get("preds", None)
    threshold_x = results.get("threshold_x", 0.002)
    buy_threshold = results.get("buy_threshold", 1.0 + threshold_x)
    sell_threshold = results.get("sell_threshold", 1.0 - threshold_x)
    
    if fig is None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2.5, 1.2, 1.2]}, sharex=True)
        new_figure = True
    else:
        new_figure = False
        ax1.clear()
        ax2.clear()
        if ax3 is not None:
            ax3.clear()
        
    test_indices = pf.close.index
    actual_prices = pf.close.values
    portfolio_value = pf.value().values
    
    # --- 1. Price and Trade Signals ---
    ax1.plot(test_indices, actual_prices, label='Price (USD)', color='#4ba3e3', linewidth=1.2, zorder=1)
    
    trades = pf.trades
    if trades.count() > 0:
        records = trades.records_readable
        entries_idx = records.get('Entry Timestamp', records.get('Entry Index'))
        entry_prices = records.get('Avg Entry Price', records.get('Entry Price'))
        exit_idx = records.get('Exit Timestamp', records.get('Exit Index'))
        exit_prices = records.get('Avg Exit Price', records.get('Exit Price'))
        direction = records['Direction']
        
        long_mask = direction == 'Long'
        short_mask = direction == 'Short'
        
        if long_mask.any():
            ax1.scatter(entries_idx[long_mask], entry_prices[long_mask], label='Buy Long (> 1+x)', marker='^', color='#4caf50', s=90, zorder=5)
            ax1.scatter(exit_idx[long_mask], exit_prices[long_mask], label='Exit Long', marker='x', color='#81c784', s=70, zorder=5)
            
        if short_mask.any():
            ax1.scatter(entries_idx[short_mask], entry_prices[short_mask], label='Sell Short (< 1-x)', marker='v', color='#f44336', s=90, zorder=5)
            ax1.scatter(exit_idx[short_mask], exit_prices[short_mask], label='Exit Short', marker='x', color='#e57373', s=70, zorder=5)
            
    ax1.set_title(f'Market Price & Trade Executions (Threshold x = ±{threshold_x:.4f} | Total Trades: {trades.count()})', fontsize=12)
    ax1.set_ylabel('Price (USD)', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- 2. Model Oscillator with 1±x Thresholds ---
    if ax3 is not None and preds_series is not None:
        ax2.plot(test_indices, preds_series.values, label='Model Oscillator Output', color='#00d2ff', linewidth=1.0)
        ax2.axhline(1.0, color='#888888', linestyle=':', label='Neutral (1.0)', alpha=0.7)
        ax2.axhline(buy_threshold, color='#4caf50', linestyle='--', label=f'Buy Level (1+x = {buy_threshold:.4f})', alpha=0.9)
        ax2.axhline(sell_threshold, color='#f44336', linestyle='--', label=f'Sell Level (1-x = {sell_threshold:.4f})', alpha=0.9)
        ax2.set_title(f'128-Candle Oscillator Output with Thresholds x = ±{threshold_x:.4f}', fontsize=11)
        ax2.set_ylabel('Output', fontsize=10)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        equity_ax = ax3
    else:
        equity_ax = ax2

    # --- 3. Portfolio Equity Curve ---
    equity_ax.plot(test_indices, portfolio_value, label='Portfolio Equity', color='#ab47bc', linewidth=1.4)
    equity_ax.axhline(initial_capital, color='#888888', linestyle='--', label='Initial Capital', alpha=0.6)
    equity_ax.set_title('Portfolio Equity Over Time', fontsize=11)
    equity_ax.set_ylabel('Equity (USD)', fontsize=10)
    equity_ax.set_xlabel('Date', fontsize=10)
    equity_ax.legend(loc='upper left', fontsize=8)
    equity_ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    if new_figure:
        plt.show()
    
    final_value = portfolio_value[-1]
    returns = (final_value - initial_capital) / initial_capital * 100
    winrate = pf.trades.win_rate() * 100 if trades.count() > 0 else 0.0
    print(f"\n--- Backtest Results (x={threshold_x:.4f}) ---")
    print(f"Total Return: {returns:.2f}% | Win Rate: {winrate:.2f}% | Max Drawdown: {pf.max_drawdown() * 100:.2f}% | Trades: {trades.count()}")
