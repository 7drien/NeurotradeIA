import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from src.data_loader import get_data
from src.features import add_advanced_features
from src.preprocessing import create_sequences, create_train_test_split
from src.model import two_sigmoid, directional_accuracy, opportunity_cost_loss
from src.config import (
    TICKER, INTERVAL, N_STEPS, K_STEPS,
    INITIAL_CAPITAL, TRANSACTION_COST,
    DEFAULT_X_ENTRY, DEFAULT_X_EXIT,
    START_DATE_STR
)

PREDICTIONS_CACHE_FILE = "data/cached/model_predictions.pkl"
_CACHED_PREDICTIONS = None

def get_or_compute_predictions(params=None, force_recompute=False, model=None, test_data=None):
    """
    Retrieves or calculates model predictions for each candle.
    Saves outputs to disk (data/cached/model_predictions.pkl) so backtests can be
    recalculated with different thresholds instantaneously without retraining.
    Can use an in-memory model and precomputed test_data slice to avoid redundant data reloading.
    """
    global _CACHED_PREDICTIONS
    
    if not force_recompute and model is None and test_data is None:
        if _CACHED_PREDICTIONS is not None:
            return _CACHED_PREDICTIONS
        if os.path.exists(PREDICTIONS_CACHE_FILE):
            try:
                print(f"Loading cached predictions from: {PREDICTIONS_CACHE_FILE}")
                _CACHED_PREDICTIONS = pd.read_pickle(PREDICTIONS_CACHE_FILE)
                return _CACHED_PREDICTIONS
            except Exception as e:
                print(f"Could not read cached predictions: {e}. Recomputing...")

    if test_data is not None:
        X_test = test_data['X_test']
        price = test_data['price']
        test_indices = test_data['test_indices']
        y_true = test_data.get('y_true', None)
        n_steps = test_data.get('n_steps', N_STEPS)
        k_steps = test_data.get('k_steps', K_STEPS)
    else:
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
        _, _, _, _, test_info = create_train_test_split(
            df_processed, n_steps=n_steps, k_steps=k_steps, train_ratio=0.8
        )
        X_test = test_info['X_test']
        test_indices = test_info['test_indices']
        price = test_info['price']
        y_true = None

    if model is None:
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

    print(f"Computing model output oscillator for {len(X_test)} candles (20% end of BTC-USD)...")
    preds = model.predict(X_test, verbose=0).flatten()

    pred_series = pd.Series(preds, index=test_indices)

    cache_data = {
        "price": price,
        "preds": pred_series,
        "test_indices": test_indices,
        "y_true": y_true,
        "n_steps": n_steps,
        "k_steps": k_steps
    }

    _CACHED_PREDICTIONS = cache_data
    os.makedirs(os.path.dirname(PREDICTIONS_CACHE_FILE), exist_ok=True)
    pd.to_pickle(cache_data, PREDICTIONS_CACHE_FILE)
    print(f"Saved {len(preds)} candle predictions to {PREDICTIONS_CACHE_FILE}")

    return cache_data

def run_backtest_with_threshold(x_entry=DEFAULT_X_ENTRY, x_exit=DEFAULT_X_EXIT, predictions_data=None):
    """
    Runs a fast VectorBT backtest based on precomputed model outputs with dual thresholds:
    - Long Entry:  output > 1.0 + x_entry (e.g. > 1.010)
    - Long Exit:   output < 1.0 + x_exit  (e.g. < 1.005)
    - Short Entry: output < 1.0 - x_entry (e.g. < 0.990)
    - Short Exit:  output > 1.0 - x_exit  (e.g. > 0.995)
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

    buy_entry = 1.0 + x_entry
    buy_exit = 1.0 + x_exit
    sell_entry = 1.0 - x_entry
    sell_exit = 1.0 - x_exit

    # Dynamic threshold signals
    entries = preds > buy_entry
    exits = preds < buy_exit

    short_entries = preds < sell_entry
    short_exits = preds > sell_exit

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

    # Compute key performance statistics
    def _safe_float(val, default=0.0):
        try:
            f = float(val)
            return default if (np.isnan(f) or np.isinf(f)) else f
        except Exception:
            return default

    final_val = float(pf.value().iloc[-1])
    ret = _safe_float((final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0)
    bh_ret = _safe_float((price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100.0)
    sharpe = _safe_float(pf.sharpe_ratio())
    max_dd = _safe_float(pf.max_drawdown() * 100.0)
    total_trades = int(pf.trades.count())
    winrate = _safe_float(pf.trades.win_rate() * 100.0) if total_trades > 0 else 0.0
    profit_factor = _safe_float(pf.trades.profit_factor()) if total_trades > 0 else 0.0
    
    return {
        "portfolio": pf,
        "initial_capital": INITIAL_CAPITAL,
        "price": price,
        "preds": preds,
        "test_indices": test_indices,
        "x_entry": x_entry,
        "x_exit": x_exit,
        "buy_entry": buy_entry,
        "buy_exit": buy_exit,
        "sell_entry": sell_entry,
        "sell_exit": sell_exit,
        "returns": ret,
        "bh_return": bh_ret,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": winrate,
        "total_trades": total_trades,
        "profit_factor": profit_factor,
    }

def simulate_backtest(params=None, model=None, test_data=None):
    """
    Executes backtest with current model (or checkpoint) on the 20% test slice of BTC-USD.
    """
    if params is None:
        params = {}
    x_entry = params.get('x_entry', DEFAULT_X_ENTRY)
    x_exit = params.get('x_exit', DEFAULT_X_EXIT)
    predictions_data = get_or_compute_predictions(
        params=params, 
        force_recompute=True, 
        model=model, 
        test_data=test_data
    )
    if predictions_data is None:
        return None
    return run_backtest_with_threshold(x_entry=x_entry, x_exit=x_exit, predictions_data=predictions_data)

def plot_backtest_results(results, fig=None, ax1=None, ax2=None, ax3=None):
    """
    Plots Price with buy/sell signals, the Model Oscillator with entry/exit thresholds,
    and the Portfolio Equity curve with Buy & Hold comparison.
    """
    if not results:
        return
    
    pf = results["portfolio"]
    initial_capital = results["initial_capital"]
    preds_series = results.get("preds", None)
    x_entry = results.get("x_entry", DEFAULT_X_ENTRY)
    x_exit = results.get("x_exit", DEFAULT_X_EXIT)
    buy_entry = results.get("buy_entry", 1.0 + x_entry)
    buy_exit = results.get("buy_exit", 1.0 + x_exit)
    sell_entry = results.get("sell_entry", 1.0 - x_entry)
    sell_exit = results.get("sell_exit", 1.0 - x_exit)
    
    ret = results.get("returns", (pf.value().iloc[-1] - initial_capital) / initial_capital * 100.0)
    bh_ret = results.get("bh_return", (results["price"].iloc[-1] - results["price"].iloc[0]) / results["price"].iloc[0] * 100.0)
    sharpe = results.get("sharpe_ratio", 0.0)
    max_dd = results.get("max_drawdown", 0.0)
    winrate = results.get("win_rate", 0.0)
    profit_factor = results.get("profit_factor", 0.0)
    
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
            ax1.scatter(entries_idx[long_mask], entry_prices[long_mask], label=f'Buy Long (> {buy_entry:.3f})', marker='^', color='#4caf50', s=90, zorder=5)
            ax1.scatter(exit_idx[long_mask], exit_prices[long_mask], label=f'Exit Long (< {buy_exit:.3f})', marker='x', color='#81c784', s=70, zorder=5)
            
        if short_mask.any():
            ax1.scatter(entries_idx[short_mask], entry_prices[short_mask], label=f'Sell Short (< {sell_entry:.3f})', marker='v', color='#f44336', s=90, zorder=5)
            ax1.scatter(exit_idx[short_mask], exit_prices[short_mask], label=f'Exit Short (> {sell_exit:.3f})', marker='x', color='#e57373', s=70, zorder=5)
            
    ax1.set_title(f'Market Price & Trade Executions (Entry x={x_entry:.3f}, Exit x={x_exit:.3f} | Total Trades: {trades.count()})', fontsize=12)
    ax1.set_ylabel('Price (USD)', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- 2. Model Oscillator with 1±x_entry and 1±x_exit Thresholds ---
    if ax3 is not None and preds_series is not None:
        ax2.plot(test_indices, preds_series.values, label='Model Output Oscillator', color='#00d2ff', linewidth=1.0)
        ax2.axhline(buy_entry, color='#4caf50', linestyle='--', label=f'Buy Entry (1+x_in = {buy_entry:.4f})', alpha=0.9)
        ax2.axhline(buy_exit, color='#81c784', linestyle=':', label=f'Buy Exit (1+x_out = {buy_exit:.4f})', alpha=0.9)
        ax2.axhline(1.0, color='#888888', linestyle=':', label='Neutral (1.0)', alpha=0.6)
        ax2.axhline(sell_exit, color='#e57373', linestyle=':', label=f'Sell Exit (1-x_out = {sell_exit:.4f})', alpha=0.9)
        ax2.axhline(sell_entry, color='#f44336', linestyle='--', label=f'Sell Entry (1-x_in = {sell_entry:.4f})', alpha=0.9)
        ax2.set_title(f'128-Candle Oscillator with Entry/Exit Hysteresis Bands (x_in={x_entry:.3f}, x_out={x_exit:.3f})', fontsize=11)
        ax2.set_ylabel('Output', fontsize=10)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        equity_ax = ax3
    else:
        equity_ax = ax2

    # --- 3. Portfolio Equity vs Buy & Hold Curve ---
    bh_equity = (actual_prices / (actual_prices[0] + 1e-8)) * initial_capital

    equity_ax.plot(test_indices, portfolio_value, label=f'Strategy Equity ({ret:+.2f}%)', color='#ab47bc', linewidth=1.5, zorder=3)
    equity_ax.plot(test_indices, bh_equity, label=f'Buy & Hold BTC ({bh_ret:+.2f}%)', color='#f59e0b', linestyle='--', linewidth=1.2, alpha=0.85, zorder=2)
    equity_ax.axhline(initial_capital, color='#888888', linestyle=':', label='Initial Capital', alpha=0.5, zorder=1)
    equity_ax.set_title(f'Portfolio Equity vs Buy & Hold (Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2f}%)', fontsize=11)
    equity_ax.set_ylabel('Equity (USD)', fontsize=10)
    equity_ax.set_xlabel('Date', fontsize=10)
    equity_ax.legend(loc='upper left', fontsize=8)
    equity_ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    if new_figure:
        plt.show()
    
    print(f"\n--- Backtest Results (x_entry={x_entry:.4f}, x_exit={x_exit:.4f}) ---")
    print(f"Strategy Return: {ret:+.2f}% | Buy & Hold BTC: {bh_ret:+.2f}% | Sharpe: {sharpe:.2f}")
    print(f"Win Rate: {winrate:.2f}% | Max Drawdown: {max_dd:.2f}% | Trades: {trades.count()} | Profit Factor: {profit_factor:.2f}")
