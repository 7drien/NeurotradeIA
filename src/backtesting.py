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
    PREDICTION_BUY_THRESHOLD, PREDICTION_SELL_THRESHOLD,
    START_DATE_STR
)

def simulate_backtest(params=None):
    """
    Simulates a walk-forward backtest using predictions from the 4-layer MLP model.
    The model predicts future price ratio: Close[t+16] / Close[t].
    - Long signal: predicted ratio > PREDICTION_BUY_THRESHOLD (> 1.001)
    - Short signal: predicted ratio < PREDICTION_SELL_THRESHOLD (< 0.999)
    - Trade duration: 16 candles (matching prediction horizon)
    """
    if params is None:
        params = {}
    n_steps = params.get('n_steps', N_STEPS)
    k_steps = params.get('k_steps', K_STEPS)
    days_to_load = params.get('days_to_load', None)
    
    start_date_str = START_DATE_STR
    if days_to_load is not None:
        import datetime
        start_date = datetime.datetime.now() - datetime.timedelta(days=days_to_load)
        start_date_str = start_date.strftime('%Y-%m-%d')
        
    data = get_data(ticker=TICKER, interval=INTERVAL, start_date=start_date_str)
    if data.empty:
        return None

    df_processed = add_advanced_features(data.copy())
    X, y_true = create_sequences(df_processed, n_steps=n_steps, k_steps=k_steps)
    if len(X) == 0:
        return None

    print("Loading best model...")
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
    
    print("Making walk-forward predictions...")
    preds = model.predict(X_test, verbose=0).flatten()

    import vectorbt as vbt
    price = df_processed['Close'].loc[test_indices]
    pred_series = pd.Series(preds, index=test_indices)

    # Strategy signals:
    # Ratio > 1.001 -> Upward momentum expected in 16 candles
    # Ratio < 0.999 -> Downward momentum expected in 16 candles
    entries = pred_series > PREDICTION_BUY_THRESHOLD
    short_entries = pred_series < PREDICTION_SELL_THRESHOLD

    # Time exit: hold for k_steps (16 candles)
    exits = entries.vbt.signals.fshift(k_steps)
    short_exits = short_entries.vbt.signals.fshift(k_steps)

    print("Running VectorBT Simulation...")
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
        "y_true": y_true[train_size:],
        "y_pred": preds
    }

def plot_backtest_results(results, fig=None, ax1=None, ax2=None):
    if not results:
        return
    
    pf = results["portfolio"]
    initial_capital = results["initial_capital"]
    
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        new_figure = True
    else:
        new_figure = False
        ax1.clear()
        ax2.clear()
        
    test_indices = pf.close.index
    actual_prices = pf.close.values
    portfolio_value = pf.value().values
    
    ax1.plot(test_indices, actual_prices, label='Actual Price', color='#4ba3e3', zorder=1)
    
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
            ax1.scatter(entries_idx[long_mask], entry_prices[long_mask], label='Open Long', marker='^', color='#4caf50', s=120, zorder=5)
            ax1.scatter(exit_idx[long_mask], exit_prices[long_mask], label='Close Long', marker='x', color='#81c784', s=120, zorder=5)
            
        if short_mask.any():
            ax1.scatter(entries_idx[short_mask], entry_prices[short_mask], label='Open Short', marker='v', color='#f44336', s=120, zorder=5)
            ax1.scatter(exit_idx[short_mask], exit_prices[short_mask], label='Close Short', marker='x', color='#e57373', s=120, zorder=5)
            
    ax1.set_title('16-Candle MLP Walk-Forward Backtest')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(test_indices, portfolio_value, label='Portfolio Value', color='#ab47bc')
    ax2.set_title('Portfolio Value Over Time')
    ax2.set_ylabel('Value (USD)')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    
    fig.tight_layout()
    if new_figure:
        plt.show()
    
    final_value = portfolio_value[-1]
    returns = (final_value - initial_capital) / initial_capital * 100
    
    print(f"\n--- Backtest Results ---")
    print(f"Total Return: {returns:.2f}% | Win Rate: {pf.trades.win_rate() * 100:.2f}% | Max Drawdown: {pf.max_drawdown() * 100:.2f}%")
