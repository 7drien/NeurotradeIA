import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model # Correct import
import joblib
from src.data_loader import get_data
from src.preprocessing import create_sequences_triple_label
from src.model import opportunity_cost_loss
from src.config import (
    TICKER, INTERVAL, N_STEPS, K_STEPS,
    INITIAL_CAPITAL, TRANSACTION_COST, ATR_MULTIPLIER, TRAILING_STOP_PCT,
    CONFIRMATION_PERIOD, REGIME_FILTER_PERIOD, RISK_PER_TRADE_PCT,
    PROFIT_TAKE_FACTOR, STOP_LOSS_FACTOR, MAX_HOLDING_PERIOD
)

def _calculate_indicators(df, atr_period_labeling=14):
    """Calculates all necessary indicators for the model and the strategy."""
    # ATR for risk management
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean() # This ATR is for strategy
    
    # ATR for Triple Barrier labeling
    df['ATR_label'] = true_range.rolling(atr_period_labeling).mean()
    
    # Long-term moving average for market regime filter
    df['MA_long'] = df['Close'].rolling(window=REGIME_FILTER_PERIOD).mean()
    
    return df

def simulate_backtest(params=None):
    """
    Performs the backtest with Triple Barrier Method labels and regime filter.
    """
    if params is None:
        params = {}
    n_steps = params.get('n_steps', N_STEPS)
    days_to_load = params.get('days_to_load', None)
    
    print("Loading data...")
    import datetime
    from src.config import START_DATE_STR
    start_date_str = START_DATE_STR
    if days_to_load is not None:
        start_date = datetime.datetime.now() - datetime.timedelta(days=days_to_load)
        start_date_str = start_date.strftime('%Y-%m-%d')
        
    data = get_data(ticker=TICKER, interval=INTERVAL, start_date=start_date_str)
    if data.empty: return None

    # Calculate all indicators on the original data
    df_with_all_indicators = _calculate_indicators(data.copy())
    
    from src.features import add_advanced_features
    df_with_features = add_advanced_features(df_with_all_indicators.copy())
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Close_FracDiff', 'GK_Vol_14', 'Rolling_VWAP', 'RSI_14']
    all_needed_cols = list(set(feature_cols + ['ATR', 'MA_long', 'ATR_label', 'High', 'Low', 'Close']))
    df_processed = df_with_features[all_needed_cols].copy()
    df_processed.dropna(inplace=True)
    
    try:
        scaler = joblib.load('scaler.joblib')
        prices_for_labels = df_processed['Close'].values
        highs_for_labels = df_processed['High'].values
        lows_for_labels = df_processed['Low'].values
        atrs_for_labeling = df_processed['ATR_label'].values

        scaled_features = scaler.transform(df_processed[feature_cols])
    except FileNotFoundError:
        print("ERROR: scaler.joblib not found.")
        return None

    X, y_true, sample_weights = create_sequences_triple_label(
        scaled_features, 
        prices_for_labels, 
        highs_for_labels,
        lows_for_labels, 
        atrs_for_labeling, 
        n_steps, 
        K_STEPS
    )
    if len(X) == 0: return None

    print("Loading model...")
    try:
        custom_objects = {'opportunity_cost_loss': opportunity_cost_loss}
        model = load_model('best_model.keras', custom_objects=custom_objects)
    except IOError:
        print("Backtesting failed: Model file not found.")
        return None

    train_size = int(len(X) * 0.8)
    X_test = X[train_size:]
    
    test_start_index_in_df = train_size + n_steps - 1
    end_slice = test_start_index_in_df + len(X_test)
    test_indices = df_processed.index[test_start_index_in_df:end_slice]
    
    print("Making predictions (Meta-Labeling)...")
    predictions = model.predict(X_test)
    ml_confidence = predictions[:, 1]
    ml_approve = pd.Series(ml_confidence > 0.5, index=test_indices)

    # --- VectorBT Institutional Backtester ---
    import vectorbt as vbt
    
    price = df_processed['Close'].loc[test_indices]
    
    # 1. Primary Model (RSI + Z-Score)
    # We must calculate this on the full df to avoid NaNs at the beginning of test_indices
    z_score_all = (df_processed['Close'] - df_processed['Close'].rolling(100).mean()) / df_processed['Close'].rolling(100).std()
    
    # RSI is already computed in df_processed as 'RSI_14' from features.py!
    rsi_all = df_processed['RSI_14']
    
    cond_long_all = (z_score_all < -2) & (rsi_all < 30)
    cond_short_all = (z_score_all > 2) & (rsi_all > 70)
    
    cross_long_all = cond_long_all & ~cond_long_all.shift(1).fillna(False)
    cross_short_all = cond_short_all & ~cond_short_all.shift(1).fillna(False)
    
    # Slice to test_indices
    ma_cross_long = cross_long_all.loc[test_indices]
    ma_cross_short = cross_short_all.loc[test_indices]
    
    # 2. Filter with ML Predictions
    entries = ma_cross_long & ml_approve
    short_entries = ma_cross_short & ml_approve
    
    # 3. Dynamic ATR Stops (TP & SL percentages)
    atr = df_processed['ATR'].loc[test_indices]
    sl_pct = (atr * STOP_LOSS_FACTOR) / price
    tp_pct = (atr * PROFIT_TAKE_FACTOR) / price
    
    # 4. Time Stops (MAX_HOLDING_PERIOD)
    exits = entries.vbt.signals.fshift(MAX_HOLDING_PERIOD)
    short_exits = short_entries.vbt.signals.fshift(MAX_HOLDING_PERIOD)
    
    # 5. Run VectorBT Portfolio Simulation
    print("Running VectorBT Simulation...")
    pf = vbt.Portfolio.from_signals(
        price,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        sl_stop=sl_pct.values,
        tp_stop=tp_pct.values,
        fees=TRANSACTION_COST,
        init_cash=INITIAL_CAPITAL,
        freq='15min'
    )
    
    return {
        "portfolio": pf,
        "initial_capital": INITIAL_CAPITAL,
    }

def plot_backtest_results(results, fig=None, ax1=None, ax2=None):
    if not results: return
    
    pf = results["portfolio"]
    initial_capital = results["initial_capital"]
    
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        new_figure = True
    else:
        new_figure = False
        ax1.clear(); ax2.clear()
        
    test_indices = pf.close.index
    actual_prices = pf.close.values
    portfolio_value = pf.value().values
    
    ax1.plot(test_indices, actual_prices, label='Actual Price', color='blue', zorder=1)
    
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
            ax1.scatter(entries_idx[long_mask], entry_prices[long_mask], label='Open Long', marker='^', color='green', s=120, zorder=5)
            ax1.scatter(exit_idx[long_mask], exit_prices[long_mask], label='Close Long', marker='x', color='green', s=120, zorder=5)
            
        if short_mask.any():
            ax1.scatter(entries_idx[short_mask], entry_prices[short_mask], label='Open Short', marker='v', color='red', s=120, zorder=5)
            ax1.scatter(exit_idx[short_mask], exit_prices[short_mask], label='Close Short', marker='x', color='red', s=120, zorder=5)
            
    ax1.set_title('VectorBT Meta-Labeling Backtest'); ax1.set_ylabel('Price (USD)'); ax1.legend(); ax1.grid(True)
    ax2.plot(test_indices, portfolio_value, label='Portfolio Value', color='purple'); ax2.set_title('Portfolio Value Over Time'); ax2.set_ylabel('Portfolio Value (USD)'); ax2.set_xlabel('Date'); ax2.grid(True)
    fig.tight_layout()
    if new_figure: plt.show()
    
    final_value = portfolio_value[-1]
    returns = (final_value - initial_capital) / initial_capital * 100
    
    print(f"\n--- VectorBT Backtest Results ---")
    print(f"Total Return: {returns:.2f}% | Win Rate: {pf.trades.win_rate() * 100:.2f}% | Max Drawdown: {pf.max_drawdown() * 100:.2f}%")
