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
    CONFIRMATION_PERIOD, REGIME_FILTER_PERIOD, RISK_PER_TRADE_PCT
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

def simulate_backtest():
    """
    Performs the backtest with Triple Barrier Method labels and regime filter.
    """
    print("Loading data...")
    data = get_data(ticker=TICKER, interval=INTERVAL)
    if data.empty: return None

    # Calculate all indicators on the original data
    df_with_all_indicators = _calculate_indicators(data.copy())
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_processed = df_with_all_indicators[feature_cols + ['ATR', 'MA_long', 'ATR_label']].copy()
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

    X, y_true = create_sequences_triple_label(
        scaled_features, 
        prices_for_labels, 
        highs_for_labels,
        lows_for_labels, 
        atrs_for_labeling, 
        N_STEPS, 
        K_STEPS
    )
    if len(X) == 0: return None

    print("Loading model...")
    try:
        custom_objects = {'opportunity_cost_loss': opportunity_cost_loss}
        model = load_model('best_model.h5', custom_objects=custom_objects) # Corrected: load_model
    except IOError:
        print("Backtesting failed: Model file not found.")
        return None

    train_size = int(len(X) * 0.8)
    X_test = X[train_size:]
    
    test_start_index_in_df = train_size + N_STEPS - 1
    end_slice = test_start_index_in_df + len(X_test)
    test_indices = df_processed.index[test_start_index_in_df:end_slice]
    actual_prices = df_processed['Close'].loc[test_indices].values
    atr_values = df_processed['ATR'].loc[test_indices].values
    ma_long_values = df_processed['MA_long'].loc[test_indices].values
    
    print("Making predictions...")
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    # --- Backtesting Logic with Regime Filter and Confirmation ---
    capital = INITIAL_CAPITAL
    position_type = None
    position_size = 0.0
    entry_price = 0.0
    portfolio_value_list = [INITIAL_CAPITAL]
    signals = []
    stop_loss_price = 0
    trailing_stop_price = 0
    buy_confirmation_count = 0
    sell_confirmation_count = 0

    min_len_sim = min(len(actual_prices) - 1, len(predicted_labels))

    for i in range(min_len_sim):
        current_price = actual_prices[i]
        label = predicted_labels[i]
        is_uptrend = current_price > ma_long_values[i]

        # --- Risk Management ---
        if position_type == 'long':
            if current_price < stop_loss_price or current_price < trailing_stop_price:
                capital += position_size * current_price * (1 - TRANSACTION_COST)
                signals.append({'day': i, 'type': 'Close Long (Stop)'})
                position_type = None; position_size = 0.0
            else:
                trailing_stop_price = max(trailing_stop_price, current_price * (1 - TRAILING_STOP_PCT))
        
        elif position_type == 'short':
            if current_price > stop_loss_price or current_price > trailing_stop_price:
                capital -= position_size * current_price * (1 + TRANSACTION_COST)
                signals.append({'day': i, 'type': 'Close Short (Stop)'})
                position_type = None; position_size = 0.0
            else:
                trailing_stop_price = min(trailing_stop_price, current_price * (1 + TRAILING_STOP_PCT))

        # --- Confirmation and Trading Logic with Regime Filter ---
        if label == 2: buy_confirmation_count += 1; sell_confirmation_count = 0
        elif label == 0: sell_confirmation_count += 1; buy_confirmation_count = 0
        else: buy_confirmation_count = 0; sell_confirmation_count = 0

        if position_type is None:
            if is_uptrend and buy_confirmation_count >= CONFIRMATION_PERIOD:
                atr_at_buy = atr_values[i]
                if pd.notna(atr_at_buy):
                    risk_amount = capital * RISK_PER_TRADE_PCT
                    stop_loss_level = current_price - (atr_at_buy * ATR_MULTIPLIER)
                    price_diff_to_stop = current_price - stop_loss_level
                    
                    if price_diff_to_stop <= 0:
                        buy_confirmation_count = 0
                        continue

                    calculated_position_size = risk_amount / price_diff_to_stop
                    max_affordable_positions = (capital * (1 - TRANSACTION_COST)) / current_price
                    position_size = min(calculated_position_size, max_affordable_positions)
                    
                    if position_size * current_price * (1 + TRANSACTION_COST) > capital:
                        buy_confirmation_count = 0
                        continue

                    capital -= position_size * current_price * (1 + TRANSACTION_COST)
                    entry_price = current_price
                    position_type = 'long'
                    signals.append({'day': i, 'type': 'Open Long'})
                    stop_loss_price = stop_loss_level
                    trailing_stop_price = current_price * (1 - TRAILING_STOP_PCT)
                    buy_confirmation_count = 0
            elif not is_uptrend and sell_confirmation_count >= CONFIRMATION_PERIOD:
                atr_at_buy = atr_values[i]
                if pd.notna(atr_at_buy):
                    risk_amount = capital * RISK_PER_TRADE_PCT
                    stop_loss_level = current_price + (atr_at_buy * ATR_MULTIPLIER)
                    price_diff_to_stop = stop_loss_level - current_price
                    
                    if price_diff_to_stop <= 0:
                        sell_confirmation_count = 0
                        continue

                    calculated_position_size = risk_amount / price_diff_to_stop
                    max_shortable_positions = (capital * (1 - TRANSACTION_COST)) / current_price
                    position_size = min(calculated_position_size, max_shortable_positions)

                    if position_size * current_price * (1 + TRANSACTION_COST) > capital:
                        sell_confirmation_count = 0
                        continue

                    capital += position_size * current_price * (1 - TRANSACTION_COST)
                    entry_price = current_price
                    position_type = 'short'
                    signals.append({'day': i, 'type': 'Open Short'})
                    stop_loss_price = stop_loss_level
                    trailing_stop_price = current_price * (1 + TRAILING_STOP_PCT)
                    sell_confirmation_count = 0
        
        elif position_type == 'long' and sell_confirmation_count >= CONFIRMATION_PERIOD:
            capital += position_size * current_price * (1 - TRANSACTION_COST)
            signals.append({'day': i, 'type': 'Close Long'})
            position_type = None; position_size = 0.0
            
        elif position_type == 'short' and buy_confirmation_count >= CONFIRMATION_PERIOD:
            capital -= position_size * current_price * (1 + TRANSACTION_COST)
            signals.append({'day': i, 'type': 'Close Short'})
            position_type = None; position_size = 0.0

        # --- Portfolio Value Calculation ---
        if position_type == 'long':
            portfolio_value = capital + (position_size * current_price)
        elif position_type == 'short':
            portfolio_value = capital + (entry_price - current_price) * position_size
        else:
            portfolio_value = capital
        portfolio_value_list.append(portfolio_value)

    if not signals: print(">>> WARNING: No trades were executed.")
    
    portfolio_value = np.array(portfolio_value_list, dtype=float)
    min_len_plot = min(len(test_indices), len(portfolio_value))
    
    return {
        "test_indices": test_indices[:min_len_plot],
        "actual_prices": actual_prices[:min_len_plot],
        "signals": signals,
        "portfolio_value": portfolio_value[:min_len_plot],
        "initial_capital": INITIAL_CAPITAL,
    }

def plot_backtest_results(results, fig=None, ax1=None, ax2=None):
    if not results: return
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        new_figure = True
    else:
        new_figure = False
        ax1.clear(); ax2.clear()
    print("Plotting results...")
    test_indices = results['test_indices']
    actual_prices = results['actual_prices']
    signals = results['signals']
    portfolio_value = results['portfolio_value']
    initial_capital = results['initial_capital']
    ax1.plot(test_indices, actual_prices, label='Actual Price', color='blue', zorder=1)
    open_long_signals = [s for s in signals if s['type'] == 'Open Long']
    close_long_signals = [s for s in signals if s['type'] in ['Close Long', 'Close Long (Stop)']]
    open_short_signals = [s for s in signals if s['type'] == 'Open Short']
    close_short_signals = [s for s in signals if s['type'] in ['Close Short', 'Close Short (Stop)']]
    if open_long_signals: ax1.scatter(test_indices[[s['day'] for s in open_long_signals]], actual_prices[[s['day'] for s in open_long_signals]], label='Open Long', marker='^', color='green', s=120, zorder=5)
    if close_long_signals: ax1.scatter(test_indices[[s['day'] for s in close_long_signals]], actual_prices[[s['day'] for s in close_long_signals]], label='Close Long', marker='x', color='green', s=120, zorder=5)
    if open_short_signals: ax1.scatter(test_indices[[s['day'] for s in open_short_signals]], actual_prices[[s['day'] for s in open_short_signals]], label='Open Short', marker='v', color='red', s=120, zorder=5)
    if close_short_signals: ax1.scatter(test_indices[[s['day'] for s in close_short_signals]], actual_prices[[s['day'] for s in close_short_signals]], label='Close Short', marker='x', color='red', s=120, zorder=5)
    ax1.set_title('Trading Strategy Backtest'); ax1.set_ylabel('Price (USD)'); ax1.legend(); ax1.grid(True)
    ax2.plot(test_indices, portfolio_value, label='Portfolio Value', color='purple'); ax2.set_title('Portfolio Value Over Time'); ax2.set_ylabel('Portfolio Value (USD)'); ax2.set_xlabel('Date'); ax2.grid(True)
    fig.tight_layout()
    if new_figure: plt.show()
    final_value = portfolio_value[-1]
    returns = (final_value - initial_capital) / initial_capital * 100
    print(f"\nInitial Capital: ${initial_capital:,.2f}\nFinal Portfolio Value: ${final_value:,.2f}\nTotal Return: {returns:.2f}%")
