from datetime import datetime, timedelta

# --- Data parameters ---
TICKER = "BTC-USD"
INTERVAL = "1h"
DAYS_TO_LOAD = 700 # ~2 years of data for better generalization

END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=DAYS_TO_LOAD)
END_DATE_STR = END_DATE.strftime('%Y-%m-%d')
START_DATE_STR = START_DATE.strftime('%Y-%m-%d')

# --- Preprocessing parameters ---
N_STEPS = 128 # 128 consecutive candles (input segment)
K_STEPS = 32  # Prediction horizon (predict price 32 candles ahead)

# Prediction & Strategy parameters (Dual-Threshold Oscillator Strategy)
DEFAULT_X_ENTRY = 0.01  # Entry threshold: Buy > 1 + 0.01 (1.010), Sell < 1 - 0.01 (0.990)
DEFAULT_X_EXIT = 0.005  # Exit threshold: Exit Long < 1 + 0.005 (1.005), Exit Short > 1 - 0.005 (0.995)
PREDICTION_BUY_THRESHOLD = 1.0 + DEFAULT_X_ENTRY
PREDICTION_SELL_THRESHOLD = 1.0 - DEFAULT_X_ENTRY
PROFIT_TAKE_FACTOR = 2.0          # Backward compatibility
STOP_LOSS_FACTOR = 1.0            # Backward compatibility
MAX_HOLDING_PERIOD = 32           # Backward compatibility

# --- Model parameters ---
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01 # Increased learning rate (was 0.005)

# --- Backtesting parameters ---
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001

# --- Strategy Filters ---
CONFIRMATION_PERIOD = 3 # Increased to 3 to reduce noise trades
REGIME_FILTER_PERIOD = 800 # Increased for a longer-term trend (800 * 15min = 200 hours = ~8 days)

# --- Risk Management ---
RISK_PER_TRADE_PCT = 0.01 # Risk 1% of capital per trade
ATR_MULTIPLIER = 2.0
TRAILING_STOP_PCT = 0.03
