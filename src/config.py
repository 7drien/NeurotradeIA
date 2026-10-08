from datetime import datetime, timedelta

# --- Data parameters ---
BTC_TICKER = "BTC-USD"
ETH_TICKER = "ETH-USD"
BTC_START_DATE = "2016-01-01"
ETH_START_DATE = "2018-01-01"

TRAIN_TICKERS = [BTC_TICKER, ETH_TICKER]
TICKER_START_DATES = {
    BTC_TICKER: BTC_START_DATE,
    ETH_TICKER: ETH_START_DATE
}

TICKER = BTC_TICKER
INTERVAL = "1h"
DAYS_TO_LOAD = None # None loads full history from start dates (BTC 2016, ETH 2018)

END_DATE = datetime.now()
START_DATE = datetime.strptime(BTC_START_DATE, '%Y-%m-%d')
END_DATE_STR = END_DATE.strftime('%Y-%m-%d')
START_DATE_STR = BTC_START_DATE

# --- Preprocessing parameters ---
N_STEPS = 256 # 256 consecutive candles (input segment: 256 * 8 = 2048 features)
K_STEPS = 32  # Prediction horizon (predict price 32 candles ahead)

# Prediction & Strategy parameters (Dual-Threshold Oscillator Strategy)
DEFAULT_X_ENTRY = 0.05  # Entry threshold: Buy > 1 + 0.05 (1.050), Sell < 1 - 0.05 (0.950)
DEFAULT_X_EXIT = -0.02  # Exit threshold: Exit Long < 1 + (-0.02) (0.980), Exit Short > 1 - (-0.02) (1.020)
PREDICTION_BUY_THRESHOLD = 1.0 + DEFAULT_X_ENTRY
PREDICTION_SELL_THRESHOLD = 1.0 - DEFAULT_X_ENTRY
PROFIT_TAKE_FACTOR = 2.0          # Backward compatibility
STOP_LOSS_FACTOR = 1.0            # Backward compatibility
MAX_HOLDING_PERIOD = 32           # Backward compatibility

# --- Model parameters ---
EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.0003 # Optimal learning rate for deep MLP with bounded sigmoid output
DEFAULT_N_MODELS = 3   # Default number of models in ensemble (each manages 1/N capital)
N_MODELS = DEFAULT_N_MODELS

# --- Backtesting parameters ---
INITIAL_CAPITAL = 10000.0
DEFAULT_FEES_PCT = 0.1 # Default fee percentage: 0.1% per trade (standard crypto spot fee)
TRANSACTION_COST = DEFAULT_FEES_PCT / 100.0 # 0.001 (decimal)

# --- Strategy Filters ---
CONFIRMATION_PERIOD = 3 # Increased to 3 to reduce noise trades
REGIME_FILTER_PERIOD = 800 # Increased for a longer-term trend (800 * 15min = 200 hours = ~8 days)

# --- Risk Management ---
RISK_PER_TRADE_PCT = 0.01 # Risk 1% of capital per trade
ATR_MULTIPLIER = 2.0
TRAILING_STOP_PCT = 0.03
