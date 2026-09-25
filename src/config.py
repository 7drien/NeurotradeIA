from datetime import datetime, timedelta

# --- Data parameters ---
TICKER = "BTC-USD"
INTERVAL = "15m"
DAYS_TO_LOAD = 180 # Increased to 180 days for more diverse market conditions

END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=DAYS_TO_LOAD)
END_DATE_STR = END_DATE.strftime('%Y-%m-%d')
START_DATE_STR = START_DATE.strftime('%Y-%m-%d')

# --- Preprocessing parameters ---
N_STEPS = 100
K_STEPS = 4 # Prediction horizon (e.g., 4 * 15min = 1 hour ahead)

# Triple Barrier Method parameters
PROFIT_TAKE_FACTOR = 2.0 # Take profit when price moves 2 * ATR
STOP_LOSS_FACTOR = 1.0   # Stop loss when price moves 1 * ATR
MAX_HOLDING_PERIOD = 10  # Max 10 candles (2.5 hours) holding period

# --- Model parameters ---
EPOCHS = 50
BATCH_SIZE = 32

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
