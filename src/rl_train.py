import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data_loader import get_data
from src.preprocessing import normalize_data
from src.trading_env import TradingEnv
from src.config import TRAIN_TICKER, INTERVAL, DAYS_TO_LOAD, N_STEPS, TRANSACTION_COST, HOLD_PENALTY

def train_rl_model():
    """
    Trains a Reinforcement Learning agent to trade.
    """
    data = get_data(ticker=TRAIN_TICKER, interval=INTERVAL, days_to_load=DAYS_TO_LOAD)
    if data.empty:
        print("Training stopped: No data available.")
        return None

    # --- Feature Engineering & Normalization ---
    df = data.copy()
    df['returns'] = df['Close'].pct_change()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    # ... (add other features like RSI)
    df.dropna(inplace=True)

    # Normalize features (except for 'Close' which is needed for reward calculation)
    feature_cols = ['Volume', 'returns', 'MA_10', 'MA_30'] # Example
    scaled_features, scaler = normalize_data(df[feature_cols])
    df[feature_cols] = scaled_features

    # Create the trading environment
    # The environment needs access to the dataframe with all data
    env_kwargs = {
        "df": df,
        "lookback_window": N_STEPS,
        "transaction_cost": TRANSACTION_COST,
        "hold_penalty": HOLD_PENALTY
    }
    env = DummyVecEnv([lambda: TradingEnv(**env_kwargs)])

    # Instantiate the agent (PPO is a robust algorithm)
    # The policy 'MlpPolicy' is a standard multi-layer perceptron, but for time series,
    # 'MlpLstmPolicy' could be even better.
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/")

    # Train the agent
    print("Starting RL agent training...")
    model.learn(total_timesteps=20000) # Number of steps to train for

    model.save("ppo_trading_model")
    return model

if __name__ == '__main__':
    train_rl_model()