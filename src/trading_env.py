import gymnasium as gym
import numpy as np
import pandas as pd
from src.config import HOLD_PENALTY, TRANSACTION_COST

class TradingEnv(gym.Env):
    """
    A custom trading environment for Reinforcement Learning.
    The agent learns a policy to maximize a reward function based on trading performance.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, lookback_window=60, transaction_cost=0.001, hold_penalty=0.0):
        super(TradingEnv, self).__init__()

        self.df = df.dropna().reset_index()
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.hold_penalty = hold_penalty

        # Define action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = gym.spaces.Discrete(3)

        # Define observation space: 'lookback_window' past data points
        # Shape: (lookback_window, number_of_features)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(lookback_window, self.df.shape[1]),
            dtype=np.float32
        )

    def _get_observation(self):
        # Get the last 'lookback_window' rows of data
        return self.df.iloc[self.current_step - self.lookback_window + 1 : self.current_step + 1].values

    def _calculate_reward(self, action_executed):
        # The primary reward is the change in portfolio value
        current_portfolio_value = self.balance + self.shares_held * self.df.loc[self.current_step, 'Close']
        reward = current_portfolio_value - self.prev_portfolio_value
        self.prev_portfolio_value = current_portfolio_value

        # Apply penalty for holding
        if not action_executed:
            reward += self.hold_penalty

        return reward 

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_trades = 0
        self.prev_portfolio_value = self.initial_balance
        
        # Start at a random point in the dataframe to see different scenarios
        # We need at least 'lookback_window' of history
        self.current_step = np.random.randint(
            self.lookback_window, len(self.df) - 1
        )

        return self._get_observation(), {}

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        action_executed = False
        
        # Execute action
        if action == 1: # Buy
            # Buy one share for simplicity
            if self.balance > current_price:
                self.balance -= current_price * (1 + self.transaction_cost)
                self.shares_held += 1
                self.total_trades += 1
                action_executed = True
        elif action == 2: # Sell
            if self.shares_held > 0:
                self.balance += current_price * (1 - self.transaction_cost)
                self.shares_held -= 1
                self.total_trades += 1
                action_executed = True

        # Move to the next time step
        self.current_step += 1

        # Calculate reward
        reward = self._calculate_reward(action_executed)

        # Check if the episode is done
        done = self.current_step >= len(self.df) - 1

        # For a more complex reward like Sharpe Ratio, you would calculate it here,
        # especially when `done` is True.
        # For now, we use a simpler, immediate reward.
        
        # The info dict can be used for logging
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_trades': self.total_trades,
            'portfolio_value': self.prev_portfolio_value
        }

        return self._get_observation(), reward, done, False, info

    def render(self, mode='human'):
        if mode == 'human':
            portfolio_value = self.balance + self.shares_held * self.df.loc[self.current_step, 'Close']
            print(
                f"Step: {self.current_step}, "
                f"Balance: {self.balance:.2f}, "
                f"Shares: {self.shares_held}, "
                f"Portfolio Value: {portfolio_value:.2f}, "
                f"Total Trades: {self.total_trades}"
            )