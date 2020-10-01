import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path

# import platform
# import pyformulas as pf

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 25000
INITIAL_ACCOUNT_BALANCE = 100000


class KeepTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(KeepTradingEnv, self).__init__()

        self.i = 0

        self.df = df

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1, 39), dtype=np.float64)

    def _next_observation(self):

        frame = self.df.loc[self.current_step].values

        obs = np.append(frame, [
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ], axis=0)
        obs = np.reshape(obs.astype('float64'), (1, 39))

        return obs

    def _take_action(self, action):
        self.current_action = action
        self.account_history = self.account_history.append([{
            'current_price': self.account_history['current_price'].values[-1],
            'reward': self.account_history['reward'].values[-1],
            'net_worth': self.account_history['net_worth'].values[-1],
            'sum_trades': self.account_history['sum_trades'].values[-1],
        }], ignore_index=True)
        # Set the current price to a random price within the time step
        current_price = self.df.loc[self.current_step, "close_keep"]
        self.account_history['current_price'].values[-1] = current_price

        if action == 1:

            total_possible = self.balance / current_price
            shares_bought = total_possible
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                                      prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought
            self.account_history['sum_trades'].values[-1] = self.account_history['sum_trades'].values[-1] + 1

        elif action == 2:
            # Sell amount % of shares held
            shares_sold = self.shares_held
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price
        self.account_history['net_worth'].values[-1] = self.net_worth

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):

        self._take_action(action)

        self.current_step += 1

        if self.account_history['net_worth'].values[-2] < self.account_history['net_worth'].values[-1]:
            reward = 1
        else:
            reward = -1
        if self.account_history['net_worth'].values[-2] == self.account_history['net_worth'].values[-1]:
            reward = 0

        self.account_history['reward'].values[-1] = reward

        self.i = self.i + 1
        obs = self._next_observation()
        done = (self.current_step >= len(self.df)-1) or (self.net_worth < INITIAL_ACCOUNT_BALANCE / 10)
        return obs, reward, done, {'net_worth': self.net_worth, 'action': self.current_action}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.account_history = pd.DataFrame([{
            'current_price': 0,
            'reward': 0,
            'net_worth': INITIAL_ACCOUNT_BALANCE,
            'sum_trades': 0,
        }], dtype=np.float32)

        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        return self.account_history['net_worth'].values[-1], self.account_history['current_price'].values[-1]
