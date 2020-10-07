import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
from lib.transform import max_min_normalize, mean_normalize, log_and_difference, difference

# import platform
# import pyformulas as pf

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 1
INITIAL_ACCOUNT_BALANCE = 100000


class KeepTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(KeepTradingEnv, self).__init__()

        self.i = 0

        self.df = df

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1, 41), dtype=np.float32)
        self.observations = pd.DataFrame()
        self.scaled_history = pd.DataFrame()
        self.profit = 0
        self.close_keep = 0

    def _next_observation(self):

        frame = self.df.loc[self.current_step]

        self.observations = self.observations.append(frame, ignore_index=True)
        observations = log_and_difference(self.observations, inplace=False)
        observations = max_min_normalize(observations)
        obs = observations.values[-1]

        frame2 = pd.DataFrame([{
            'balance':self.balance,
            'max_net_worth':self.max_net_worth,
            'shares_held':self.shares_held,
            'cost_basis':self.cost_basis,
            'total_shares_sold':self.total_shares_sold,
            'total_hold':self.total_hold,
            'net_worth':self.account_history['net_worth'].values[-1],
            'current_step':self.current_step
        }])

        self.scaled_history = self.scaled_history.append(frame2, ignore_index=True)

        scaled_history = log_and_difference(self.scaled_history, inplace=False)
        scaled_history = max_min_normalize(scaled_history, inplace=False)

        history = scaled_history.values[-1]

        obs = np.append(obs,history, axis=0)
        obs = np.reshape(obs.astype('float32'), (1, 41))

        return obs

    def _take_action(self, action):
            self.current_action = action
            current_price = self.df.loc[self.current_step, "close_keep"]
            self.close_keep = current_price
            self.account_history = self.account_history.append([{
                'current_price': current_price,
                'reward': self.account_history['reward'].values[-1],
                'net_worth': self.account_history['net_worth'].values[-1],
                'sum_trades': self.account_history['sum_trades'].values[-1],
                'action': action,
            }], ignore_index=True)
            # # Set the current price to a random price within the time step
            # current_price = self.df.loc[self.current_step, "close_keep"]
            # self.account_history['current_price'].values[-1] = current_price
            self.profit = 0

            if action == 1:
                #total_possible = self.balance / current_price
                # shares_bought = total_possible
                # prev_cost = self.cost_basis * self.shares_held
                # additional_cost = shares_bought * current_price
                if self.total_hold == 0:
                    self.total_hold = self.balance / current_price
                    self.balance = 0
                    self.price_buy = current_price

                    self.trade_history = self.trade_history.append([{
                        'step_buy': self.current_step,
                        'price_buy': current_price,
                        'amount': self.total_hold,
                        'step_sell': 0,
                        'profit': 0,
                        'action': self.current_action

                    }], ignore_index=True)

            elif action == 2:
                if self.total_hold != 0:
                    self.balance = self.total_hold * current_price
                    self.profit = (current_price - self.price_buy) * self.total_hold
                    self.total_hold = 0


            # # Sell amount % of shares held
            #     shares_sold = self.shares_held
            #     self.balance += shares_sold * current_price
            #     self.shares_held -= shares_sold
            #     self.total_shares_sold += shares_sold
            #     self.total_sales_value += shares_sold * current_price
            #     self.profit = (self.trade_history['price_sell'].values[-1] - self.trade_history['price_buy'].values[-1]) * self.trade_history['amount'].values[-1]
            #     self.trade_history['profit'].values[-1] = self.profit
            #     self.trade_history['step_sell'].values[-1] = self.current_step
            #     self.trade_history['price_sell'].values[-1] = current_price

            #self.net_worth = self.balance + self.shares_held * self.price_buy
            self.net_worth = self.account_history['net_worth'].values[-1] + self.profit
            self.account_history['net_worth'].values[-1] = self.net_worth


    def step(self, action):

        self._take_action(action)

        self.current_step += 1
        reward = 0
        if self.account_history['net_worth'].values[-2] < self.account_history['net_worth'].values[-1]:
            reward = 1

        elif self.account_history['net_worth'].values[-2] > self.account_history['net_worth'].values[-1]:
            reward = -1
        if self.account_history['net_worth'].values[-2] == self.account_history['net_worth'].values[-1]:
            reward = 0

        if self.current_action == 1 and self.account_history['action'].values[-2] == 1:
            reward -= 0.1
        elif self.current_action == 2 and self.account_history['action'].values[-2] == 2:
            reward -= 0.1
        if len(self.account_history) > 15:
            if self.account_history['action'].values[-10:].sum() == 0:
                reward -= 2

        self.account_history['reward'].values[-1] = reward

        self.i = self.i + 1
        obs = self._next_observation()
        done = (self.current_step >= len(self.df)-1)

        return obs, reward, done, {'net_worth': self.account_history['net_worth'].values[-1],
                                   'action': self.account_history['action'].values[-1],
                                   'profit': self.profit,
                                   'balance': self.balance,
                                   'total_hold': self.total_hold}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.account_history = pd.DataFrame([{
            'current_price': 0,
            'reward': 0,
            'net_worth': INITIAL_ACCOUNT_BALANCE,
            'sum_trades': 0,
        }], dtype=np.float32)

        self.trade_history = pd.DataFrame([{
            'step_buy': 0,
            'step_sell': 0,
            'price_buy': 0,
            'price_sell': 0,
            'amount': 0,
            'profit': 0,
            'action' : 0
        }], dtype=np.float16)

        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = 0
        self.price_buy = 0
        self.current_action = 0
        self.total_hold = 0
        self.observations = pd.DataFrame()
        self.scaled_history = pd.DataFrame()
        self.close_keep = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        return self.account_history['net_worth'].values[-1], self.account_history['current_price'].values[-1]
