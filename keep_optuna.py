import pandas as pd
import optuna
import numpy as np
import time as time

import psycopg2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from KeepTradingEnv import KeepTradingEnv
import pandas.io.sql as psql


def get_keep_data():
    conn = psycopg2.connect(database="keep_data", user="postgres", password="postgres", host="localhost", port="5432")
    market_data = psql.read_sql("Select * from keep_info", conn)
    conn.close()
    return market_data


df = get_keep_data()
df = df.sort_values('index')
df.drop(['index'], axis=1, inplace=True)
df = df.astype(np.float64)
dfTest = df
n_cpu = 2


def optimize_ppo2(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }


def optimize_agent(trial):
    print("Start Optuna Agent")
    time_statr = time.time()
    model_params = optimize_ppo2(trial)
    env = DummyVecEnv([lambda: KeepTradingEnv(df)])
    env_test = DummyVecEnv([lambda: KeepTradingEnv(dfTest)])
    model = PPO2(MlpLnLstmPolicy, env, verbose=0, nminibatches=1, **model_params)
    model.learn(180)

    rewards = []
    n_episodes, reward_sum = 0, 0.0

    obs = env_test.reset()
    while n_episodes < 90:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env_test.step(action)
        reward_sum += reward

        if all(done):
            rewards.append(reward_sum)
            reward_sum = 0.0
            n_episodes += 1
            obs = env_test.reset()

    last_reward = np.mean(rewards)
    trial.report(-1 * last_reward)
    print("End Optuna Agent, time work:", int(time.time() - time_statr), 's')
    return -1 * last_reward


study = optuna.create_study(study_name='cartpol_optuna', storage='sqlite:///params.db', load_if_exists=True)
study.optimize(optimize_agent, n_trials=50, n_jobs=1)
print(f'Finished trials: {len(study.trials)}')
print(f'Best trial: {study.best_trial.value}')
print('Params: ')
for key, value in study.best_trial.params.items():
    print(f'    {key}: {value}')
print(study.best_params)
