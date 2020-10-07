import pandas as pd
import optuna
import numpy as np
import time as time


import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import psycopg2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from KeepTradingEnv_new2 import KeepTradingEnv
import pandas.io.sql as psql


def get_keep_data():
    conn = psycopg2.connect(database="keep_data", user="postgres", password="postgres", host="localhost", port="5432")
    market_data = psql.read_sql("Select * from keep_info", conn)
    conn.close()
    return market_data


# df = get_keep_data()
# df = df.sort_values('index')
# df.drop(['index'], axis=1, inplace=True)
# df = df.astype(np.float64)
# df.to_csv('keep_info.csv',index=False)

df = pd.read_csv('keep_info.csv')
df.drop(['index'], axis=1, inplace=True)
df = df.astype(np.float64)
dfTest = df


storage='sqlite:///params_final.db'
n_jobs = 8
n_trials = 150
learn_timesteps = 15000
test_epizod = 15



def optimize_ppo2(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 512)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 10e-5),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }



def optimize_agent(trial):

    print("Start Optuna Agent")
    time_statr = time.time()
    model_params = optimize_ppo2(trial)
    def make_envTest(rank, seed=0):

        def _init():
            env = KeepTradingEnv(dfTest)
            env.seed(seed + rank)
            return env

        set_global_seeds(seed)
        return _init

    def make_env(rank, seed=0):

        def _init():
            env = KeepTradingEnv(df)
            env.seed(seed + rank)
            return env

        set_global_seeds(seed)
        return _init

    n_cpu = 1
    env = SubprocVecEnv([make_env(i) for i in range(n_cpu)])  # задаем колл-во процессоров
    test_env = DummyVecEnv([make_envTest(i) for i in range(1)])
    model = PPO2(MlpLnLstmPolicy, env, verbose=0, nminibatches=1, **model_params)
    model.learn(learn_timesteps)
    rewards = []
    n_episodes, reward_sum = 0, 0.0
    zero_completed_obs = np.zeros((n_cpu,) + env.observation_space.shape)
    zero_completed_obs[0, :] = test_env.reset()
    state = None
    time_test = time.time()
    while n_episodes < test_epizod:
        action, state = model.predict(zero_completed_obs, state=state)
        obs, reward, done, info = test_env.step(action)
        zero_completed_obs[0, :] = obs
        reward_sum += reward[0]
        if done[0]:
            rewards.append(reward_sum)
            reward_sum = 0.0
            n_episodes += 1
            zero_completed_obs[0, :] = test_env.reset()
            state = None

    last_reward = np.mean(rewards)
    trial.report(-1 * last_reward)
    print("End Optuna Agent reward:",last_reward, "total time:", int((time.time() - time_statr)/60), 'min', ' test time:',int((time.time() - time_test)/60), 'min' )
    del env, test_env, model
    return -1 * last_reward

if __name__ == '__main__':
	print('Imput DataFrame:')
	print(df.head())
	print(df.tail())
	study = optuna.create_study(study_name='cartpol_optuna', storage=storage, load_if_exists=True)
	study.optimize(optimize_agent, n_trials=n_trials, n_jobs=n_jobs)
	print(f'Finished trials: {len(study.trials)}')
	print(f'Best trial: {study.best_trial.value}')
	print('Params: ')
	for key, value in study.best_trial.params.items():
	    print(f'    {key}: {value}')
	print(study.best_params)
