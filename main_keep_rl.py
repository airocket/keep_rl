import optuna
import psycopg2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.pyplot as plt
import os
from os import path
import platform
import time
import tensorflow as tf
import numpy as np
import pandas.io.sql as psql
import pandas as pd
import logging
import pprint
tf.get_logger().setLevel(logging.ERROR)

from KeepTradingEnv_new2 import KeepTradingEnv


def get_keep_data():
    conn = psycopg2.connect(database="keep_data", user="postgres", password="postgres", host="localhost", port="5432")
    market_data = psql.read_sql("Select * from keep_info", conn)
    conn.close()
    return market_data


# df = get_keep_data()
# df = df.sort_values('index')
# df.drop(['index'], axis=1, inplace=True)

df = pd.read_csv('keep_info.csv')
df.drop(['index'], axis=1, inplace=True)
df = df.astype(np.float64)
dfTest = df
print(df.tail())

if __name__ == '__main__':
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


    def get_model_params():
        params = study.best_trial.params
        return {
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'ent_coef': params['ent_coef'],
            'cliprange': params['cliprange'],
            'noptepochs': int(params['noptepochs']),
            'lam': params['lam'],
        }


    study = optuna.create_study(study_name='cartpol_optuna', storage='sqlite:///params_final.db', load_if_exists=True)
    model_params = get_model_params()
    model_params['learning_rate'] = model_params['learning_rate']/10
    model_params['gamma'] = 0.99
    print('Imput DataFrame:')
    print(df.head())
    print(df.tail())
    pprint.pprint(model_params)

    n_cpu = 8
    learn_timesteps = 30000
    test_timesteps = 4000

    env = SubprocVecEnv([make_env(i) for i in range(n_cpu)])  # задаем колл-во процессоров
    test_env = DummyVecEnv([make_envTest(i) for i in range(1)])
    policy_kwargs1 = dict(n_lstm=512)

    model = PPO2(MlpLnLstmPolicy, env, nminibatches=1, verbose=1, n_steps=50, tensorboard_log="./tensorboard_keep/")
    #model = PPO2.load("model/Linux/model_epoch_14.pkl",env, nminibatches=1, verbose=1, n_steps=49, tensorboard_log="./tensorboard_keep/", policy_kwargs=policy_kwargs1, **model_params)
    model.is_tb_set = True

    for n_epoch in range(0, 50):
        summary_writer = tf.compat.v1.summary.FileWriter("./tensorboard_keep/" + "Keep_trade_test_" + str(n_epoch+1))
        print('\x1b[6;30;42m' + '**************  Calculate epoch:', n_epoch, '**************' + '\x1b[0m')
        time_learn = time.time()
        save_s = model.learn(total_timesteps=learn_timesteps, tb_log_name='Keep_learn')
        delta_time_learn = time.time() - time_learn
        print(datetime.now(), 'Learn end', ' epoch:', n_epoch,' steps:', learn_timesteps, ' time to work:', int((delta_time_learn) / 60), 'min', ' FPS: ', round(learn_timesteps / delta_time_learn,1))
        print()
        zero_completed_obs = np.zeros((n_cpu,) + env.observation_space.shape)
        zero_completed_obs[0, :] = test_env.reset()
        state = None
        reward_sum = 0
        last_done_net_worth = 0
        all_net_worth_np = np.array([])
        time_test = time.time()
        for i in range(test_timesteps):
            action, states = model.predict(zero_completed_obs, state=state)
            obs, reward, done, info = test_env.step(action)
            zero_completed_obs[0, :] = obs
            reward_sum += reward[0]
            if done[0]:
                net_worth = info[0]['net_worth']
                last_done_net_worth = net_worth
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag='test_episode_reward/' + str(n_epoch + 1), simple_value=reward_sum)])
                summary_writer.add_summary(summary, i)
                all_net_worth_np = np.append(all_net_worth_np, [net_worth])
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag='net_worth/' + str(n_epoch + 1), simple_value=net_worth)])
                summary_writer.add_summary(summary, i)
                reward_sum = 0
                zero_completed_obs[0, :] = test_env.reset()
                state = None

        average_net_worth = int(np.sum([all_net_worth_np]) / len(all_net_worth_np))
        delta_time_test = time.time() - time_test
        print(datetime.now(), 'Test end', ' epoch:', n_epoch,' steps:', test_timesteps, ' time to work:', int((delta_time_test) / 60), 'min', ' FPS: ', round(learn_timesteps / delta_time_test,1))
        print(datetime.now(), "Last net worth test: ", round(last_done_net_worth), 'Average net_worth test:', round(average_net_worth))
        summary_writer.close()
        model_path = path.join('model', platform.system(), f'model_epoch_{n_epoch}.pkl')
        model.save(model_path)
        time.sleep(10)
        print(datetime.now(), 'Model saved to :', model_path)

    time.sleep(10)
