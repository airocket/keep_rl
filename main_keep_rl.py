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

from KeepTradingEnv import KeepTradingEnv


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


    study = optuna.create_study(study_name='cartpol_optuna', storage='sqlite:///params.db', load_if_exists=True)
    model_params = get_model_params()
    print(model_params)

    n_cpu = 4

    env = SubprocVecEnv([make_env(i) for i in range(n_cpu)])  # задаем колл-во процессоров
    test_env = DummyVecEnv([make_envTest(i) for i in range(1)])

    model = PPO2(MlpLnLstmPolicy, env, nminibatches=1, verbose=1, n_steps=49, tensorboard_log="./tensorboard_keep/",
                 **model_params)
    # model = PPO2.load("model/Darwin/Model_Mac.pkl", nminibatches=n_cpu, env=env, verbose=1, n_steps=18, tensorboard_log="./tensorboard/")
    model.is_tb_set = True

    for n_epoch in range(0, 10):
        summary_writer = tf.compat.v1.summary.FileWriter("./tensorboard_keep/" + "Keep_trade_test_" + str(n_epoch+1))
        print('\x1b[6;30;42m' + '***************************  Calculate epoch:', n_epoch,
              '***************************' + '\x1b[0m')
        time_learn = time.time()
        save_s = model.learn(total_timesteps=20000, tb_log_name='Keep_learn')
        delta_time_learn = time.time() - time_learn
        print('Learn end', ' epoch:', n_epoch, ' time to work:', int((delta_time_learn) / 60), ' m')
        print('FPS: ', 1000000 / delta_time_learn)
        zero_completed_obs = np.zeros((n_cpu,) + env.observation_space.shape)
        zero_completed_obs[0, :] = test_env.reset()
        state = None
        reward_sum = 0
        all_net_worth_np = np.array([])
        time_test = time.time()
        for i in range(5000):
            action, _states = model.predict(zero_completed_obs, state=state)
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

        average_net_worth = int(np.sum([all_net_worth_np]) / len(all_net_worth_np))
        delta_time_test = time.time() - time_test
        print('Test  end', ' epoch:', n_epoch, ' time to work:', int((delta_time_test) / 60), ' m')
        print('FPS: ', 5000 / delta_time_test)
        print("Last done net worth in test: ", last_done_net_worth)
        print('Average net_worth in test:', average_net_worth)
        summary_writer.close()
        model_path = path.join('model', platform.system(), f'model_epoch_{n_epoch}.pkl')
        model.save(model_path)
        time.sleep(10)
        print('Model saved to :', model_path)

    time.sleep(10)
