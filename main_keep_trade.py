import csv

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
from io import StringIO
from sqlalchemy import create_engine
from KeepTradingEnv import KeepTradingEnv


database_name = 'keep_data'
table_name ='keep_rl_trade'
engine = create_engine('postgresql://postgres:postgres@localhost:5432/'+database_name)



def get_keep_data():
    conn = psycopg2.connect(database="keep_data", user="postgres", password="postgres", host="localhost", port="5432")
    market_data = psql.read_sql("Select * from keep_info", conn)
    conn.close()
    return market_data


df = get_keep_data()
df = df.sort_values('index')
df_save = df.copy()
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

    def psql_insert_copy(table, conn, keys, data_iter):
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            s_buf = StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)

            columns = ', '.join('"{}"'.format(k) for k in keys)
            if table.schema:
                table_name = '{}.{}'.format(table.schema, table.name)
            else:
                table_name = table.name

            sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
                table_name, columns)
            cur.copy_expert(sql=sql, file=s_buf)


    study = optuna.create_study(study_name='cartpol_optuna', storage='sqlite:///params.db', load_if_exists=True)
    model_params = get_model_params()
    print(model_params)

    n_cpu = 4

    env = SubprocVecEnv([make_env(i) for i in range(n_cpu)])  # задаем колл-во процессоров
    test_env = DummyVecEnv([make_envTest(i) for i in range(1)])

    # model = PPO2(MlpLnLstmPolicy, env, nminibatches=1, verbose=1, n_steps=49, tensorboard_log="./tensorboard_keep/",
    #              **model_params)
    model = PPO2.load("model/Windows/model_epoch_6.pkl", nminibatches=n_cpu, env=env, verbose=1, n_steps=49, tensorboard_log="./tensorboard_keep_trade/")
    model.is_tb_set = True

    for n_epoch in range(0, 1):
        summary_writer = tf.compat.v1.summary.FileWriter("./tensorboard_keep/" + "Keep_trade_test_" + str(n_epoch+1))
        print('\x1b[6;30;42m' + '***************************  Calculate epoch:', n_epoch,
              '***************************' + '\x1b[0m')

        #save_s = model.learn(total_timesteps=20000, tb_log_name='Keep_learn')

        zero_completed_obs = np.zeros((n_cpu,) + env.observation_space.shape)
        zero_completed_obs[0, :] = test_env.reset()
        state = None
        reward_sum = 0
        all_net_worth_np = np.array([])
        time_test = time.time()
        save = pd.DataFrame(columns=['time', 'action','reward','keep_price','net_worth','btc_price','eth_price'])

        for i in range(200):
            action, states = model.predict(zero_completed_obs, state=state)
            obs, reward, done, info = test_env.step(action)
            zero_completed_obs[0, :] = obs

            keep_price = df_save['close_keep'][i+1]
            net_worth_log = info[0]['net_worth']
            action_log = info[0]['action']
            reward_log = reward[0]

            print(f"{i} {df_save['index'][i+1]} acttion:{action_log} reward:{reward_log} keep_price:{keep_price} net_worth:{net_worth_log} ")
            if action_log == 0: acton_t = 'PASS'
            if action_log == 1: acton_t = 'BUY'
            if action_log == 2: acton_t = 'SELL'
            buf = pd.DataFrame([{'time': df_save['index'][i+1],
                                 'action': acton_t,
                                 'reward': reward[0],
                                 'keep_price': df_save['close_keep'][i+1],
                                 'net_worth': info[0]['net_worth'],
                                 'btc_price': df_save['close_btc'][i+1],
                                 'eth_price': df_save['close_eth'][i+1]}])
            save = save.append(buf)
            reward_sum += reward[0]
            if done[0]:
                save.set_index(save['time'], inplace=True)
                del save['time']
                save.to_sql(table_name, engine, if_exists='replace', method=psql_insert_copy)
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
                break


    time.sleep(10)
