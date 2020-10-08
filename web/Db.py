import psycopg2
import pandas as pd


class Db:
    def __init__(self):
        self.dbname = 'keep_data'
        self.user = 'postgres'
        self.password = 'postgres'
        self.host = 'localhost'

    def get_market_data(self):
        conn = psycopg2.connect(dbname=self.dbname, user=self.user,
                                password=self.password, host=self.host)
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM keep_info")
        records = cur.fetchall()
        conn.close()
        col_names = []
        for elt in cur.description:
            col_names.append(elt[0])

        df = pd.DataFrame(records, columns=col_names)

        return df

    def get_predict(self):
        conn = psycopg2.connect(dbname=self.dbname, user=self.user,
                                password=self.password, host=self.host)
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM keep_predict")
        records = cur.fetchall()
        conn.close()
        col_names = []
        for elt in cur.description:
            col_names.append(elt[0])

        df = pd.DataFrame(records, columns=col_names)

        return df

    def get_trades(self):
        conn = psycopg2.connect(dbname=self.dbname, user=self.user,
                                password=self.password, host=self.host)
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM keep_rl_trade")
        records = cur.fetchall()
        conn.close()
        col_names = []
        for elt in cur.description:
            col_names.append(elt[0])

        df = pd.DataFrame(records, columns=col_names)

        return df
