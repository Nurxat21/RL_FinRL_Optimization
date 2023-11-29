import numpy as np
import pandas as pd
import yfinance as yf
from Train_ import *
from Env_ import PortfolioOptimizationEnv
tic_list = ["AAPL", "T", "CAT", "BA"]
start_date = '2011-01-01'
end_date = '2023-01-01'
time_interval = '1d'
dataframe = pd.DataFrame()
for tic in tic_list:
        temp_df = yf.download(tic, start=start_date, end=end_date, interval=time_interval)
        temp_df["Tic"] = tic
        dataframe = pd.concat([dataframe, temp_df], axis=0, join="outer")
dataframe.reset_index(inplace=True)
dataframe["Day"] = dataframe["Date"].dt.dayofweek
df_portfolio = dataframe[["Date", "Tic", "Close", "High", "Low"]]

environment = PortfolioOptimizationEnv(
        df_portfolio,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=50,
        features=["Close", "High", "Low"]
    )

algo = PG(environment, lr=0.0001)

algo.train(episodes=250)