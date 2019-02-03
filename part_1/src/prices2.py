import os
import sys 
import numpy as np
import pandas as pd
import utilities as util
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr

def download_prices_csv(tickers, start_date, end_date, path):
    """Downloads price data from yahoo finance and saves as csv.

    Args:
        tickers (list of tuple): List of tuples (instrument_name, ticker_symbol) which will be downloaded.
        start_date (str): The start date to query for the prices data.
        end_date (str): The end date to query for the prices data.
        path (str): The path where to save the csv files. 

    """
    yf.pdr_override() 
    for i in range(len(tickers)):
        tmp_prices = pdr.get_data_yahoo(tickers[i][1], start = start_date, end = end_date)
        tmp_prices.to_csv(path + tickers[i][0] + ".csv", sep=",")

def csv_df_dictionary(path):
    df_dictionary = {}
    directory = os.listdir(path)
    for file in directory:
        tmp_key = os.path.splitext(file)[0]
        df_dictionary[tmp_key] = pd.read_csv(path + file)     
    return df_dictionary

def df_log_diff(dataframes, columns, column_headers):
    dfs_log_dif = util.df_to_dfs(dataframes) 
    for i in range(len(dataframes)):
        dfs_log_dif[i][column_headers] = np.log(dataframes[i][columns] / dataframes[i][columns].shift(1))
    return dfs_log_dif