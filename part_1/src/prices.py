from pandas_datareader import data as pdr 
import fix_yahoo_finance as yf 
import pandas as pd

def download_prices_csv(tickers, start_date, end_date, path):
    yf.pdr_override()
    for i in range(len(tickers)):
        tmp_prices = pdr.get_data_yahoo(tickers[i], start = start_date, end = end_date)
        tmp_prices.to_csv(path + tickers[i] + ".csv", sep=",")