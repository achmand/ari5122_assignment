# importing dependencies 
import numpy as np 
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr

yf.pdr_override() 

def download_prices(ticker, start_date, end_date):
    return pdr.get_data_yahoo(ticker, start = start_date, end = end_date)

def log_diff(dataframe, column, shift = 1):
    return np.log(dataframe[column] / dataframe[column].shift(shift))

