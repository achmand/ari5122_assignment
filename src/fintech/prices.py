###### importing dependencies ############################################################################
import numpy as np 
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr

###### prices functions ##################################################################################
yf.pdr_override() # yahoo fix 

def download_prices(ticker, start_date, end_date, interval="1d"):
    """ Dowloads historical prices for the specified ticked from yahoo finance. 

        Args:
            ticker (str): The asset symbol to download prices for. 
            start_date (str): The starting date to fetch the historical prices from ('%Y-%m-%d'). 
            end_date (str): The end date to fetch the historical prices to ('%Y-%m-%d').
            interval (str): By default set to 1d (daily) interval. Other options 1wk (weekly) or 1mo (monthly).

        Returns: 
            pandas dataframe: A dataframe containing the historical prices for the specified ticker. 
    """
    return pdr.get_data_yahoo(ticker, start = start_date, end = end_date, interval=interval)

def log_diff(dataframe, column, shift=1):
    """ Calculates log difference for a specified column in a dataframe.

        Args:
            dataframe (pandas df): The dataframe used to compute the log difference for.
            column (str): The column used to calculate the log difference for.
            shift (int): The number of shifted values to calculate the difference with. 

        Returns:
            numpy array: A numpy array with the log differences between the column and shifted column.
    """
    return np.log(dataframe[column] / dataframe[column].shift(shift))

##########################################################################################################
