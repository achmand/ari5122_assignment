# importing dependencies 
import os 
import pandas as pd

# does not take care of race conditions...
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def df2csv(path, filename, function, *args):
    create_dir(path)
    path_file = path + filename + ".csv"

    # file does not exist so save df to csv
    if(os.path.isfile(path_file) == False):
        df = function(*args)
        df.to_csv(path_file, sep = ",")
    else:
        print("File already found in path specified: {0}.".format(path_file))

def csv2df(path, filename, index_col= None):
    return pd.read_csv(path + filename + ".csv", index_col=index_col)