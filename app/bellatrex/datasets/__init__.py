import pkg_resources
import pandas as pd
import numpy as np

def load_data(filename):
    #returns a file system path for a resource (like a data file), within a package.
    filepath = pkg_resources.resource_filename('bellatrex.datasets', filename)
    return pd.read_csv(filepath)

# Functions to load specific datasets
# Spearation between X and y is done 'manually', that is: case by case 
def load_binary_data(return_X_y=False):
    if return_X_y is False:
        return load_data('bin_tutorial.csv')
    else:
        df = load_data('bin_tutorial.csv')
        X= df.iloc[:,:-1]
        y = df.iloc[:,-1]
        return X, y

def load_regression_data(return_X_y=False):
    if return_X_y is False:
        return load_data('regress_tutorial.csv')
    else:
        df = load_data('regress_tutorial.csv')
        X= df.iloc[:,:-1]
        y = df.iloc[:,-1]
        return X, y



def load_survival_data(return_X_y=False):
    if return_X_y is False:
        return load_data('surv_tutorial.csv')
    else:
        df = load_data('surv_tutorial.csv')
        X = df.iloc[:,:-2]
        y = df.iloc[:,-2:]

        dtypes_map = {
            str(y.columns[0]): np.bool_,
            str(y.columns[1]): np.float32
            }
        y = y.to_records(index=False, column_dtypes=dtypes_map) 

        return X, y

def load_mlc_data(return_X_y=False):
    if return_X_y is False:
        return load_data('mlc_tutorial.csv')
    else:
        df = load_data('mlc_tutorial.csv')
        columns_out = [col for col in df.columns if 'tag' in col]
        columns_in = [col for col in df.columns if 'tag' not in col]

        return df.loc[:,columns_in], df.loc[:,columns_out]


def load_mtr_data(return_X_y=False):
    if return_X_y is False:
        return load_data('mtr_tutorial.csv')
    else:
        df = load_data('mtr_tutorial.csv')
        columns_out = [col for col in df.columns if 'target' in col]
        columns_in = [col for col in df.columns if 'target' not in col]

        return df.loc[:,columns_in], df.loc[:,columns_out]