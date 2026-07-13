from importlib.resources import files

import numpy as np
import pandas as pd


def load_data(filename: str) -> pd.DataFrame:
    filepath = files("bellatrex.datasets") / filename
    return pd.read_csv(filepath)


# Functions to load specific datasets
# Separation between X and y is done 'manually', that is: case by case
def load_binary_data(return_X_y=False):
    if return_X_y is False:
        return load_data("binary_tutorial.csv")
    else:
        df = load_data("binary_tutorial.csv")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y


def load_regression_data(return_X_y=False):
    if return_X_y is False:
        return load_data("regression_tutorial.csv")
    else:
        df = load_data("regression_tutorial.csv")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y


def load_survival_data(return_X_y=False):
    if return_X_y is False:
        return load_data("survival_tutorial.csv")
    else:
        df = load_data("survival_tutorial.csv")
        X = df.iloc[:, :-2]
        y = df.iloc[:, -2:]

        dtypes_map = {str(y.columns[0]): np.bool_, str(y.columns[1]): np.float32}
        y = y.to_records(index=False, column_dtypes=dtypes_map)

        return X, y


def load_mlc_data(return_X_y=False):
    if return_X_y is False:
        return load_data("multi-label_tutorial.csv")
    else:
        df = load_data("multi-label_tutorial.csv")
        columns_out = [col for col in df.columns if "tag" in col]
        columns_in = [col for col in df.columns if "tag" not in col]

        return df.loc[:, columns_in], df.loc[:, columns_out]


def load_mtr_data(return_X_y=False):
    if return_X_y is False:
        return load_data("multi-target_tutorial.csv")
    else:
        df = load_data("multi-target_tutorial.csv")
        columns_out = [col for col in df.columns if "target" in col]
        columns_in = [col for col in df.columns if "target" not in col]

        return df.loc[:, columns_in], df.loc[:, columns_out]
