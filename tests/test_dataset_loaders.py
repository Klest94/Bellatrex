import numpy as np
import pandas as pd
import pytest
from bellatrex.datasets import (
    load_binary_data,
    load_mlc_data,
    load_mtr_data,
    load_regression_data,
    load_survival_data,
)

# Parametrize the test cases: (load_function, expected_y_shape, return_xy)
dataset_loaders = [
    (load_binary_data, (748,), True),  # Binary: single column y
    (load_regression_data, (1503,), True),  # Regression: single column y
    (load_survival_data, (500,), True),  # Survival: structured array
    (load_mlc_data, (193, 7), True),  # Multi-label: 7 target columns
    (load_mtr_data, (1137, 3), True),  # Multi-target: 3 target columns
]


@pytest.mark.parametrize("loader_func, expected_y_shape, return_xy", dataset_loaders)
def test_dataset_loader_output(loader_func, expected_y_shape, return_xy):
    X, y = loader_func(return_X_y=return_xy)

    assert isinstance(X, pd.DataFrame)
    assert len(X) == expected_y_shape[0]
    if isinstance(y, np.ndarray) and y.dtype.names is not None:
        # survival: structured array
        assert y.shape == expected_y_shape
        assert set(y.dtype.names) == {"event", "time"} or set(y.dtype.names) == set(y.dtype.fields)
    elif isinstance(y, pd.DataFrame):
        assert y.shape == expected_y_shape
    else:
        # y is a Series or 1D array
        assert len(y) == expected_y_shape[0]


@pytest.mark.parametrize(
    "loader_func",
    [
        load_binary_data,
        load_regression_data,
        load_survival_data,
        load_mlc_data,
        load_mtr_data,
    ],
)
def test_dataset_loader_df(loader_func):
    df = loader_func(return_X_y=False)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
