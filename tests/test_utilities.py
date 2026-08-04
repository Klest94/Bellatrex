import numpy as np
from bellatrex.utilities import (
    concatenate_helper,
    frmt_pretty_print,
    get_auto_setup,
    string_to_pretty_digits,
    trail_pretty_digits,
)


def test_get_auto_setup():
    # Test binary classification
    y_test = np.array([0, 1, 0, 1])
    assert get_auto_setup(y_test) == "binary"

    # Test regression
    y_test = np.array([1.2, 3.4, 5.6])
    assert get_auto_setup(y_test) == "regression"

    # Test multi-label classification
    y_test = np.array([[0, 1], [1, 0], [0, 1]])
    assert get_auto_setup(y_test) == "multi-label"

    # Test multi-target regression
    y_test = np.array([[1.2, 3.4], [5.6, 7.8]])
    assert get_auto_setup(y_test) == "multi-target"


def test_concatenate_helper():
    y_pred = np.empty((0, 2))
    y_local_pred = np.array([[1, 2], [3, 4]])
    result = concatenate_helper(y_pred, y_local_pred)
    assert result.shape == (2, 2)
    assert np.array_equal(result, y_local_pred)


def test_trail_pretty_digits():
    assert trail_pretty_digits(0.00123, 4) == 4  # Updated expected output
    assert trail_pretty_digits(123.456, 4) == 1


def test_string_to_pretty_digits():
    assert string_to_pretty_digits("123.456", digits_single=4) == "123.4"
    assert string_to_pretty_digits("1.23e-4", digits_single=4) == "1.23e-4"


def test_frmt_pretty_print():
    assert frmt_pretty_print(np.array([0.123, 0.456]), digits_vect=3) == "0.123, 0.456"
    assert frmt_pretty_print(123.456, digits_single=4) == "123.5"
