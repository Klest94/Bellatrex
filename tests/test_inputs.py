import pytest
from bellatrex.bellatrex_explain import BellatrexExplain
from bellatrex.datasets import (
    load_binary_data,
    load_mlc_data,
    load_mtr_data,
    load_regression_data,
    load_survival_data,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest

DATA_LOADERS = {
    "binary": load_binary_data,
    "regression": load_regression_data,
    "survival": load_survival_data,
    "multi-label": load_mlc_data,
    "multi-target": load_mtr_data,
}

MAX_TEST_SAMPLES = 2

# --- Edge Case Test: Single Data Point ---
# def test_single_data_point():
#     for loader in DATA_LOADERS:
#         X, y = loader(return_X_y=True)
#         X_single, y_single = X[:1], y[:1]  # Single data point
#         SETUP = get_auto_setup(y_single)

#         if SETUP.lower() in "survival":
#             clf = RandomSurvivalForest(n_estimators=10, min_samples_split=2, n_jobs=-1, random_state=0)
#         elif SETUP.lower() in ["binary", "multi-label"]:
#             clf = RandomForestClassifier(n_estimators=10, min_samples_split=2, n_jobs=-1, random_state=0)
#         elif SETUP.lower() in ["regression", "multi-target"]:
#             clf = RandomForestRegressor(n_estimators=10, min_samples_split=2, n_jobs=-1, random_state=0)
#         else:
#             raise ValueError(f"Detection task {SETUP} not compatible with Bellatrex (yet)")

#         clf.fit(X_single, y_single)
#         btrex_fitted = BellatrexExplain(clf, set_up="auto", verbose=3).fit(X_single, y_single)
#         explanation = btrex_fitted.explain(X_single, 0)
#         assert explanation is not None, "Explanation failed for single data point"


# --- Error Handling Tests ---
def test_invalid_hyperparameters():
    for setup, loader in DATA_LOADERS.items():
        X, y = loader(return_X_y=True)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)

        if setup.lower() in "survival":
            clf = RandomSurvivalForest(
                n_estimators=10, min_samples_split=5, n_jobs=-1, random_state=0
            )
        elif setup.lower() in ["binary", "multi-label"]:
            clf = RandomForestClassifier(
                n_estimators=10, min_samples_split=5, n_jobs=-1, random_state=0
            )
        elif setup.lower() in ["regression", "multi-target"]:
            clf = RandomForestRegressor(
                n_estimators=10, min_samples_split=5, n_jobs=-1, random_state=0
            )
        else:
            raise ValueError(f"Detection task {setup} not compatible with Bellatrex (yet)")

        clf.fit(X_train, y_train)

        invalid_p_grid = {"n_trees": [-1, 0], "n_dims": [0, 1], "n_clusters": [0, "invalid"]}
        with pytest.raises(ValueError):
            BellatrexExplain(clf, set_up="auto", p_grid=invalid_p_grid).fit(X_train, y_train)


def test_unfitted_model():
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    btrex = BellatrexExplain(clf, set_up="auto")
    with pytest.raises(Exception):
        btrex.explain(None, 0)


def test_unsupported_model():
    class UnsupportedModel:
        """Just a dummy class to simulate an unsupported model type."""

        pass

    clf = UnsupportedModel()
    try:
        BellatrexExplain(clf, set_up="auto")
    except ValueError as e:
        assert "Unsupported model type" in str(e), "Expected ValueError for unsupported model"
