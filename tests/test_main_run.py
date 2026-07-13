"""
Author: Klest Dedja
Here we manually test most of the the pipeline, from data loading to model explanation building.
We cover most of the combinations of tasks, models, and settings.
"""

import matplotlib
import pytest

matplotlib.use("Agg")  # Must be before importing pyplot

import os
from importlib import import_module

import matplotlib.pyplot as plt  # Safe after backend is set
from bellatrex import BellatrexExplain

# from bellatrex.wrapper_class import pack_trained_ensemble
from bellatrex.datasets import (
    load_binary_data,
    load_mlc_data,
    load_mtr_data,
    load_regression_data,
    load_survival_data,
)
from bellatrex.utilities import get_auto_setup
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest

IS_CI = os.environ.get("CI") == "true"

MAX_TEST_SAMPLES = 2

DATA_LOADERS = {
    "binary": load_binary_data,
    "regression": load_regression_data,
    "survival": load_survival_data,
    "multi-label": load_mlc_data,
    "multi-target": load_mtr_data,
}


# --- Main setup logic shared by the other tests ---
def prepare_fitted_bellatrex(setup, loader):
    X, y = loader(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    assert setup == get_auto_setup(
        y
    ), f"Automatic task detection failed: found {get_auto_setup(y)} instead of {setup}"

    if setup.lower() == "survival":
        clf = RandomSurvivalForest(
            n_estimators=100, min_samples_split=10, n_jobs=-2, random_state=0
        )
    elif setup.lower() in ["binary", "multi-label"]:
        clf = RandomForestClassifier(
            n_estimators=100, min_samples_split=5, n_jobs=-2, random_state=0
        )
    elif setup.lower() in ["regression", "multi-target"]:
        clf = RandomForestRegressor(
            n_estimators=100, min_samples_split=5, n_jobs=-2, random_state=0
        )
    else:
        raise ValueError(f"Detection task {setup} not compatible with Bellatrex (yet)")

    clf.fit(X_train, y_train)
    test_grid = {"n_trees": [0.6, 1.0], "n_dims": [2, None], "n_clusters": [1, 2, 3]}
    btrex_fitted = BellatrexExplain(clf, set_up="auto", p_grid=test_grid, verbose=3).fit(
        X_train, y_train
    )

    return btrex_fitted, X_test


# --- Core (non-GUI) test ---
def test_core_workflow():

    for setup, loader in DATA_LOADERS.items():  # Iterate over loading functions
        btrex_fitted, X_test = prepare_fitted_bellatrex(setup, loader)
        for i in range(MAX_TEST_SAMPLES):
            tuned_method = btrex_fitted.explain(X_test, i)
            tuned_method.plot_overview(show=not IS_CI, plot_gui=False)


# --- Rules and file handling test ---
def test_create_rules_txt(tmp_path):
    for setup, loader in DATA_LOADERS.items():
        X, y = loader(return_X_y=True)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)

        if setup.lower() == "survival":
            clf = RandomSurvivalForest(
                n_estimators=100, min_samples_split=5, n_jobs=-1, random_state=0
            )
        elif setup.lower() in ["binary", "multi-label"]:
            clf = RandomForestClassifier(
                n_estimators=100, min_samples_split=5, n_jobs=-1, random_state=0
            )
        elif setup.lower() in ["regression", "multi-target"]:
            clf = RandomForestRegressor(
                n_estimators=100, min_samples_split=5, n_jobs=-1, random_state=0
            )
        else:
            raise ValueError(f"Detection task {setup} not compatible with Bellatrex (yet)")

        clf.fit(X_train, y_train)
        btrex_fitted = BellatrexExplain(clf, set_up="auto", verbose=1).fit(X_train, y_train)

        for i in range(MAX_TEST_SAMPLES):
            btrex_fitted.explain(X_test, i)
            out_file = "test_rules.txt"
            explanation_out_dir_path = tmp_path
            out_file, out_file_extra = btrex_fitted.create_rules_txt(
                out_dir=str(explanation_out_dir_path), out_file=out_file
            )
            assert os.path.exists(out_file), "Rules file was not created"

            with open(out_file, "r", encoding="utf8") as f:
                content = f.read()
                assert len(content) > 0, "Rules file is empty."

            enough_trees = clf.n_estimators * min(btrex_fitted.p_grid["n_trees"]) > max(
                btrex_fitted.p_grid["n_clusters"]
            )
            os.remove(out_file)
            # If there are more trees than clusters, check for extra rules file and remove it
            if enough_trees:

                assert os.path.exists(out_file_extra), "Rules file was not created"
                with open(out_file_extra, "r", encoding="utf8") as f:
                    content = f.read()
                    assert len(content) > 0, "Rules file is empty."

                os.remove(out_file_extra)


# --- GUI test with plot_gui=True ---
# @pytest.mark.gui
# def test_gui_workflow_old():
#     if IS_CI:
#         matplotlib.use("Agg")  # Non-blocking backend when running in CI

#     for _, loader in DATA_LOADERS.items():  # Iterate over loading functions
#         btrex_fitted, X_test = prepare_fitted_bellatrex(loader)

#         for i in range(MAX_TEST_SAMPLES):
#             tuned_method = btrex_fitted.explain(X_test, i)
#             fig, obj = tuned_method.plot_overview(show=not IS_CI, plot_gui=True)
#             if not IS_CI: # if show=True
#                 assert fig is not None
#                 plt.close(fig)
#             else:
#                 assert obj is not None  # NiceGUI plot objects


@pytest.mark.gui
def test_gui_workflow(monkeypatch):

    pytest.importorskip("nicegui", reason="Install Bellatrex[gui] to run GUI tests")
    nicegui_plots_code = import_module("bellatrex.nicegui_plots_code")
    launch_calls = []

    def fake_launch_nicegui_window(plots, temp_files_dir, blocking=None, render_context=None):
        launch_calls.append(
            {
                "plots": plots,
                "temp_files_dir": temp_files_dir,
                "blocking": blocking,
                "render_context": render_context,
            }
        )

    monkeypatch.setattr(
        nicegui_plots_code,
        "launch_nicegui_window",
        fake_launch_nicegui_window,
    )
    pytest.importorskip("nicegui", reason="Install Bellatrex[gui] to run GUI tests")
    pytest.importorskip("plotly", reason="Install Bellatrex[gui] to run GUI tests")

    for setup, loader in DATA_LOADERS.items():
        btrex_fitted, X_test = prepare_fitted_bellatrex(setup, loader)

        for i in range(MAX_TEST_SAMPLES):
            tuned_method = btrex_fitted.explain(X_test, i)

            # Exercise the show=True GUI path without opening real OS windows in tests.
            fig, obj = tuned_method.plot_overview(show=True, plot_gui=True)
            assert obj is not None
            assert launch_calls[-1]["plots"] is obj
            assert launch_calls[-1]["render_context"]["clf"] is tuned_method.tuned_method.clf
            plt.close(fig)

    assert len(launch_calls) == len(DATA_LOADERS) * MAX_TEST_SAMPLES
