"""
This is a Python script with the tutorial, to demonstrate how to use Bellatrex.
Less user friendly compare to the notebook, but easier to run tests.
For a more user-friendly tutorial, please check the Jupyter notebook with the same name.
"""

import os

import bellatrex
import matplotlib.pyplot as plt
from bellatrex import BellatrexExplain, pack_trained_ensemble, predict_helper
from bellatrex import datasets as bellatrex_datasets
from bellatrex.utilities import get_auto_setup
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest

print("Bellatrex version:", bellatrex.__version__)
print("Working directory:", os.getcwd())

PLOT_GUI = True

# Uncomment the dataset that matches the prediction task you want to explore:
X, y = bellatrex_datasets.load_binary_data(return_X_y=True)  # binary classification
# X, y = bellatrex_datasets.load_regression_data(return_X_y=True)  # regression
# X, y = bellatrex_datasets.load_survival_data(return_X_y=True)    # survival analysis
# X, y = bellatrex_datasets.load_mlc_data(return_X_y=True)         # multi-label classification
# X, y = bellatrex_datasets.load_mtr_data(return_X_y=True)         # multi-target regression

X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)

# --- Step 1: Train a Random Forest ------------------------------------------

SETUP = get_auto_setup(y)
print("Detected prediction task 'SETUP':", SETUP)

if SETUP.lower() == "survival":
    clf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, n_jobs=-2, random_state=0)
elif SETUP.lower() in ["binary", "multi-label"]:
    clf = RandomForestClassifier(n_estimators=100, min_samples_split=5, n_jobs=-2, random_state=0)
elif SETUP.lower() in ["regression", "multi-target"]:
    clf = RandomForestRegressor(n_estimators=100, min_samples_split=5, n_jobs=-2, random_state=0)
else:
    raise ValueError(f"Unknown prediction task SETUP={SETUP}.")

clf.fit(X_train, y_train)
print("Model fitting complete.")


# --- Step 2: Pack the trained model (optional) -------------------------------

# pack_trained_ensemble converts the fitted forest into a portable dictionary.
# Pass clf_packed (or the original clf) to BellatrexExplain – both are supported.
clf_packed = pack_trained_ensemble(clf)
print(f"Packed {clf_packed['ensemble_class']} with {len(clf_packed['trees'])} trees.")


# --- Step 3: Fit Bellatrex and explain predictions --------------------------

Btrex_fitted = BellatrexExplain(
    clf, set_up="auto", p_grid={"n_clusters": [1, 2, 3]}, verbose=1
).fit(X_train, y_train)

# Pre-compute training predictions once, used as background distribution in plot_visuals
y_train_pred = predict_helper(clf, X_train)

N_TEST_SAMPLES = 2
for i in range(N_TEST_SAMPLES):
    print(f"\n--- Explaining sample i={i} ---")

    tuned_method = Btrex_fitted.explain(X_test, i)

    # Plot 1: cluster overview (shows pre-selected trees and selected rules)
    tuned_method.plot_overview(plot_gui=PLOT_GUI, show=PLOT_GUI)
    # When plot_gui=True the NiceGUI window blocks until closed, then the loop continues.
    if PLOT_GUI:
        # Ensure no placeholder matplotlib figure leaks from overview when NiceGUI is used.
        plt.close("all")
    else:
        plt.show(block=True)

    # Plot 2: rule-level detail (single-output tasks only)
    if SETUP.lower() in ["binary", "survival", "regression"]:
        tuned_method.plot_visuals(
            plot_max_depth=5, preds_distr=y_train_pred, conf_level=0.9, tot_digits=4, show=False
        )
        # plot_visuals is always a plain matplotlib figure; show it regardless of GUI mode.
        plt.show(block=True)
        plt.close("all")
