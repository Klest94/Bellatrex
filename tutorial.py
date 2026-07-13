"""
This script demonstrates how to use Bellatrex.
It mirrors the Jupyter notebook, but is easier to run locally or in automated checks.
"""

import os

import bellatrex
import matplotlib.pyplot as plt
import joblib

from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from bellatrex import BellatrexExplain, pack_trained_ensemble, predict_helper
from bellatrex import datasets as bellatrex_datasets
from bellatrex.utilities import get_auto_setup

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


# --- Step 2: Pack or load the trained model (optional) ----------------------

# The pre-trained model may be stored under app/bellatrex/datasets/model_example.pkl
# To save your own model, uncomment and adjust the line below:
# joblib.dump(clf, os.path.join('app', 'bellatrex', 'datasets', 'model_example.pkl'))

model_path = os.path.join("app", "bellatrex", "datasets", f"{SETUP}_pretrained.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No pre-trained model found at {model_path}")

clf = joblib.load(model_path)
print(f"Loaded pre-trained model from {model_path}")

# pack_trained_ensemble converts the fitted forest into a memory-efficient dictionary.
# Pass clf_packed (or the original clf) to BellatrexExplain – both are supported.
clf_packed = pack_trained_ensemble(clf)
print(f"Packed {clf_packed['ensemble_class']} with {len(clf_packed['trees'])} trees.")


# --- Step 3: Fit Bellatrex and explain predictions --------------------------

Btrex_fitted = BellatrexExplain(
    clf,
    set_up="auto",
    p_grid={"n_clusters": [1, 2, 3], "n_dims": [2, None]},
    verbose=1,
).fit(X_train, y_train)

# Pre-compute training predictions once, used as background distribution in plot_visuals
y_train_pred = predict_helper(clf, X_train)

SAMPLE_INDEX = 0
print(f"\n--- Explaining sample i={SAMPLE_INDEX} ---")

tuned_method = Btrex_fitted.explain(X_test, SAMPLE_INDEX)

# Plot 1: cluster overview (shows pre-selected trees and selected rules).
# In GUI mode this opens one NiceGUI explorer for SAMPLE_INDEX.
fig_overview, _ = tuned_method.plot_overview(plot_gui=PLOT_GUI, show=PLOT_GUI)
if not PLOT_GUI:
    plt.show(block=True)
plt.close(fig_overview)

# Plot 2: rule-level detail (single-output tasks only).
# plot_visuals is always a plain matplotlib figure; show it regardless of GUI mode.
if SETUP.lower() in ["binary", "regression", "survival"]:
    fig_visuals, _ = tuned_method.plot_visuals(
        plot_max_depth=5,
        preds_distr=y_train_pred,
        conf_level=0.9,
        tot_digits=4,
        show=False,
    )
    plt.show(block=True)
    plt.close(fig_visuals)

# Save the text explanation and print it to the console.
tuned_method.create_rules_txt()
tuned_method.print_rules_txt()
