"""Train a forest on a bundled dataset and explain it; mirrors tutorial.ipynb.

Run ``python tutorial.py --headless`` for a non-interactive run.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Disable all GUI windows.")
    args = parser.parse_args()
    headless = args.headless or os.environ.get("BELLATREX_TUTORIAL_HEADLESS") == "1"
    if headless:
        os.environ["MPLBACKEND"] = "Agg"

    import bellatrex
    import matplotlib

    if headless:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt
    from bellatrex import BellatrexExplain, predict_helper
    from bellatrex import datasets as bellatrex_datasets
    from bellatrex.utilities import get_auto_setup
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sksurv.ensemble import RandomSurvivalForest

    print("Bellatrex version:", bellatrex.__version__)
    plot_gui = not headless

    # Uncomment the dataset matching the task you want to explore.
    X, y = bellatrex_datasets.load_binary_data(return_X_y=True)
    # X, y = bellatrex_datasets.load_regression_data(return_X_y=True)
    # X, y = bellatrex_datasets.load_survival_data(return_X_y=True)
    # X, y = bellatrex_datasets.load_mlc_data(return_X_y=True)
    # X, y = bellatrex_datasets.load_mtr_data(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)

    # --- Step 1: Train a Random Forest --------------------------------------
    setup = get_auto_setup(y).lower()
    if setup == "survival":
        clf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, random_state=0)
    elif setup in ["binary", "multi-label"]:
        clf = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=0)
    else:
        clf = RandomForestRegressor(n_estimators=100, min_samples_split=5, random_state=0)
    clf.fit(X_train, y_train)
    print(f"Trained {type(clf).__name__} for {setup}.")

    # --- Step 2: Fit Bellatrex and explain predictions -----------------------
    # fit() prepares the explainer without retraining the forest we just fitted.
    Btrex_fitted = BellatrexExplain(
        clf,
        set_up="auto",
        p_grid={"n_clusters": [1, 2, 3], "n_dims": [2, None]},
        verbose=1,
    ).fit(X_train, y_train)

    # Background predictions are used by the rule-level visualization.
    y_train_pred = predict_helper(clf, X_train)
    sample_index = 0
    print(f"\n--- Explaining sample i={sample_index} ---")
    tuned_method = Btrex_fitted.explain(X_test, sample_index)

    # Plot 1: cluster overview. Interactive runs open a NiceGUI explorer.
    fig_overview, _ = tuned_method.plot_overview(plot_gui=plot_gui, show=plot_gui)
    plt.close(fig_overview)

    # Plot 2: rule-level detail, supported for single-output tasks.
    if setup in ["binary", "regression", "survival"]:
        fig_visuals, _ = tuned_method.plot_visuals(
            plot_max_depth=5,
            preds_distr=y_train_pred,
            conf_level=0.9,
            tot_digits=4,
            show=False,
        )
        if not headless:
            plt.show(block=True)
        plt.close(fig_visuals)

    # Save the text explanation and print it to the console.
    tuned_method.create_rules_txt()
    tuned_method.print_rules_txt()


if __name__ == "__main__":
    main()
