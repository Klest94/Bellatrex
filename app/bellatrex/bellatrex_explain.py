import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.model_selection import ParameterGrid
from sklearn.utils.validation import check_is_fitted
from sksurv.ensemble import RandomSurvivalForest

from .tree_extraction import TreeExtraction
from .utilities import (
    _infer_set_up,
    frmt_pretty_print,
    normalize_set_up,
    plot_preselected_trees,
    predict_helper,
    rule_print_inline,
    rule_to_file,
)
from .visualization import plot_rules, read_rules
from .visualization_extra import _input_validation
from .wrapper_class import EnsembleWrapper


class BellatrexExplain:
    """
    Explain individual predictions of a Random Forest model using Bellatrex.

    Bellatrex pre-selects a subset of trees used to make a prediction, projects
    their rule representations into a low-dimensional space, clusters them, and
    returns one representative rule per cluster as the explanation.

    Parameters
    ----------
    clf : RandomForestClassifier, RandomForestRegressor, RandomSurvivalForest,
          EnsembleWrapper, or dict
        The (possibly pre-fitted) Random Forest model to explain.  A packed
        dictionary produced by ``pack_trained_ensemble`` is also accepted.
    set_up : str, default="auto"
        Prediction task.  ``"auto"`` infers the task from the fitted model.
        Explicit choices: ``"binary"``, ``"regression"``, ``"survival"``,
        ``"multi-label"``, ``"multi-target"``.
    force_refit : bool, default=False
        If ``True``, always re-fit ``clf`` even when it is already trained.
    verbose : int, default=0
        Verbosity level.  ``0`` = silent; ``1`` = summary; ``>=3`` = detailed.
    proj_method : str or None, default="PCA"
        Dimensionality-reduction method applied to rule vectors.  ``None``
        disables projection.
    dissim_method : str, default="rules"
        Dissimilarity metric used to compare tree rules.
    feature_represent : str, default="weighted"
        Strategy for building per-tree feature vectors.
    p_grid : dict or None, default=None
        Hyperparameter search grid.  Bellatrex selects the combination with
        the highest fidelity to the original model prediction.  Recognised
        keys and their defaults::

            {
                "n_trees":    [0.6, 0.8, 1.0],  # fraction or count of trees
                "n_dims":     [2, None],          # PCA output dimensions
                "n_clusters": [1, 2, 3],          # number of rules returned
            }

        Pass a subset of keys to override only those defaults.
    pre_select_trees : str, default="L2"
        Method used to pre-select trees before clustering.
    fidelity_measure : str, default="L2"
        Metric used to score candidate hyperparameter combinations.
    n_jobs : int, default=1
        Number of parallel jobs for the hyperparameter search.  Values > 1
        use thread-based parallelism (experimental).
    ys_oracle : array-like or None, default=None
        Optional ground-truth labels for the test set, used for oracle-aware
        fidelity scoring.  Rarely needed in practice.
    """

    FONT_SIZE = 14
    MAX_FEATURE_PRINT = 10

    _DEFAULT_P_GRID = {
        "n_trees": [0.6, 0.8, 1.0],
        "n_dims": [2, None],
        "n_clusters": [1, 2, 3],
    }

    def __init__(
        self,
        clf,
        set_up="auto",
        force_refit=False,
        verbose=0,
        proj_method="PCA",
        dissim_method="rules",
        feature_represent="weighted",
        p_grid=None,
        pre_select_trees="L2",
        fidelity_measure="L2",
        n_jobs=1,
        ys_oracle=None,
    ):
        self.clf = clf
        self.set_up = set_up if set_up == "auto" else normalize_set_up(set_up)
        self.force_refit = force_refit
        self.proj_method = proj_method
        self.dissim_method = dissim_method
        self.feature_represent = feature_represent
        self.p_grid = p_grid if p_grid is not None else {}
        self.pre_select_trees = pre_select_trees
        self.fidelity_measure = fidelity_measure
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.ys_oracle = ys_oracle

        if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer")

        # Initialised by explain(); guards plot/txt methods against being called early.
        self.sample = None
        self.tuned_method = None
        self.sample_index = None
        self.surrogate_pred_str = None

    def _validate_p_grid(self):
        """
        This method validates and sets the parameters for the hyperparameter grid.
        It checks if the provided keys in the p_grid dictionary are valid,
        and sets default values if any of them are missing.
        It also checks if the values provided for n_trees are valid and raises
        errors if necessary. Finally, it converts n_trees to the number
        of trees used by the underlying ensemble model in case the values
        for n_trees are given as proportions.

        Raises:
            ValueError: If n_trees is less than or equal to 0, or if the list of n_trees contains both
                        proportions and integers, or if any n_trees value is greater than n_estimators.
            Warning: If the hyperparameter list contains unexpected keys other
            than the default set, this function reverts to using default values.
        """

        if not isinstance(self.p_grid, dict):
            raise ValueError(
                "p_grid parameter is expected to be a dictionary. Found {type(self.p_grid)} instead."
            )

        default_keys = ["n_trees", "n_dims", "n_clusters"]

        unexpected_keys = [key for key in self.p_grid.keys() if key not in default_keys]
        if unexpected_keys:
            warnings.warn(
                f"The hyperparameter list contains unexpected keys: {unexpected_keys}. Ignoring them."
            )

        # A partial grid overrides only the named defaults. Copy the lists so
        # callers and instances cannot mutate the class-level defaults.
        merged_grid = {key: list(values) for key, values in self._DEFAULT_P_GRID.items()}
        merged_grid.update(
            {key: value for key, value in self.p_grid.items() if key in default_keys}
        )
        self.p_grid = merged_grid

        if "n_trees" not in self.p_grid.keys():
            self.n_trees = [0.6, 0.8, 1.0]  # set to default if not existing
        else:
            self.n_trees = self.p_grid["n_trees"]  # CAN BE A LIST

        if "n_dims" not in self.p_grid.keys() or self.p_grid["n_dims"] is None:
            self.n_dims = [None]  # set to default if not existing
        else:
            self.n_dims = self.p_grid["n_dims"]  # CAN BE A LIST
            # treat 'all' as None (compatible with sklearn's PCA)
            self.n_dims = [None if x == "all" else x for x in self.n_dims]

        if "n_clusters" not in self.p_grid.keys():
            self.n_clusters = [1, 2, 3]  # set to default if not existing
        else:
            self.n_clusters = self.p_grid["n_clusters"]  # CAN BE A LIST

        if min(self.n_trees) <= 0:
            raise ValueError("n_trees must be all > 0")

        if min(self.n_trees) < 1.0 and max(self.n_trees) > 1.0:
            raise ValueError(
                "The list of n_trees must either indicate a proportion"
                " of trees in the (0,1] interval, or indicate the number"
                " of tree learners."
            )

        # Check that the n_trees provided by the user does not exceed the number of total trees
        # in the R(S)F. This works for both a fitted sklearn model and a dictionary

        if max(self.n_trees) > self.clf.n_estimators:
            raise ValueError("'n_trees' hyperparater value cannot be greater than n_estimators")

        # if proportion of n_trees is given instead, check correctness and transform to integer values:
        if np.array(
            [isinstance(i, float) for i in self.n_trees]
        ).all():  # if all elements are floats
            if (
                max(self.n_trees) <= 1.0 and min(self.n_trees) > 0
            ):  # all elements are in the (0, 1] interval:
                # round to closest integer
                self.n_trees = (
                    (np.array(self.n_trees) * self.clf.n_estimators + 0.5).astype(int).tolist()
                )  # all params are sorted as list, keep consistency

    def is_fitted(self):  # auxiliary function that returns boolean
        """
        This function determines whether the classifier (`self.clf`) has been fitted.
        It considers two scenarios:
        - If `self.clf` is a dictionary, it is assumed to represent a pre-trained model.
            In this case, the function wraps  the dictionary using the `EnsembleWrapper` class
            and returns `True`, indicating the model is fitted.
        - If `self.clf` is not a dictionary, the function performs a check using `check_is_fitted`
            from sklearn. This check is applicable to sklearn or sksurv models. If `check_is_fitted`
            does not raise an exception, the function returns `True`. If an exception is raised,
            the function returns `False`, indicating the model is not fitted.

        Returns:
            bool: `True` if the model is fitted, `False` otherwise.
        """

        if isinstance(self.clf, dict):  # case of simple, packed dictionary:
            self.clf = EnsembleWrapper(
                self.clf
            )  # EnsembleWrapper() ensures compatibily of (packed) dictionaries
            return True
        elif isinstance(
            self.clf, EnsembleWrapper
        ):  # case where compatibility is already taken care of:
            return True
        else:  # case where full sklearn/sksurv model is given. Check if it is fitted or not
            try:
                check_is_fitted(self.clf)  # only with sklearn models (but works with all of them)
                return True
            except NotFittedError:
                return False
        # Note that from sklearn 1.3. we can simply use return _is_fitted(self.clf) # returns boolean already

    def __repr__(self):
        clf_name = (
            self.clf.__class__.__name__
            if not isinstance(self.clf, dict)
            else "dict (packed ensemble)"
        )
        return (
            f"BellatrexExplain("
            f"clf={clf_name}, "
            f"set_up={self.set_up!r}, "
            f"proj_method={self.proj_method!r}, "
            f"n_jobs={self.n_jobs}, "
            f"verbose={self.verbose})"
        )

    def fit(self, X, y):
        """
        Fits the classifier to the data if not already fitted or if force refit is requested.
        It also checks the validity of the hyperparameters and sets up the prediction task
        based on the type of fitted classifier.

        Parameters:
        - X : array-like, shape (n_samples, n_features). Training dataset.
        - y : array-like, shape (n_samples,) or (n_samples, n_outputs). Target values.

        Raises:
        - ValueError: If an incompatible model type is provided or specific conditions required by
            the model setup are not met. For example, if a dictionary format is used when 'auto'
            set-up is selected, or if the classifier is not recognized/supported by the framework.

        Returns:
        - self : object
        Returns the instance itself.

        Notes:
        - If 'force_refit' is False and the model is already fitted, it skips the fitting process
            and proceeds to build an explanation.
        - If 'verbose' is 1 or higher, it will print the fitting status.
        - Automatically determines the prediction task ('set_up') based on classifier properties.
        """

        if self.force_refit is False and self.is_fitted():
            if self.verbose >= 1:
                print("Model is already fitted, building explanation.")
        else:
            if self.verbose >= 1:
                print("Fitting the model...", end="")
            if hasattr(self.clf, "n_jobs"):
                self.clf.n_jobs = self.n_jobs
            self.clf.fit(X, y)
            if self.verbose >= 1:
                print("fitting complete")

        # then check whether the input grid values are admissible
        self._validate_p_grid()

        if self.verbose >= 2:
            print(f"oracle_sample is: {self.ys_oracle}")

        if self.set_up == "auto":  # automatically determine scenario based on fitted classifier
            self.set_up = _infer_set_up(self.clf, y)
            if self.verbose > 0:
                print(f"Automatically setting prediction task to: {self.set_up}")

        return self

    def explain(self, X, idx):
        """
        Run Bellatrex for a single test sample and store the results.

        This method performs the full hyperparameter search (over ``p_grid``),
        selects the best tree subset and clustering, and prepares the explanation
        objects used by ``plot_overview``, ``plot_visuals``, ``create_rules_txt``,
        and ``print_rules_txt``.  Returns ``self`` to allow method chaining.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            Test dataset.  Must be a pandas DataFrame so that feature names are
            preserved in the explanation.  Use ``pd.DataFrame(arr, columns=names)``
            to wrap a NumPy array if needed.
        idx : int
            Positional (iloc) index of the sample to explain within ``X``.

        Returns
        -------
        self : BellatrexExplain
            The fitted explainer, ready for visualization or text output.

        Raises
        ------
        TypeError
            If ``X`` is not a pandas DataFrame.
        IndexError
            If ``idx`` is out of bounds for ``X``.
        ValueError
            If the sample at ``idx`` contains NaN or infinite values, or if the
            column names of ``X`` do not match the feature names seen during fit.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame. "
                "Wrap your array with: pd.DataFrame(X, columns=feature_names)"
            )

        if not (0 <= idx < len(X)):
            raise IndexError(f"idx={idx} is out of bounds for X with {len(X)} rows.")

        sample = X.iloc[[idx]]

        sample_values = sample.values
        if np.any(np.isnan(sample_values)):
            raise ValueError(f"Sample at idx={idx} contains NaN values.")
        if np.any(np.isinf(sample_values)):
            raise ValueError(f"Sample at idx={idx} contains infinite values.")

        if hasattr(self.clf, "feature_names_in_") and self.clf.feature_names_in_ is not None:
            expected = list(self.clf.feature_names_in_)
            actual = list(X.columns)
            if expected != actual:
                raise ValueError(
                    f"Column names of X do not match the feature names seen during fit.\n"
                    f"Expected: {expected}\n"
                    f"Got:      {actual}"
                )

        if self.ys_oracle is None:
            ys_oracle = None
        else:
            try:
                oracle_length = len(self.ys_oracle)
            except TypeError as exc:
                raise TypeError(
                    "ys_oracle must be an array-like object with one value per row"
                ) from exc
            if oracle_length != len(X):
                raise ValueError(
                    f"ys_oracle has {oracle_length} rows, but the explained X has {len(X)} rows."
                )
            ys_oracle = (
                self.ys_oracle.iloc[idx] if hasattr(self.ys_oracle, "iloc") else self.ys_oracle[idx]
            )

        param_grid = {"n_trees": self.n_trees, "n_dims": self.n_dims, "n_clusters": self.n_clusters}

        for key, value in param_grid.items():
            if not isinstance(value, (list, np.ndarray)):
                param_grid[key] = [value]

        grid_list = list(ParameterGrid(param_grid))
        best_perf = -np.inf

        trees_extract = TreeExtraction(
            self.proj_method,
            self.dissim_method,
            self.feature_represent,
            self.n_trees,
            self.n_dims,
            self.n_clusters,
            self.pre_select_trees,
            self.fidelity_measure,
            self.clf,
            ys_oracle,
            self.set_up,
            sample,
            self.verbose,
        )

        best_params = {
            "n_clusters": 2,
            "n_dims": None,
            "n_trees": 80,
        }  # fallback if all grid candidates fail

        if self.n_jobs == 1:
            for params in grid_list:
                try:
                    candidate = trees_extract.set_params(**params).main_fit()
                    perf = candidate.score(self.fidelity_measure, ys_oracle)
                except (ConvergenceWarning, ValueError) as e:
                    warnings.warn(f"Skipping candidate {params}: {e}")
                    perf = -np.inf

                if self.verbose >= 5:
                    print("params:", params)
                    print(f"fidelity current candidate: {perf:.4f}")

                if perf > best_perf:
                    best_perf = perf
                    best_params = params

            if best_perf == -np.inf:
                raise RuntimeError("No hyperparameter configuration completed successfully.")
        elif self.n_jobs > 1:
            # Thread-based parallelism; speed-up is dataset-dependent.
            def missing_params_dict(given_params, class_instance):
                param_names = class_instance.__init__.__code__.co_varnames[1:]
                param_values = {name: getattr(class_instance, name) for name in param_names}
                missing_params = {
                    key: value for key, value in param_values.items() if key not in given_params
                }
                return missing_params

            provided_params = list(grid_list[0].keys())
            constant_params = missing_params_dict(provided_params, trees_extract)

            def create_btrex_candidate(constant_params, **params):
                return TreeExtraction(**constant_params, **params)

            def run_candidate(
                create_instance_func, fidelity_measure, ys_oracle, constant_params, **params
            ):
                try:
                    etrees_instance = create_instance_func(constant_params, **params)
                    candidate = etrees_instance.main_fit()
                    perf = candidate.score(fidelity_measure, ys_oracle)
                except (ConvergenceWarning, ValueError) as e:
                    warnings.warn(f"Skipping candidate {params}: {e}")
                    perf = -np.inf
                return perf, params

            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(run_candidate)(
                    create_btrex_candidate,
                    self.fidelity_measure,
                    ys_oracle,
                    constant_params,
                    **params,
                )
                for params in grid_list
            )

            perfs, params_list = zip(*results)

            finite_scores = np.flatnonzero(np.isfinite(np.asarray(perfs, dtype=float)))
            if finite_scores.size:
                best_idx = finite_scores[np.argmax(np.asarray(perfs)[finite_scores])]
                best_perf = float(perfs[best_idx])
                best_params = params_list[best_idx]

            if best_perf == -np.inf:
                raise RuntimeError("No hyperparameter configuration completed successfully.")

        tuned_method = trees_extract.set_params(**best_params).main_fit()
        tuned_method.sample_score = tuned_method.score(self.fidelity_measure, ys_oracle)

        # Ensure final_trees_idx and cluster_sizes are not None and are arrays
        if tuned_method.final_trees_idx is not None:
            final_extract_trees = np.array(tuned_method.final_trees_idx)
        else:
            final_extract_trees = np.array([])
        if tuned_method.cluster_sizes is not None:
            final_cluster_sizes = np.array(tuned_method.cluster_sizes)
        else:
            final_cluster_sizes = np.array([])

        if not isinstance(self.clf, RandomSurvivalForest):
            surrogate_pred = np.array([0.0] * self.clf.n_outputs_).reshape(sample.shape[0], -1)
        else:  # Assumed RSF is only single-output
            surrogate_pred = np.array([0.0])

        for tree_idx, cluster_size in zip(final_extract_trees, final_cluster_sizes):
            if len(final_cluster_sizes) > 0:
                cluster_weight = cluster_size / np.sum(final_cluster_sizes)
            else:
                cluster_weight = 0

            surrogate_pred += predict_helper(self.clf[tree_idx], sample.values) * cluster_weight

        surrogate_pred_str = frmt_pretty_print(surrogate_pred, digits_single=4)

        if self.verbose >= 1:
            print("best params:", best_params)
            print(f"Achieved fidelity: {best_perf:.4f}")

        if self.verbose >= 2:
            print(f"final trees indices: {final_extract_trees}")
            print(f"final cluster sizes: {final_cluster_sizes}")

        # Store state for method chaining into plot/txt methods.
        self.sample = X.iloc[[idx]]
        self.sample_index = idx
        self.tuned_method = tuned_method
        self.surrogate_pred_str = surrogate_pred_str

        return self  # enables method chaining

    def plot_overview(
        self, show=True, plot_max_depth=None, colormap=None, plot_gui=False, temp_gui_dir=None
    ):

        if self.sample is None or self.tuned_method is None:
            raise ValueError(
                "Call 'explain()' method first, to generate the explanation and set up the sample."
            )
        sample = self.sample
        tuned_method = self.tuned_method

        if self.verbose >= 0:
            y_pred_orig = predict_helper(self.clf, sample)
            print("Bellatrex prediction:", self.surrogate_pred_str)
            print("Black box prediction: " + frmt_pretty_print(y_pred_orig, digits_single=4))
            print("#" * 58, flush=True)

        if self.verbose >= 4.0:  # print more details in the console:
            if tuned_method.final_trees_idx and tuned_method.cluster_sizes:
                for tree_idx, cluster_size in zip(
                    tuned_method.final_trees_idx, tuned_method.cluster_sizes
                ):
                    rule_print_inline(
                        self.clf[tree_idx],
                        sample,
                        weight=cluster_size / np.sum(tuned_method.cluster_sizes),
                        max_features_print=self.MAX_FEATURE_PRINT,
                    )

        # Set interactive mode to False
        plt.ioff()
        fig, axes = None, None

        # Prepare data and info to be plotted
        plot_kmeans, plot_data_bunch = tuned_method.preselect_represent_cluster_trees()

        if not plot_gui:  # Plot standard overview plots, without interactive features
            if plot_max_depth is not None:
                warnings.warn(
                    f"Max depth for tree visualization = {plot_max_depth} "
                    f"has no effect if plot_gui is set to {plot_gui}"
                )

            fig, axes = plot_preselected_trees(
                plot_data_bunch,
                plot_kmeans,
                tuned_method,
                base_font_size=self.FONT_SIZE,
                colormap=colormap,
            )
            fig.suptitle("Plot overview", fontsize=16)

        else:  # Interactive GUI plotting
            if isinstance(self.clf, EnsembleWrapper):
                raise ValueError(
                    "GUI interface is not compatible with packed EnsembleWrapper yet."
                    "\nPlease use the original sklearn.ensemble class and do not call"
                    "the pack_trained_ensemble function on it."
                )

            from .gui_utils import check_and_import_gui_dependencies

            check_and_import_gui_dependencies()  # raises ImportError if nicegui is absent
            from .nicegui_plots_code import launch_nicegui_window, plot_with_interface

            # A temporary directory is used to store, read and clear files created during
            #  the User interactions, a writable GUI temp dir (no __file__, no site-packages)
            base_dir = self._pick_base_dir(out_dir_name="temp-plots")
            temp_files_dir = (
                temp_gui_dir
                if (isinstance(temp_gui_dir, str) and temp_gui_dir.strip())
                else base_dir
            )
            os.makedirs(temp_files_dir, exist_ok=True)

            # Call plot_with_interface and retrieve the interactive plots
            plots = plot_with_interface(
                plot_data_bunch,
                plot_kmeans,
                tuned_method,
                temp_files_dir=temp_files_dir,
                max_depth=plot_max_depth,
                colormap=colormap,
            )

            if show:
                # Launch the NiceGUI window (blocking until the window is closed).
                launch_nicegui_window(
                    plots,
                    temp_files_dir,
                    render_context={
                        "clf": tuned_method.clf,
                        "sample": tuned_method.sample,
                        "sample_index": self.sample_index,
                        "max_depth": plot_max_depth,
                    },
                )

            # Create a placeholder figure and axes to match the expected output format
            fig = plt.figure(figsize=(8, 6))
            axes = plots  # Use the returned plots as axes-like objects

        return fig, axes  # plt.gcf()

    def _pick_base_dir(self, out_dir_name="explanations-output"):
        """Return the base output directory, honouring the BELLATREX_EXPLAIN_DIR env var."""
        env_dir = os.getenv("BELLATREX_EXPLAIN_DIR")
        if env_dir:
            return env_dir
        return os.path.join(os.getcwd(), out_dir_name)

    def _resolve_output_dir(self, out_dir):
        """
        Resolve *out_dir* to an absolute path.

        - ``None``           → ``_pick_base_dir()``  (env var or ``<cwd>/explanations-output``)
        - absolute path      → used as-is
        - relative path      → resolved relative to the current working directory
        """
        if out_dir is None:
            return self._pick_base_dir()
        if os.path.isabs(out_dir):
            return out_dir
        return os.path.join(os.getcwd(), out_dir)

    def create_rules_txt(self, out_dir=None, out_file=None):
        """
        Write the Bellatrex explanation rules to a pair of ``.txt`` files and return their paths.

        Parameters
        ----------
        out_dir : str or None, default=None
            Directory to write rule files to.

            * ``None`` → ``$BELLATREX_EXPLAIN_DIR`` if set, otherwise
              ``<cwd>/explanations-output``.
            * Absolute path → used as-is.
            * Relative path → resolved relative to the current working directory.
        out_file : str or None, default=None
            Filename for the main rule file.  Only the basename is used; any
            directory portion is ignored.  Defaults to
            ``"Btrex_sample_<sample_index>.txt"``.

        Returns
        -------
        main_path : str
            Path to the main rule file (selected rules only).
        extra_path : str
            Path to the extra rule file (all remaining rules, weight = 0).
        """
        tuned_method = self.tuned_method

        # Ensure the sample is a DataFrame with stable column names.
        if isinstance(self.sample, np.ndarray):
            self.sample = pd.DataFrame(
                self.sample, columns=[f"X_{i}" for i in range(self.sample.shape[1])]
            )

        target_dir = self._resolve_output_dir(out_dir)
        filename = os.path.basename(out_file or f"Btrex_sample_{self.sample_index}.txt")
        main_path = os.path.join(target_dir, filename)
        stem = os.path.splitext(main_path)[0]
        extra_path = f"{stem}_extra.txt"

        os.makedirs(target_dir, exist_ok=True)

        # Write main file (selected rules).
        with open(main_path, "w", encoding="utf-8") as f:
            for tree_idx, clus_size in zip(
                tuned_method.final_trees_idx, tuned_method.cluster_sizes
            ):
                rule_to_file(
                    self.clf[tree_idx],
                    self.sample,
                    clus_size / np.sum(tuned_method.cluster_sizes),
                    self.MAX_FEATURE_PRINT,
                    f,
                )
            f.write(f"Bellatrex prediction: {self.surrogate_pred_str}")

        # Write extra file (all non-selected rules, assigned weight 0).
        with open(extra_path, "w", encoding="utf-8") as f:
            for tree_idx in range(self.clf.n_estimators):
                if tree_idx not in tuned_method.final_trees_idx:
                    rule_to_file(
                        self.clf[tree_idx], tuned_method.sample, 0, self.MAX_FEATURE_PRINT, f
                    )

        # Parse & validate the written files.
        rules, preds, baselines, weights, _ = read_rules(file=main_path, file_extra=extra_path)
        if isinstance(preds[0], list):
            preds = [list(map(float, pred)) for pred in preds]
        if isinstance(baselines[0], list):
            baselines = [list(map(float, baseline)) for baseline in baselines]
        _input_validation(rules, preds, baselines, weights)

        self._last_rules_main_path = main_path
        self._last_rules_extra_path = extra_path

        return main_path, extra_path

    def print_rules_txt(self, out_dir=None, out_file=None):
        """
        Print the Bellatrex explanation rules to stdout.

        If ``out_dir`` and ``out_file`` are both ``None``, the most recently
        created rules file (from ``create_rules_txt``) is read.  Otherwise the
        path is resolved using the same logic as ``create_rules_txt``.

        Parameters
        ----------
        out_dir : str or None, default=None
            Directory where the rules file is located.  Same resolution rules
            as in ``create_rules_txt``.
        out_file : str or None, default=None
            Filename of the rules file.  Defaults to
            ``"Btrex_sample_<sample_index>.txt"``.

        Returns
        -------
        main_path : str
            Path of the file that was printed.

        Raises
        ------
        ValueError
            If no rules file can be found at the resolved path.
        """
        if out_dir is None and out_file is None:
            main_path = getattr(self, "_last_rules_main_path", None)
            if not main_path or not os.path.exists(main_path):
                raise ValueError(
                    "No existing rules file found. Call create_rules_txt() first, "
                    "or pass out_dir / out_file to point to an existing file."
                )
        else:
            target_dir = self._resolve_output_dir(out_dir)
            filename = os.path.basename(out_file or f"Btrex_sample_{self.sample_index}.txt")
            main_path = os.path.join(target_dir, filename)
            if not os.path.exists(main_path):
                raise ValueError(
                    f"No rules file found at '{main_path}'. " "Call create_rules_txt() first."
                )

        with open(main_path, "r", encoding="utf-8") as f:
            rules_text = f.read()
        print("Bellatrex rules (text explanation):")
        print(rules_text)

        return main_path

    def plot_visuals(
        self,
        plot_max_depth=None,
        preds_distr=None,
        conf_level=None,
        tot_digits=4,
        b_box_pred=None,
        keep_files=False,
        out_file=None,
        show=True,
    ):
        """
        Generate a visual explanation of the rules for a single-output task.

        This method creates rule files, parses them, and plots a visual representation
        of the rules extracted by Bellatrex. Intended only for single-output
        setups (e.g., binary classification, regression or survival analysis). For multi-output tasks,
        use `plot_overview()` instead.

        Parameters:
            plot_max_depth (int, optional): Maximum number of rules (depth) to display.
            preds_distr (np.ndarray or Series, optional): Distribution of training predictions,
                used for background histograms or contextual visuals.
            conf_level (float, optional): Confidence level to display for predictions.
            tot_digits (int, default=4): Number of digits to round numeric values in the plot.
            b_box_pred (np.ndarray or Series, optional): Optional black-box prediction reference.
            keep_files (bool, default=False): Whether to keep intermediate `.txt` rule files.
            out_file (str, optional): Custom file path to store generated rule files.
            show (bool, default=True): whether to immediately display the figure.

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
                The figure and axes containing the plotted rule visualization, regardless of `show`.

        Raises:
            ValueError: If called on a multi-output setup, which is not supported by this function.
        """
        tuned_method = self.tuned_method
        multi_output_cases = tuned_method.MSA_KEYS + tuned_method.MTC_KEYS + tuned_method.MTR_KEYS

        if tuned_method.set_up in multi_output_cases:
            raise ValueError(
                f"plot_visuals() is compatible with single-output tasks only,\n"
                f"found '{tuned_method.set_up}'. Use plot_overview() instead"
            )

        out_file, file_extra = self.create_rules_txt()
        rules, preds, baselines, weights, other_preds = read_rules(
            file=out_file, file_extra=file_extra
        )

        _input_validation(rules, preds, baselines, weights)

        if keep_files and self.verbose >= 3:
            print(f"Stored Bellatrex rules in:\n{out_file}")

        if keep_files is False:  # delete txt files after reading stored rules
            os.remove(out_file)
            os.remove(file_extra)
            if self.verbose >= 3:
                print(f"Removed txt files from:\n{os.path.dirname(out_file)}")

        fig, axs = plot_rules(
            rules,
            preds,
            baselines,
            weights,
            max_rulelen=plot_max_depth,
            other_preds=other_preds,
            base_fontsize=self.FONT_SIZE - 1,
            conf_level=conf_level,
            tot_digits=tot_digits,
            # cmap='shap',
            preds_distr=preds_distr,
            b_box_pred=b_box_pred,
        )  # b_box_pred=y_test.iloc[idx]

        if show is True:
            plt.show()

        return fig, axs

    def predict_survival_curve(self, X, idx):
        """
        To be implemented
        """
        if self.set_up != "survival":
            raise ValueError("Input set-up is not a time-to-event!")
        else:
            raise NotImplementedError("predict_survival_curve is not implemented yet.")

    def predict_median_surv_time(self, X, idx):
        """
        To be implemented
        """
        if self.set_up != "survival":
            raise ValueError("Input set-up is not a time-to-event!")
        else:
            raise NotImplementedError("predict_median_surv_time is not implemented yet.")
