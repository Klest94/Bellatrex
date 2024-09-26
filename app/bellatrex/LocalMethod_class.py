
import os
import warnings
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ParameterGrid
from sklearn.utils.validation import check_is_fitted

import sksurv
from sksurv.ensemble import RandomSurvivalForest

from .wrapper_class import EnsembleWrapper
from .utilities import predict_helper
from .visualization_extra import _input_validation
from .TreeExtraction_class import TreeExtraction
from .utilities import plot_preselected_trees, rule_print_inline
from .utilities import rule_to_file, frmt_pretty_print
from .visualisation import read_rules, plot_rules

class BellatrexExplain:

    FONT_SIZE = 14
    MAX_FEATURE_PRINT = 10

    def __init__(self, clf, set_up="auto",
                 force_refit=False,
                 verbose=0,
                 proj_method="PCA",
                 dissim_method="rules",
                 feature_represent="weighted",
                 p_grid = {"n_trees": [0.6, 0.8, 1.0],
                           "n_dims": [2, None],
                           "n_clusters": [1, 2, 3],
                             },

                 pre_select_trees="L2",
                 fidelity_measure="L2",
                 n_jobs=1,
                 ys_oracle=None):

        self.clf = clf #(un)fitted instance of R(S)F
        self.set_up = set_up
        self.force_refit = force_refit
        self.proj_method = proj_method
        self.dissim_method = dissim_method
        self.feature_represent = feature_represent
        self.p_grid = p_grid
        self.pre_select_trees = pre_select_trees
        self.fidelity_measure = fidelity_measure
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.ys_oracle = None

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
            raise ValueError("p_grid parameter is expected to be a dictionary. Found {type(self.p_grid)} instead.")

        default_keys = ["n_trees", "n_dims", "n_clusters"]

        unexpected_keys = [key for key in self.p_grid.keys() if key not in default_keys]
        if unexpected_keys:
            warnings.warn(f"The hyperparameter list contains unexpected keys: {unexpected_keys}. Ignoring them.")

        if "n_trees" not in self.p_grid.keys():
            self.n_trees = [0.6, 0.8, 1.0] # set to default if not existing
        else:
            self.n_trees = self.p_grid["n_trees"]           #CAN BE A LIST

        if "n_dims" not in self.p_grid.keys() or self.p_grid["n_dims"] is None:
            self.n_dims = [None] # set to default if not existing
        else:
            self.n_dims = self.p_grid["n_dims"]             #CAN BE A LIST
            # treat 'all' as None (compatible with sklearn's PCA)
            self.n_dims = [None if x == 'all' else x for x in self.n_dims]

        if "n_clusters" not in self.p_grid.keys():
            self.n_clusters = [1, 2, 3] # set to default if not existing
        else:
            self.n_clusters = self.p_grid["n_clusters"]     # CAN BE A LIST

        if min(self.n_trees) <= 0:
            raise ValueError("n_trees must be all > 0")

        if min(self.n_trees) < 1.0 and max(self.n_trees) > 1.0:
            raise ValueError('The list of n_trees must either indicate a proportion'
                             ' of trees in the (0,1] interval, or indicate the number'
                             ' of tree learners.')

        # Check that the n_trees provided by the user does not exceed the number of total trees
        # in the R(S)F. This works for both a fitted sklearn model and a dictionary

        if max(self.n_trees) > self.clf.n_estimators:
            raise ValueError("\'n_trees\' hyperparater value cannot be greater than n_estimators")

        # if proportion of n_trees is given instead, check correctness and transform to integer values:
        if np.array([isinstance(i, float) for i in self.n_trees]).all(): # if all elements are floats
            if max(self.n_trees) <= 1.0 and min(self.n_trees) > 0: # all elements are in the (0, 1] interval:
                # round to closest integer
                self.n_trees = (np.array(self.n_trees)*self.clf.n_estimators+0.5).astype(int)


    def is_fitted(self): #auxiliary function that returns boolean
        '''
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
        '''

        if isinstance(self.clf, dict): # case of simple, packed dictionary:
            self.clf = EnsembleWrapper(self.clf) # EnsembleWrapper() ensures compatibily of (packed) dictionaries
            return True
        elif isinstance(self.clf, EnsembleWrapper): # case where compatibility is already taken care of:
            return True
        else: #case where full sklearn/sksurv model is given. Check if it is fitted or not
            try:
                check_is_fitted(self.clf) #only with sklearn models (but works with all of them)
                return True
            except Exception: # if 'check_is_fitted' throws exception, we need it to output 'False'
                return False
        # Note that from sklearn 1.3. we can simply use return _is_fitted(self.clf) # returns boolean already


    def fit(self, X, y):
        '''
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
        - If 'verbose' is set to 0 or higher, it will print the fitting status.
        - Automatically determines the prediction task ('set_up') based on classifier properties.
        '''

        if self.force_refit is False and self.is_fitted():
            if self.verbose >=0:
                print("Model is already fitted, building explanation.")
        else:
            if self.verbose >= 0:
                print("Fitting the model...", end='')
            self.clf.fit(X, y, self.n_jobs)
            if self.verbose >= 0:
                print("fitting complete")

        # then check whether the input grid values are admissible
        self._validate_p_grid()

        if self.verbose >= 2:
            print(f"oracle_sample is: {self.ys_oracle}")

        if self.set_up == "auto": # automatically determine scenario based on fitted classifier

            if isinstance(self.clf, EnsembleWrapper):

                if self.clf.ensemble_class == "RandomForestClassifier" and self.clf.n_outputs_ == 1:
                    self.set_up = "binary"
                elif self.clf.ensemble_class == "RandomForestClassifier" and self.clf.n_outputs_ > 1:
                    self.set_up = "multi-label"
                elif self.clf.ensemble_class == "RandomForestRegressor" and self.clf.n_outputs_ == 1:
                    self.set_up = "regression"
                elif self.clf.ensemble_class == "RandomForestRegressor" and self.clf.n_outputs_ > 1:
                    self.set_up = "multi-target"
                elif self.clf.ensemble_class == "RandomSurvivalForest" and y.shape[1] == 2: #ideally, check for 1 field boolean and one field numeric
                    self.set_up = "survival"
                elif self.clf.ensemble_class == "RandomSurvivalForest" and y.shape[1] > 2:
                    raise ValueError(f"Shape of recarray labels {y.shape} implies multi-output survival analysis, "
                                     "which is not implemented yet")
                else:
                    raise ValueError(f"Classifier {self.clf.ensemble_class} not compatible"
                                     "with \'auto\' set-up selection. PLease select the set-up manually")

            elif isinstance(self.clf, sklearn.ensemble.RandomForestClassifier):
                if self.clf.n_outputs_ == 1:
                    self.set_up = 'binary'
                else:
                    self.set_up = 'multi-label'
            elif isinstance(self.clf, sklearn.ensemble.RandomForestRegressor):
                if np.array(y).ndim < 2 or self.clf.n_outputs_ == 1:
                    self.set_up = 'regression'
                else:
                    self.set_up = 'multi-target'
            elif isinstance(self.clf, sksurv.ensemble.forest.RandomSurvivalForest):
                if self.clf.n_outputs_ == self.clf.unique_times_.shape[0]:
                    self.set_up = 'survival'
                else:
                    self.set_up = 'multi-variate-sa'
                    raise ValueError("n_outputs_ shape != unique_times_ shape:"
                                     f"{self.clf.n_outputs_.shape} != {self.clf.unique_times_.shape} \n"
                                     "Note that multi-event Survival analysis is not supported yet")
            else:
                raise ValueError("Provided model is not recognized or compatible with Bellatrex:", self.clf)

            if self.verbose > 0:
                print(f"Automatically setting prediction task to: {self.set_up}")

        return self


    def explain(self, X, idx):

        sample = X.iloc[[idx]]

        if self.ys_oracle is not None:
            self.ys_oracle = self.ys_oracle.iloc[idx]  # pick needed one

        param_grid = {
            "n_trees": self.n_trees,
            "n_dims": self.n_dims,
            "n_clusters": self.n_clusters
        }

        for key, value in param_grid.items():
            if not isinstance(value, (list, np.ndarray)):
                param_grid[key] = [value]

        grid_list = list(ParameterGrid(param_grid))
        best_perf = -np.inf

        trees_extract = TreeExtraction(
            self.proj_method, self.dissim_method,
            self.feature_represent,
            self.n_trees, self.n_dims, self.n_clusters,
            self.pre_select_trees,
            self.fidelity_measure,
            self.clf,
            self.ys_oracle,
            self.set_up, sample, self.verbose
        )

        best_params = {"n_clusters": 2, "n_dims": None, "n_trees": 80} #default combination, in case everything fails

        if self.n_jobs == 1:
            for params in grid_list:
                try:
                    candidate = trees_extract.set_params(**params).main_fit()
                    perf = candidate.score(self.fidelity_measure, self.ys_oracle)
                except ConvergenceWarning as e:
                    warnings.warn(f'Something went wrong (ConvergenceWarning). {e}, skipping candidate: {params}')
                    perf = -np.inf
                # except KeyboardInterrupt as e:
                #     raise e

                if self.verbose >= 5:
                    print("params:", params)
                    print(f"fidelity current candidate: {perf:.4f}")

                if perf > best_perf:
                    best_perf = perf
                    best_params = params

            if best_perf == -np.inf:
                warnings.warn("The GridSearch did not find any meaningful configuration,"
                            " setting default parameters")
        elif self.n_jobs > 1:
            warnings.warn('Multi-processing is not optimized in this version and the'
                        ' speed-up gain is marginal. Set n_jobs = 1 to avoid this warning')

            def missing_params_dict(given_params, class_instance):
                param_names = class_instance.__init__.__code__.co_varnames[1:]
                param_values = {name: getattr(class_instance, name) for name in param_names}
                missing_params = {key: value for key, value in param_values.items() if key not in given_params}
                return missing_params

            provided_params = list(grid_list[0].keys())
            class_instance = trees_extract
            constant_params = missing_params_dict(provided_params, class_instance)

            def create_btrex_candidate(constant_params, **params):
                return TreeExtraction(**constant_params, **params)

            def run_candidate(create_instance_func, fidelity_measure, ys_oracle, constant_params, **params):
                candidate = None
                try:
                    etrees_instance = create_instance_func(constant_params, **params)
                    candidate = etrees_instance.main_fit()
                    perf = candidate.score(fidelity_measure, ys_oracle)
                except Exception as e:
                    warnings.warn(f'Something went wrong ({e}), skipping candidate: {params}')
                    perf = -np.inf
                return perf, params

            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(run_candidate)(create_btrex_candidate, self.fidelity_measure,
                                    self.ys_oracle, constant_params, **params)
                for params in grid_list)

            perfs, params_list = zip(*results)

            if best_perf > -np.inf:
                best_idx = np.argsort(perfs)[::-1][0]
                best_perf = perfs[best_idx]
                best_params = params_list[best_idx]

            if best_perf == -np.inf:
                warnings.warn("The GridSearch did not find any functioning hyperparameter"
                              " configuration, setting default configuration")

        tuned_method = trees_extract.set_params(**best_params).main_fit()
        tuned_method.sample_score = tuned_method.score(self.fidelity_measure, self.ys_oracle)

        final_extract_trees = tuned_method.final_trees_idx
        final_cluster_sizes = tuned_method.cluster_sizes

        if not isinstance(self.clf, RandomSurvivalForest):
            surrogate_pred = np.array([0.0] * self.clf.n_outputs_).reshape(sample.shape[0], -1)
        else: # Assumed RSF is only single-output
            surrogate_pred = np.array([0.0])

        for tree_idx, cluster_size in zip(final_extract_trees, final_cluster_sizes):
            cluster_weight = cluster_size / np.sum(final_cluster_sizes)
            surrogate_pred += predict_helper(self.clf[tree_idx], sample.values) * cluster_weight

        tuned_method.prediction = surrogate_pred

        surrogate_pred_str = frmt_pretty_print(surrogate_pred, digits_single=4)

        if self.verbose >= 1:
            print("best params:", best_params)
            print(f"Achieved fidelity: {best_perf:.4f}")

        if self.verbose >= 2:
            print(f"final trees indices: {final_extract_trees}")
            print(f"final cluster sizes: {final_cluster_sizes}")

        # Store for use in plot_overview (offer possibility for method chaining)
        self.sample = X.iloc[[idx]]
        self.sample_iloc = idx
        self.tuned_method = tuned_method
        self.surrogate_pred_str = surrogate_pred_str

        return self#, tuned_method, surrogate_pred_str


    def plot_overview(self, show=True,
                      plot_max_depth=None,
                      colormap=None, plot_gui=False):

        sample =  self.sample
        tuned_method = self.tuned_method

        if self.verbose >= 0:
            y_pred_orig = predict_helper(self.clf, sample)
            print('Bellatrex prediction:', self.surrogate_pred_str)
            print('Black box prediction: ' + frmt_pretty_print(y_pred_orig, digits_single=4))
            print('#' * 58, flush=True)

        if self.verbose >= 4.0: # print more details in the console:
            for tree_idx, cluster_size in zip(tuned_method.final_trees_idx, tuned_method.cluster_sizes):
                rule_print_inline(self.clf[tree_idx], sample,
                                  weight=cluster_size/ np.sum(tuned_method.cluster_sizes),
                                  max_features_print=self.MAX_FEATURE_PRINT)

        # plotting with matplotlib here:
        plt.ioff() # Set interactive mode to False
        fig, axes = [None, None]

        # prepare data and info to be plotted:
        plot_kmeans, plot_data_bunch = tuned_method.preselect_represent_cluster_trees()

        if plot_gui is False: # plot stadnard overview plots, without interactive features:

            if plot_max_depth is not None:
                warnings.warn(f"Max depth for tree visualization = {plot_max_depth}"
                              f" has no effect if set_gui is set to {plot_gui}")

            fig, axes = plot_preselected_trees(plot_data_bunch, plot_kmeans, tuned_method,
                                               base_font_size=self.FONT_SIZE,
                                               colormap=colormap)
            fig.suptitle("Plot overview", fontsize=16)

            if show is True and fig is not None:
                return plt.show() # seems to work.
                # An alternative (depending on the plotting backend) is fig.show()

        if plot_gui is True:

            matplotlib.use('Agg')
            print('Matplotlib set in a non-interactive backend, with: \'matplotlib.use(\'Agg\')\'')
            from .gui_utils import check_and_import_gui_dependencies
            dearpygui, dearpygui_ext = check_and_import_gui_dependencies()
            from .gui_plots_code import plot_with_interface

            if show is False:
                warnings.warn("Plots are shown immediately while in an interactive session (plot_gui = True).\n"
                            "The variable show = False is therefore ignored.")

            # A 'temporary' directory is used to store, read and clear files created during the User interactions:
            current_file_dir = os.path.dirname(os.path.abspath(__file__)) # app/bellatrex
            temp_files_dir = os.path.join(current_file_dir, "temp_files") # app/bellatrex/temp_files
            os.makedirs(temp_files_dir, exist_ok=True)

            plot_with_interface(plot_data_bunch, plot_kmeans, tuned_method, temp_files_dir,
                                max_depth=plot_max_depth, colormap=colormap)
            # REMARK: the temp files are deleted with os.remove in plot_with_interface, above

        return fig, axes # plt.gcf()



    def plot_visuals(self, plot_max_depth=None, preds_distr=None,
                     conf_level=None, tot_digits=4,
                     b_box_pred=None, keep_files=False,
                     out_file=None, show=True):
        '''
        - create rules in txt
        - read rules from txt files
        - parse them and plot them
        '''
        tuned_method = self.tuned_method
        multi_output_cases = tuned_method.MSA_KEYS + tuned_method.MTC_KEYS + tuned_method.MTR_KEYS

        if tuned_method.set_up in multi_output_cases:
            raise ValueError(f"plot_overview() is compatible with single-output tasks only,\n"
                             f"found \'{tuned_method.set_up}\'")

        # fixing columns names, making sure they exist:
        if isinstance(self.sample, np.ndarray):
            self.sample = pd.DataFrame(self.sample)
            self.sample.columns = [f"X_{i}" for i in range(len(self.sample.columns))]

        if out_file is None:
            current_file_dir = os.path.dirname(os.path.abspath(__file__)) # app/bellatrex
            temp_files_dir = os.path.join(current_file_dir, "explanations_text") # app/bellatrex/explanations_text
            os.makedirs(temp_files_dir, exist_ok=True)
            out_file = os.path.join(temp_files_dir, f"Btrex_sample_{self.sample_iloc}.txt")
        else:
            out_file = os.path.join(os.getcwd(), out_file)


        with open(out_file, 'w+', encoding="utf8") as f: #re-initialize file: overwrite in case
            pass

        with open(out_file, 'a+', encoding="utf8") as f:
            for idx, clus_size in zip(tuned_method.final_trees_idx, tuned_method.cluster_sizes):
                rule_to_file(self.clf[idx], self.sample,
                             clus_size/np.sum(tuned_method.cluster_sizes),
                             self.MAX_FEATURE_PRINT, f)

            f.write(f"Bellatrex prediction: {self.surrogate_pred_str}")
            f.close()

        file_extra = out_file.replace('.txt', '_extra.txt')

        with open(file_extra, 'w+', encoding="utf8") as f:
            pass

        with open(file_extra, 'a+', encoding="utf8") as f:
            for idx in range(self.clf.n_estimators):
                if idx not in tuned_method.final_trees_idx:
                    rule_to_file(self.clf[idx], tuned_method.sample, 0,
                                    self.MAX_FEATURE_PRINT, f)
            f.close()


        rules, preds, baselines, weights, other_preds = read_rules(
                            file=out_file, file_extra=file_extra)

        _input_validation(rules, preds, baselines, weights)

        if keep_files and self.verbose >= 3:
            print(f"Stored Bellatrex rules in:\n{out_file}")

        if keep_files is False: # delete txt files after reading stored rules
            os.remove(out_file)
            os.remove(file_extra)

        fig, axs = plot_rules(rules, preds, baselines, weights,
                              max_rulelen=plot_max_depth, other_preds=other_preds,
                              base_fontsize=self.FONT_SIZE-1,
                              conf_level=conf_level,
                              tot_digits=tot_digits,
                              #cmap='shap',
                              preds_distr=preds_distr, b_box_pred=b_box_pred)#b_box_pred=y_test.iloc[idx]

        if show is True:
            plt.show()
        else:
            return fig, axs




    def predict_survival_curve(self, X, idx):
        '''
        To be implemented
        '''
        if not self.set_up in ["surv", "survival"]:
            raise ValueError("Input set-up is not a time-to-event!")
        else:
            raise ValueError("Not implemented yet")

    def predict_median_surv_time(self, X, idx):
        '''
        To be implemented
        '''
        if not self.set_up in ["surv", "survival"]:
            raise ValueError("Input set-up is not a time-to-event!")
        else:
            raise ValueError("Not implemented yet")
