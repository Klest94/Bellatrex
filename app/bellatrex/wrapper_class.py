# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:55:55 2023

@author:       Klest Dedja
@institution:  KU Leuven
"""
import numpy as np
import pandas as pd
import warnings
from scipy.sparse import csr_matrix, hstack

from sklearn.utils.validation import check_is_fitted

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest


def pack_trained_ensemble(clf, set_up="auto", time_to_bin=None):
    """
    Packs a trained ensemble model into a dictionary format compatible with scikit-learn.

    Parameters
    ----------
    clf : RandomForestRegressor, RandomForestClassifier, RandomSurvivalForest
        The trained ensemble classifier to be packed. The classifier must be one of the supported types:
        RandomForestRegressor, RandomForestClassifier, or RandomSurvivalForest.

    set_up : str, optional, default="auto"
        Specifies the format in which the tree data should be output. This argument is passed to the
        `tree_to_dict` function. The default value is "auto".

    time_to_bin : float or int, optional
        This parameter is specific to `RandomSurvivalForest`. If provided, it will be passed to the
        `tree_to_dict` function for each tree. If `time_to_bin` is provided for any other classifier
        type, a warning will be issued, and the parameter will be ignored.

    Returns
    -------
    dict
        A dictionary representation of the packed ensemble model. This format is compatible with scikit-learn's
        model loading functions.

    Raises
    ------
    ValueError
        If the `clf` object is not one of the supported classifier types.

    Notes
    -----
    This function first checks if the classifier is fitted using the `check_is_fitted` method. The individual trees
    of the ensemble are then converted into a dictionary format using the `tree_to_dict` function and stored in a
    list. Finally, the list of tree dictionaries is converted into a model dictionary using `tree_list_to_model`.
    """

    if not isinstance(clf, (RandomForestRegressor, RandomForestClassifier, RandomSurvivalForest)):
        raise ValueError(f"Incompatible classifier object, found: {type(clf)}")

    check_is_fitted(clf) # if not fitted, exception is raised

    if time_to_bin is not None and not isinstance(clf, RandomSurvivalForest):
        warnings.warn(f"time_to_bin parameter is ignored for any estimator other than RandomSurvivalForest.\n"
                        f"Found: {type(clf)}")

    tree_list = []
    for t in range(clf.n_estimators):
        tree_dict = tree_to_dict(clf, t,
                                 output_format=set_up,
                                 time_to_bin=time_to_bin)
        tree_list.append(tree_dict)
    # load the model in a dict format somewhat compatible with scikit-learn:
    clf_out = tree_list_to_model(tree_list)

    return clf_out

class EnsembleWrapper:
    '''
    This class serves as a wrapper for compatibility with
    tree ensemble models stored as a list of dictionaries.
    see "Example of loading a custom tree model into SHAP" in
    https://shap.readthedocs.io/en/stable/tabular_examples.html
    (link verified as of July 2024)
    It is designed to be compatible with:\
         RandomForestClassifier,
        RandomForestRegressor, and
        RandomSurvivalForest.
    It therefore assumes that the predictions are AVERAGED in the ensemble step
    '''
    class Estimator:

        class Tree_:
            def __init__(self, feature, n_node_samples, children_left,
                         children_right, threshold, value,
                         feature_names_in_, n_features_in_, learner_class):
                self.feature = feature
                self.n_node_samples = n_node_samples
                self.children_left = children_left
                self.children_right = children_right
                self.threshold = threshold
                self.value = value
                self.feature_names_in_ = feature_names_in_
                self.n_features_in_ = n_features_in_
                self.learner_class = learner_class

            @property
            def n_outputs_(self):
                return len(self.value[0])

            @property
            def node_count(self):
                return len(self.feature)

            def apply(self, X):
                n_samples = X.shape[0]
                node_indices = []
                for i in range(n_samples):
                    node_indices.append(self._apply_tree(0, X[i, :]))
                return node_indices


            def _apply_tree(self, node_index, sample):
                if self.children_left[node_index] == -1 and self.children_right[node_index] == -1:
                    return [node_index]

                path_indices = [node_index]
                if sample[self.feature[node_index]] <= self.threshold[node_index]:
                    path_indices.extend(self._apply_tree(self.children_left[node_index], sample))
                else:
                    path_indices.extend(self._apply_tree(self.children_right[node_index], sample))

                return path_indices


            def decision_path(self, X):
                if isinstance(X, pd.DataFrame):
                    X = X.to_numpy()
                node_indices_list = self.apply(X)
                data, indices, indptr = [], [], [0]
                for sample_nodes in node_indices_list:
                    data.extend([1] * len(sample_nodes))
                    indices.extend(sample_nodes)
                    indptr.append(len(indices))
                return csr_matrix((data, indices, indptr), shape=(X.shape[0], self.node_count))

        # this is the __init__ of the Estimator class (intermediate layer)
        def __init__(self, tree_dict):
            self.tree_ = self.Tree_(  # ORDER IS IMPORTANT! Check how class Tree_ is initialized
                # The .get method returns None if key is missing
                tree_dict.get('features'),
                tree_dict.get('node_sample_weight'),
                tree_dict.get('children_left'),
                tree_dict.get('children_right'),
                tree_dict.get('thresholds'),
                tree_dict.get('values'), # needs to be singular: .value!
                tree_dict.get('feature_names_in_'),
                tree_dict.get('n_features_in_'),
                tree_dict.get('learner_class'),
                )
            self.n_outputs_ = self.tree_.n_outputs_ # Inferred from underlying tree_.
            self.learner_class = self.tree_.learner_class


        def predict(self, X):
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            leaf_indices = self.tree_.apply(X)
            n_samples = X.shape[0]
            predictions = np.zeros((n_samples, self.n_outputs_))

            for i in range(n_samples):
                leaf_idx = leaf_indices[i][-1]  # Assuming the last node is the leaf
                predictions[i, :] = self.tree_.value[leaf_idx]

            return predictions

        def decision_path(self, X):
            return self.tree_.decision_path(X)

    # init EnsembleWrapper here (most external class)
    def __init__(self, clf_dict):
        self.estimators_ = [self.Estimator(tree) for tree in clf_dict['trees']]
        self.n_outputs_ = self.estimators_[0].n_outputs_
        self.n_estimators = len(self.estimators_)
        self.feature_names_in_ = clf_dict['trees'][0]['feature_names_in_']
        self.n_features_in_ = clf_dict['trees'][0]['n_features_in_']
        self.ensemble_class = clf_dict['ensemble_class']

        if 'unique_times_' in clf_dict['trees'][0]:
            self.unique_times_ = clf_dict['trees'][0]['unique_times_']

    def __getitem__(self, index):
        return self.estimators_[index]

    def predict(self, X): # the output format depends on what was stored in the original dict
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        n_samples = X.shape[0]
        ensemble_predictions = np.zeros((n_samples, self.n_outputs_))

        for estimator in self.estimators_: #sum over learners, stored dictionaries assume this
            ensemble_predictions += estimator.predict(X)

        return ensemble_predictions / self.n_estimators # assuming NON additive ensemble here


    def decision_path(self, X):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        all_paths = []
        n_nodes_ptr = [0]

        for estimator in self.estimators_:
            path_csr = estimator.decision_path(X)
            all_paths.append(path_csr)
            n_nodes_ptr.append(n_nodes_ptr[-1] + path_csr.shape[1])
        all_paths_csr = hstack(all_paths).tocsr()

        return all_paths_csr, np.array(n_nodes_ptr)



def tree_to_dict(clf_obj, idx, output_format, time_to_bin=None):
    '''
    Compatible with single output trees only, at the moment.
    compatible with SurvivalTree learners of a RandomSurvivalForest
    (scikit-survival 0.21)
    with DecisionTreeClassifier and DecisionTreeRegressor from RandomForests
    (scikit-learn 1.2.2)
    '''
    tree_obj = clf_obj[idx]
    tree = tree_obj.tree_

    tree_dict = {
        "children_left" : tree.children_left,
        "children_right" : tree.children_right,
        "children_default" : tree.children_right.copy(), # to be changed when sklearn can handle missing values
        "features" : tree.feature,
        "thresholds" : tree.threshold,
        "node_sample_weight": tree.weighted_n_node_samples,
        "feature_names_in_": getattr(clf_obj, "feature_names_in_", None),
        "n_features_in_": getattr(tree_obj, "n_features_in_", None),
        "unique_times_": getattr(tree_obj, "unique_times_", None),
        "is_event_time_": getattr(tree_obj, "is_event_time_", None),
        "random_state": getattr(tree_obj, "random_state", None),
        "ensemble_class": clf_obj.__class__.__name__,
        "learner_class": tree_obj.__class__.__name__
    }

    tree_dict['output_format'] = output_format

    if isinstance(tree_obj, SurvivalTree):

        if tree_dict['unique_times_'] is None and output_format not in ['probability']:
            raise KeyError('Missing \'unique_times_\' in the tree ensemble.')

        if time_to_bin is None and tree_dict['unique_times_'] is not None: # select median time interval
            time_to_bin = tree_dict['unique_times_'][len(tree_dict['unique_times_'])//2]

        if output_format == "predict_survival_curve":
            tree_dict["values"] = tree.value[:,:, 1]

        elif output_format in ["hazard", "auto"]:
            # sum CHF over unique_times_ that have an event (imitating SurvivalTree .predict)
            tree_dict["values"] = np.sum(tree.value[:,tree_dict['is_event_time_'], 0],
                                          axis=1).reshape(-1, 1)

        elif output_format in ["survival", 'time-to-event']:

            tree_dict["values"] = np.trapz(tree.value[:,:, 1],
                                           tree_obj.unique_times_,
                                           axis=1).reshape(-1, 1) # integrate S(t) over unique_times_
            # output shape: (n_nodes, 1) with E[S(t)] at each node

        elif output_format == "probability":
            # pick last "False" index before "True" appears
            index_T = np.argmax(tree_obj.unique_times_ > time_to_bin)-1
            # it DOES work when all times are > T_bin, as it will again select -1
            if min(tree_obj.unique_times_) > time_to_bin:
                index_T = 0
            # probability of experiencing the event by t=2 -> P(t) = 1 - S(t)
            tree_dict["values"] = 1 - tree.value[:,index_T, 1].reshape(-1, 1)
            # Why was the reshape needed? Why single-element arrays like this? Hmm...

        else:
            raise ValueError('Input not recognized. Double check')

    # Decision Tree Classifier case here:
    elif isinstance(tree_obj, DecisionTreeClassifier) and tree_obj.n_outputs_ == 1 \
        and output_format in ["probability", "auto"]:
        partials = tree.value[:,0,:] # output for 2 classes, now take average
        tree_dict["values"] = (partials[:,1] / (partials[:,0] + partials[:,1])).reshape(-1, 1)

    elif isinstance(tree_obj, DecisionTreeClassifier) and tree_obj.n_outputs_ > 1 \
        and output_format in ["probability", "auto"]:
        partials = tree.value
        tree_dict["values"] = (partials[:,:,1] / (partials[:,:,0] + partials[:,:,1]))


    elif isinstance(tree_obj, DecisionTreeRegressor) and tree_obj.n_outputs_ == 1 \
        and output_format in ["probability", "auto"]:
        tree_dict["values"] = tree.value[:,0,0].reshape(-1, 1) # output (n_nodes, 1)

    elif isinstance(tree_obj, DecisionTreeRegressor) and tree_obj.n_outputs_ > 1 \
        and output_format in ["probability", "auto"]:
        tree_dict["values"] = tree.value[:,:,0] # output (n_nodes, n_outputs_)

    else:
        raise ValueError(f"Combination of learner \'{tree_obj.__class__.__name__}\' and"
                         f" scenario \'{output_format}\' is not recognized")

    tree_dict["base_offset"] = tree_dict['values'][0] # root node (0) prediction

    return tree_dict


def tree_list_to_model(tree_list):
    '''
    given each learner is stored as a dict, create a list of such dictionaries.
    NOTE that this is NOT directly compatible with the SHAP library,
    which assumes that the input custom tree models are additive

    To use SHAP, the user needs to divide the predicted values accordingly.
    See commmented lines below
    '''
    # for t in tree_list: # divide here
    #     t["values"] = t["values"] / len(tree_list)

    base_offset = np.mean(np.array([t['base_offset'] for t in tree_list]))

    assert len(set([t['output_format'] for t in tree_list])) == 1 # consistency check
    assert len(set([t['ensemble_class'] for t in tree_list])) == 1 # consistency check


    output_format = tree_list[0]['output_format']
    ensemble_class = tree_list[0]['ensemble_class']


    model_as_dict = {
        "trees": tree_list, #list of dicts
        "base_offset": base_offset, # single value (average of array)
        "output_format": output_format,
        "ensemble_class": ensemble_class,
        "input_dtype": np.float32,
        "internal_dtype": np.float32}

    return model_as_dict
