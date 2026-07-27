# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:42:47 2021

@author: u0135479
"""

import warnings

import numpy as np


def add_emergency_noise(tree_matrix, noise_level=1e-5):
    warnings.warn("MDS matrix has rank 0")
    noise = np.random.standard_normal(size=tree_matrix.shape)
    # now make noise for distance matrix symmetric:
    for i in range(tree_matrix.shape[0]):
        for j in range(i, tree_matrix.shape[1]):
            if j == i:
                noise[i, j] = 0
            else:
                noise[i, j] = noise[j, i]
    return tree_matrix + noise * noise_level


def tree_splits_to_vector(clf, idx1, split_weight=None):  # it's a SIMILARITY measure

    the_tree = clf.estimators_[idx1].tree_  # does it work for both binary and survival data??
    # store feature splits, excluding leafs which have none ( == -1)
    the_splits = the_tree.feature[the_tree.feature > -1]
    if split_weight == "by_samples":
        # store relative n_samples that go across each (internal) node
        the_weights = (the_tree.n_node_samples / the_tree.n_node_samples[0])[the_tree.feature > -1]
    elif split_weight == "simple":
        the_weights = None  # no weighting (all equal weights)
    else:
        raise KeyError(
            "split_weight_style = '{}' not recognized,\
                 accepted values are 'simple' and  'by_samples'."
        )
    tree_vector_no_pad = np.bincount(the_splits, weights=the_weights)
    # PROBLEM: some features might be missing (at the tail), fill with zeros
    # does this work for both binary and survival?
    tree_vector = np.zeros(clf.n_features_in_)
    # the bincount will ignore features with no splits that come after
    # the last feature with at least one split
    # we need to include them with extra 0-s to preserve vector length
    tree_vector[: len(tree_vector_no_pad)] += tree_vector_no_pad

    return tree_vector


def rule_splits_to_vector(clf, idx1, feature_represent, sample):  # it's a SIMILARITY measure

    the_tree = clf.estimators_[idx1].tree_  # does it work for both binary and survival data??

    if not isinstance(sample, np.ndarray):  # pd.Series or smth else ( list? or..?)
        sample = sample.to_numpy()

    # for t, idx in zip([idx1, idx2], [0,1]):
    tree_splits = clf[idx1].tree_.feature
    # surv_tree = clf[idx1]

    # keep only nodes ( indices) that go through sample's path
    tree_path = clf[idx1].tree_.decision_path(sample.reshape(1, -1).astype(np.float32))
    tree_path = tree_path.toarray().reshape(-1)

    # store feature splits along path ( exclude last element: it's a leaf!)
    path_splits = [tree_splits[i] for i in range(len(tree_splits)) if tree_path[i] > 0][:-1]

    if feature_represent in ["by_samples", "weighted"]:
        # store weights across all (internal) PATH nodes
        # assign weight proportional to n_samples going through
        the_weights = (the_tree.n_node_samples / the_tree.n_node_samples[0])[tree_path > 0][:-1]
    elif feature_represent == "simple":
        the_weights = None  # no weighting (all equal weights)
    else:
        raise KeyError(
            "split_weight_style = '{}' not recognized,\
                  accepted values are 'simple' or  'by_samples' and 'weighted' ."
        )

    rule_vec_no_pad = np.bincount(path_splits, weights=the_weights)

    # pad rule vector with zero's for tail features that have no splitting
    rule_vector = np.zeros(clf.n_features_in_)  # rule_representation
    rule_vector[: len(rule_vec_no_pad)] += rule_vec_no_pad

    return rule_vector


def tree_to_vector(clf, idx, method="trees", feature_represent=None, sample=None):
    """Unified wrapper to obtain a vector representation for a tree or a rule-path.

    - method: 'trees' for full-tree split representation, 'rules' for sample path.
    """
    if method == "trees":
        # default split_weight interpreted as 'by_samples' for trees
        return tree_splits_to_vector(clf, idx, split_weight="by_samples")
    elif method == "rules":
        if feature_represent is None:
            feature_represent = "weighted"
        if sample is None:
            raise ValueError("sample must be provided when method='rules'")
        return rule_splits_to_vector(clf, idx, feature_represent, sample)
    else:
        raise ValueError(f"Unknown method {method}")


def rules_to_vector(clf, idx, feature_represent, sample):
    """Alias for rule_splits_to_vector for clearer external API."""
    return rule_splits_to_vector(clf, idx, feature_represent, sample)
