
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:42:47 2021

@author: u0135479
"""

import numpy as np
import warnings

def add_emergency_noise(tree_matrix, noise_level=1e-5):
    warnings.warn("MDS matrix has rank 0")
    noise = np.random.standard_normal(size=tree_matrix.shape)
    # now make noise for distance matrix symmetric:
    for i in range(tree_matrix.shape[0]):
        for j in range(i, tree_matrix.shape[1]):
            if j == i:
                noise[i,j] = 0
            else:
                noise[i,j] = noise[j,i]
    return tree_matrix + noise*noise_level
    
    
def count_rule_length(clf, idx1, sample): # it's a SIMILARITY measure

    #DTree1 = clf.estimators_[idx1].tree_ #does it work for both binary and survival data??
    #for t, idx in zip([idx1, idx2], [0,1]):         
    #DT_splits = clf[idx1].tree_.feature
    #surv_tree = clf[idx1]
    # keep only nodes ( indeces) that go through sample's path
    
    if hasattr(clf, "n_estimators"): # ensemble case
    
        DT_path = clf[idx1].tree_.decision_path(sample.to_numpy().reshape(1,-1).astype(np.float32))
        DT_path  = DT_path.toarray().reshape(-1)
        
    else: # single learner case
    
        DT_path = clf.tree_.decision_path(sample.to_numpy().reshape(1,-1).astype(np.float32))
    
    return np.sum(DT_path)-1

    
    
def tree_splits_to_vector(clf, idx1, split_weight=None): # it's a SIMILARITY measure

    DTree1 = clf.estimators_[idx1].tree_ #does it work for both binary and survival data??
    # store feature splits, excluding leafs which have none ( == -1)
    the_splits = DTree1.feature[DTree1.feature > -1 ]
    if split_weight == "by_samples":
        # store relative n_smaples that go across each (interbal) node
        the_weights = (DTree1.n_node_samples/DTree1.n_node_samples[0])[DTree1.feature > -1 ]
    elif split_weight == "simple":
        the_weights = None # no weighting (all equal weights)
    else:
        raise KeyError("split_weight_style = \'{}\' no recognized,\
                 accpeted values are \'simple\' and  \'by_samples\'.")
    fe1 = np.bincount(the_splits, weights=the_weights) 
    # PROBLEM: some features might be missing (at the tail), fill with zeros
    # does this work for both binary and survival?
    tree_vector = np.zeros(clf.n_features_in_)
    #the bincount will drop all features with no splits that come after the last feature with at least one split
    # we need ot inlcude them with extra 0-s at the end to preserve vector lenght. We achieve it by doing the following:
    tree_vector[:len(fe1)] += fe1

    return tree_vector
    
    
def rule_splits_to_vector(clf, idx1, feature_represent, sample): # it's a SIMILARITY measure

    DTree1 = clf.estimators_[idx1].tree_ #does it work for both binary and survival data??

    #for t, idx in zip([idx1, idx2], [0,1]):         
    DT_splits = clf[idx1].tree_.feature
    #surv_tree = clf[idx1]
    
    # keep only nodes ( indeces) that go through sample's path
    DT_path = clf[idx1].tree_.decision_path(sample.to_numpy().reshape(1,-1).astype(np.float32))
    DT_path  = DT_path.toarray().reshape(-1)

    # store feature splits along path ( exclude last element: it's a leaf!)
    path_splits = [DT_splits[i] for i in range(len(DT_splits)) if DT_path[i] > 0][:-1]
    
    if feature_represent in ["by_samples", "weighted"]:
        # store weights across all (internal) PATH nodes
        # assign weight proportional to n_samples going through
        the_weights = (DTree1.n_node_samples/DTree1.n_node_samples[0])[DT_path > 0 ][:-1]
    elif feature_represent == "simple":
        the_weights = None # no weighting (all equal weights)
    else:
        raise KeyError("split_weight_style = \'{}\' no recognized,\
                  accpeted values are \'simple\' or  \'by_samples\' and \'weighted\' .")
    
    rule_vec = np.bincount(path_splits, weights=the_weights)

    # pad rule vector with zero's for tails features have no splititng)
    rule_vector = np.zeros(clf.n_features_in_) # rule_representation
    rule_vector[:len(rule_vec)] += rule_vec
    
    return rule_vector
