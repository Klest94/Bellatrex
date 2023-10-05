import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA
import matplotlib as mpl
import pylab
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
# from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from sklearn import tree
import sklearn
import sksurv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.tree import SurvivalTree
from sklearn.tree import _tree
from scipy.sparse import csr_matrix, hstack

class EnsembleWrapper:
    
    ''' 
    This class serves as a wrapper for compatibility with 
    tree ensemble models stored as a list of dictionaries.
    (see example in https://shap.readthedocs.io/en/stable/example_notebooks/tabular_examples/tree_based_models/Example%20of%20loading%20a%20custom%20tree%20model%20into%20SHAP.html)
    (link as of shap version 0.41)
    It is designed to be compatible with RandomForestClassifier,
    RandomForestRegressor, and RandomSurvivalForest.
    ''' 
    
    class Estimator:
        
        class Tree_:
            def __init__(self, feature, n_node_samples, children_left, 
                         children_right, threshold, value, feature_names_in_):
                self.feature = feature
                self.n_node_samples = n_node_samples
                self.children_left = children_left
                self.children_right = children_right
                self.threshold = threshold
                self.value = value
                self.feature_names_in_ = feature_names_in_
                
            
            @property
            def n_outputs(self):
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
                ) 
            self.n_outputs_ = self.tree_.n_outputs # Inferred from underlying tree_. 
            

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
        self.n_features_in_ = clf_dict['trees'][0]['n_features_in_']
        self.feature_names_in_ = clf_dict['trees'][0]['feature_names_in_']

        
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



def DT_to_dict(clf_obj, idx, output_format, T_to_bin=2):

    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sksurv.tree import SurvivalTree
    
    ''' Compatible with single output trees only, at the moment.
        compatible with SurvivalTree learners of a RandomSurvivalForest
        (scikit-survival 0.21)
        with DecisionTreeClassifier and DecisionTreeRegressor from RandomForest-s
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
        "n_features_in_": getattr(tree_obj, "n_features_in_", None),
        "feature_names_in_": getattr(clf_obj, "feature_names_in_", None),
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
        
        if T_to_bin == None and tree_dict['unique_times_'] is not None: # select median time to (any) event
            T_to_bin = tree_dict['unique_times_'][len(tree_dict['unique_times_'])//2]
    
        if output_format == "predict_survival_curve":
            tree_dict["values"] = tree.value[:,:, 1]
    
        elif output_format in ["hazard", "auto"]:
            #sum CHF over unique_times_ that have an event (imitating SurvivalTree .predict)
            tree_dict["values"] = np.sum(tree.value[:,tree_dict['is_event_time_'], 0], 
                                          axis=1).reshape(-1, 1) 
                
        elif output_format in ["survival", 'time-to-event']:
    
            tree_dict["values"] = np.trapz(tree.value[:,:, 1], 
                                           tree_obj.unique_times_,
                                           axis=1).reshape(-1, 1) # integrate S(t) over unique_times_
            # output shape: (n_nodes, 1) with E[S(t)] at each node
    
        elif output_format == "probability":
            # pick last "False" index before "True" appears
            index_T = np.argmax(tree_obj.unique_times_ > T_to_bin)-1
            # it DOES work when all times are > T_bin, as it will again select -1
            if min(tree_obj.unique_times_) > T_to_bin:
                index_T = 0
            # probability of experiencing the event by t=2 -> P(t) = 1 - S(t)
            tree_dict["values"] = 1 - tree.value[:,index_T, 1].reshape(-1, 1)
            # Why was the reshape needed? Why single-element arrays like this? Hmm...
            
        else:
            raise ValueError('Input not recognized. Double check')
            
    # Decision Tree Classifier case here:
    elif isinstance(tree_obj, DecisionTreeClassifier) and output_format in ["probability", "auto"]:
        partials = tree.value[:,0,:] # output for 2 classes, now take average
        tree_dict["values"] = (partials[:,1] / (partials[:,0] + partials[:,1])).reshape(-1, 1)
    
    elif isinstance(tree_obj, DecisionTreeRegressor) and output_format in ["probability", "auto"]:
        tree_dict["values"] = tree.value[:,0,0].reshape(-1, 1) # output (n_nodes, 1)
        
    else:
        raise ValueError("Combination of learner \'{}\' and scenario \'{}\' not recognized".format(tree_obj.__class__.__name__, output_format))
            
    tree_dict["base_offset"] = tree_dict['values'][0] # root node (0) prediction
    
    return tree_dict


def tree_list_to_model(tree_list):
    
    ''' given each learner is stored as a dict, create a list of 
    such dictionaries.
    NOTE that this is NOT directly compatible with the SHAP library, 
    which assumes that the input custom tree models are additive 
    
    To use SHAP, the user needs to divide the predicted values accordingly.
    See commmented lines below '''
    
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


def frmt_preds_to_print(y_pred, format_string='{:.4f}') -> str:
    
    # if 2-d, the only acceptable option is that it is a nested 1-d vector
    if isinstance(y_pred, np.ndarray) and y_pred.ndim == 2:
        if y_pred.shape[1] != 1:
            raise ValueError('Output vector must be 1d, or a nested 1d vector in 2d')
        y_pred = y_pred.ravel()
    
    # whether it is real 1-d vector (n,), or a float in disguise
    if isinstance(y_pred, np.ndarray):
        y_pred_str = ", ".join(format_string.format(val) for val in y_pred)
        
    elif isinstance(y_pred, float):
        y_pred_str = format_string.format(y_pred)
        
    else:
        raise ValueError('Format for y_pred not recognized')
    
    return y_pred_str

def return_partial_preds(clf_i):
    
    if isinstance(clf_i, sklearn.tree._classes.DecisionTreeClassifier):
        partials = clf_i.tree_.value[:,0,:] # now take average
        partial_preds = partials[:,1] / (partials[:,0] + partials[:,1])
 
    elif isinstance(clf_i, sklearn.tree._classes.DecisionTreeRegressor):
        partial_preds = clf_i.tree_.value.ravel()
            
    elif isinstance(clf_i, SurvivalTree):
        # clf_i.tree_.value: np array of [node, time, [H(node), S(node)]]
        #                                              ^idx 0   ^idx 1
        # OLD version currently: 'integral' of the CHF (without taking into account time spacing)
        # partial_preds = np.sum(clf_i.tree_.value[:,:, 0], axis=1)
        # NOW imitating .predict of the SurvivalTree even better:
        partial_preds = np.sum(clf_i.tree_.value[:,clf_i.is_event_time_, 0], axis=1)
            

    elif isinstance(clf_i, EnsembleWrapper.Estimator):
         partial_preds =  clf_i.tree_.value.ravel() # .ravel seems to be the needed formatting  
                            
    else:
        raise ValueError('Tree learner not recognized, or not implemented')
    
    # print('Partial preds of shape:', partial_preds.shape)
    return partial_preds


def used_feature_set(clf_i, feature_names, sample):
    
    unique_features = set()

    # tested for RandomForestClassifier and EnsembleWrapper (binary set-up)

    node_indicator_csr = clf_i.decision_path(sample.values) #sparse matrix (1, n_nodes)
    feature_idx_per_node = clf_i.tree_.feature # array (n_nodes, )

    node_index = node_indicator_csr.indices[
        node_indicator_csr.indptr[0] : node_indicator_csr.indptr[1]
    ] # csr matrix fomratted in this way
    
    for node_id in node_index[:-1]: #internal nodes (exclude leaf)
        feature_node_id = feature_idx_per_node[node_id]
        unique_features.add(feature_names[feature_node_id]) # add element to set if not in there yet
        
    return list(unique_features)
    

def rule_print_inline(clf_i, sample, weight=None, max_features_print=12):
    
    ''' sample is a pd.Series or a single-row pd.DataFrame?? '''
    ## consider treating it as a numpy array
    if isinstance(sample, np.ndarray):
        sample = pd.DataFrame(sample)
    
    # node_indicator = clf_i.decision_path(sample)
    node_indicator_csr = clf_i.decision_path(sample.values)    
    node_weights = clf_i.tree_.n_node_samples/(clf_i.tree_.n_node_samples[0])
    children_left = clf_i.tree_.children_left
    children_right = clf_i.tree_.children_right
    feature = clf_i.tree_.feature
    threshold = clf_i.tree_.threshold

    is_traversed_node = node_indicator_csr.indices[
        node_indicator_csr.indptr[0] : node_indicator_csr.indptr[1]
    ] # csr matrix fomratted in this way
    
    
    unique_features = used_feature_set(clf_i, sample.columns, sample)

    # Print only the relevant features with :.2f
    # take care of selection of sample columns so that it stays as pd.DataFrame:
    unique_features_formatted = sample.loc[:, unique_features].applymap(lambda x: '{:.2f}'.format(x))

    if len(unique_features) <= max_features_print:
        print('#'*20, '   SAMPLE   ', '#'*20)
        print(unique_features_formatted.to_string(col_space=4))
        print('#'*54)
    else:
        print('Too many features are used in the extracted rules, therefore we', end='')
        print('skip the printing. \n Increase the max_features_print parameter in case')
    
    
    partial_preds = return_partial_preds(clf_i)
    
    if weight == None or weight == 1:
        print('Baseline prediction: {:.4f}'.format(partial_preds[0]))
    else:
        print('Baseline prediction: {:.4f} \t (weight = {:.2f})'.format(partial_preds[0], weight))

    for node_id in is_traversed_node[:-1]: #internal nodes (exclude leaf)
        # continue to the next node if it is a leaf node
    
        # check if value of the split feature for sample 0 is below threshold
        if sample.values[0, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
            next_child = children_left[node_id]
        else:
            threshold_sign = "> "
            next_child = children_right[node_id]
            
        print(
            "node {node:3}: w: {weight:1.3f} "
            "{feature:8} {inequality} {threshold:6.2f}" 
            " ({feature:8} = {value:6.2f})" 
            "  -->  {partial:5.4f}".format(
                node=node_id,
                weight=node_weights[node_id],
                feature=sample.columns[feature[node_id]],
                value=sample.values[0, feature[node_id]],
                inequality=threshold_sign,
                threshold=threshold[node_id],
                partial=partial_preds[next_child]
                )
            )

    print(
        "leaf {node:3}: predicts: {predict:.4f}".format(
            node=is_traversed_node[-1],
            predict=partial_preds[is_traversed_node[-1]]
        )
    )


def rule_to_file(clf_i, sample, feature_names, rule_weight,
                 max_features_print, f):
    
    
    if isinstance(clf_i, sklearn.tree._classes.DecisionTreeClassifier):
        leaf_print = clf_i.predict_proba(sample.values)[:,1].ravel()
    elif isinstance(clf_i, sklearn.tree._classes.DecisionTreeRegressor):
        leaf_print = clf_i.predict(sample.values).ravel() # risk score
    elif isinstance(clf_i, sksurv.tree.SurvivalTree) or isinstance(clf_i, EnsembleWrapper.Estimator):
        # leaf_surv_curve = clf_i.predict_survival_function([sample.values], return_array=False)
        leaf_print = clf_i.predict(sample.values).ravel() # risk score
    else:
        raise ValueError('Input not recognized:', clf_i)


    partial_preds = return_partial_preds(clf_i)
    
    def recurse_print(node, depth, tree_, sample, feature_names, is_traversed_node, f):
        indent = "  " * depth
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED: #if feature is not undefined != -2
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            
            if is_traversed_node[node] == 1 and sample[tree_.feature[node]] <= threshold:
                is_traversed_node[node] = 0 #otherwise it will keep printing this rulesplit
                child_node = tree_.children_left[node]
                f.write("node.{:4}: {}  {} <= {}  --> {}\n".format(node, indent, name, threshold,
                                            frmt_preds_to_print(partial_preds[child_node], '{:.4f}')))
                
                recurse_print(child_node, depth + 1, tree_, sample, 
                              feature_names, is_traversed_node, f)
            
            if is_traversed_node[node] == 1 and sample[tree_.feature[node]] > threshold:
                is_traversed_node[node] = 0
                child_node = tree_.children_right[node]
                f.write("node.{:4}: {}  {} > {}  --> {}\n".format(node, indent, name, threshold,
                                            frmt_preds_to_print(partial_preds[child_node], '{:.4f}')))

                recurse_print(child_node, depth + 1, tree_, sample, 
                              feature_names, is_traversed_node, f)
                
        else: #if feature split is undefined (index == -2), then we are in a leaf
            if is_traversed_node[node] == 1:
                f.write("leaf.{:4}: {}returns {}\n".format(node, indent,
                                                           frmt_preds_to_print(partial_preds[node], '{:.4f}')))
        
    tree_structure = clf_i.tree_
    unique_features = used_feature_set(clf_i, feature_names, sample)

    # Take care of selection of sample columns so that 
    # it stays in the pd.DataFrame format. This works:
    unique_features_formatted = sample.loc[:, unique_features].applymap(lambda x: '{:.2f}'.format(x))
        
    if len(unique_features) <= max_features_print:
        f.write('#'*24 + '  SAMPLE  ' + '#'*24 + '\n')
        f.write(unique_features_formatted.to_string(col_space=4)+ '\n')
        f.write('#'*18 + '   RULE WEIGHT: {:.2f}  '.format(rule_weight) + '#'*18 + '\n')
        f.write('Baseline prediction: {}\n'.format(frmt_preds_to_print(partial_preds[0], '{:.4f}')))


    is_traversed_node = clf_i.decision_path(sample.values).toarray()[0]
    sample = sample.to_numpy().reshape(-1) #from single column to single line
    ## and here most of the printing is done (recursive calls)
    recurse_print(0, 0, tree_structure, sample, feature_names, is_traversed_node, f) #feature_name list missing?



def rule_to_code(clf_i, traversed_nodes, sample, feature_names, full_save_name):
    
    if isinstance(clf_i, sklearn.tree._classes.DecisionTreeClassifier):
        leaf_print = clf_i.predict_proba([sample.values])[:,1].ravel()
    elif isinstance(clf_i, sklearn.tree._classes.DecisionTreeRegressor):
        leaf_print = clf_i.predict([sample.values]).ravel() # risk score
    elif isinstance(clf_i, sksurv.tree.SurvivalTree) or isinstance(clf_i, sksurv.tree.SurvivalTree):
        # leaf_surv_curve = clf_i.predict_survival_function([sample.values], return_array=False)
        leaf_print = clf_i.predict(sample.values).ravel() # risk score

    tree_ = clf_i.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature]
        
    intervals = {feat : [-np.inf, np.inf] for feat in feature_names}
    
    if full_save_name is not None:
        with open(full_save_name, 'w+') as f:
            f.write("###### SAMPLE to explain ######\n")
            
            for i,k in zip(feature_names, range(len(feature_names))):
                f.write("{:13}: {:7} \n".format(str(i), str(sample[k])))
                
            f.write("\n###############################\n")
    
            sample = sample.to_numpy().reshape(-1) #from single column to single line
        
            def recurse(node, depth, sample, intervals):
                indent = "  " * depth
                if tree_.feature[node] != _tree.TREE_UNDEFINED: #if feature is not undefined (??)
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    if traversed_nodes[node] == 1 and sample[tree_.feature[node]] <= threshold:
                        intervals[name][1] = threshold # reduce feature upper bound
                        traversed_nodes[node] = 0
                        f.write("node.{}: {}if {} <= {}\n".format(node, indent, name, threshold))
                        
                    recurse(tree_.children_left[node], depth + 1, sample, intervals)
                    
                    if traversed_nodes[node] == 1 and sample[tree_.feature[node]] > threshold:
                        intervals[name][0] = threshold # increase feature lower bound 
                        traversed_nodes[node] = 0
                        #print("node.{}: {}if {} > {}".format(node, indent, name, threshold))
                        f.write("node {}: {}if {} > {}\n".format(node, indent, name, threshold))
                    recurse(tree_.children_right[node], depth + 1, sample, intervals)
                else: #it is undefined, it is therefore a leaf (?)
                    if traversed_nodes[node] == 1:
                        #print("leafnode.{}: {}return {}".format(node, indent, leaf_print2)) #tree_.value[node].ravel()
                        f.write("leafnode.{}: {}returns {}\n".format(node, indent, leaf_print))
                        name_save_plot = full_save_name.split(".")[0] + "-plot.png"
                        
                        f.write("predicted:{}\n".format(leaf_print))
            recurse(0, 1, sample, intervals)
            f.close()


    
def rule_to_code_and_intervals(tree, scenario, traversed_nodes, sample, feature_names, full_save_name):
    learner = tree
    
    if isinstance(learner, sklearn.tree._classes.DecisionTreeClassifier):
        leaf_print = learner.predict_proba([sample])[:,1].ravel()
    elif isinstance(learner, sklearn.tree._classes.DecisionTreeRegressor):
        leaf_print = learner.predict([sample]).ravel() # risk score
    elif isinstance(learner, sksurv.tree._classes.SurvivalTree):
        leaf_surv_curve = learner.predict_survival_function([sample], return_array=False)
        

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature]
        
    intervals = {feat : [-np.inf, np.inf] for feat in feature_names}
    
    if full_save_name is not None:
        with open(full_save_name, 'w+') as f:
            f.write("###### SAMPLE to explain ######\n")
            
            for i,k in zip(feature_names, range(len(feature_names))):
                f.write("{:13}: {:7} \n".format(str(i), str(sample[k])))
                
            f.write("\n###############################\n")
    
            sample = sample.to_numpy().reshape(-1) #from single column to single line
        
            def recurse(node, depth, sample, intervals):
                indent = "  " * depth
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    if traversed_nodes[node] == 1 and sample[tree_.feature[node]] <= threshold:
                        intervals[name][1] = threshold # reduce feature upper bound
                        traversed_nodes[node] = 0
                        f.write("node.{}: {}if {} <= {}\n".format(node, indent, name, threshold))
                        
                    recurse(tree_.children_left[node], depth + 1, sample, intervals)
                    
                    if traversed_nodes[node] == 1 and sample[tree_.feature[node]] > threshold:
                        intervals[name][0] = threshold # increase feature lower bound 
                        traversed_nodes[node] = 0
                        #print("node.{}: {}if {} > {}".format(node, indent, name, threshold))
                        f.write("node.{}: {}if {} > {}\n".format(node, indent, name, threshold))
                    recurse(tree_.children_right[node], depth + 1, sample, intervals)
                else: #it is undefined, it is therefore a leaf (?)
                    if traversed_nodes[node] == 1:
                        #print("leafnode.{}: {}return {}".format(node, indent, leaf_print2)) #tree_.value[node].ravel()
                        f.write("leafnode.{}: {}return {}\n".format(node, indent, leaf_print))
                        name_save_plot = full_save_name.split(".")[0] + "-plot.png"
                        
                        #print("predicted:{}\n".format(leaf_print))
                        f.write("predicted:{}\n".format(leaf_print))
            recurse(0, 1, sample, intervals)
            f.close()
    
    if full_save_name is not None:
        with open(full_save_name.split(".")[0] + "-simplif." + full_save_name.split(".")[-1], 'w+') as f:
            f.write("###### SAMPLE to explain ######\n")
            
            for i,k in zip(feature_names, range(len(feature_names))):
                f.write("{:10}: {:7} \n".format(str(i), str(sample[k])))
    
                
            f.write("\n###### final intervals ########\n")
            
            for item in intervals:
                if intervals[item][0] != -np.inf or intervals[item][1] != np.inf:
                    f.write("{:6} < {:8} <= {:6} \n".format(str(intervals[item][0]),
                                    str(item).center(8), str(intervals[item][1])))
    
                
            f.close()
            
            with open(full_save_name) as f: #printing tree-rule structure on console
                print(f.read())
            
            print("###############################") #separator between tree rule print, and leaf interval representation
            
            with open(full_save_name.split(".")[0] + "-simplif." + full_save_name.split(".")[-1]) as f:
                print(f.read()) #printing (simplified) leaf structure on console
            

def get_data_list(set_up, root_folder, filter_out=True):
        
    if set_up.lower() in ["multi", "multi-label", "multi-l", "mtc"]:
        datapath = os.path.join(root_folder, "datasets", "multi_label")
        scenario_subpath = "Multi_label"
    elif set_up.lower() in ["survival", "surv"]:
        datapath = os.path.join(root_folder, "datasets", "survival")
        scenario_subpath = "Survival"

    elif set_up.lower() in ["binary", "bin"]:
        datapath = os.path.join(root_folder, "datasets", "binary")
        scenario_subpath = "Binary"
        
    elif set_up.lower() in ["regression", "regress", "regr"]:
        datapath = os.path.join(root_folder, "datasets", "regression")
        scenario_subpath = "Regression"
    
    elif set_up.lower() in ["multi-target", "multi-t", "mtr", "mt_regress"]:
        datapath = os.path.join(root_folder, "datasets", "mt_regression")
        scenario_subpath = "MTR"
        
    dnames = os.listdir(datapath)
    for dname in dnames:
        new_dir_name = os.path.join(datapath, dname.split(".csv")[0])
        if not os.path.exists(new_dir_name):
            os.makedirs(new_dir_name)
            print("created directory:\n", os.path.join(datapath, dname.split(".")[:-1][0]))
    
    #get folders only
    dnames = [f for f in os.listdir(datapath) if not os.path.isfile(os.path.join(datapath, f))]

    print("dnames:", dnames)
    return dnames, datapath, scenario_subpath


def output_X_y(df, set_up):
    
    if set_up in ["bin", "binary"] + ["regress", "regression"]:   
        y_cols = df.columns[-1:] #as list of single element

    elif set_up in ["surv", "survival"]:        

        df[df.columns[-2]] = df[df.columns[-2]].astype('bool') # needed for correct recarray
        y_cols = df.columns[-2:]


    elif set_up in ["mtr", "multi-target"]:        
        y_cols = [col for col in df.columns if "target_" in col]
    
    elif set_up in ["mtc", "multi-label"]:         
        y_cols = [col for col in df.columns if "label" in col or "tag_" in col]
    else:
        raise ValueError('Setup {} not recognized'.format(set_up))
    
    y = df[y_cols]
    X = df[df.columns[~df.columns.isin(y_cols)]]

    return X, y    
    
    
    

def preparing_data(set_up, datapath, folder, n_folds=5, stratify=True):
    
    if set_up.lower() in ["bin", "binary"]:
        df1 = pd.read_csv(datapath + folder + "/train1.csv") 
        df2 = pd.read_csv(datapath + folder + "/valid1.csv")
        df3 = pd.read_csv(datapath + folder + "/test1.csv")
        data = pd.concat([df1, df2, df3], ignore_index=True)
        del df1, df2, df3
        X = data.drop(data.columns[-1], axis=1, inplace=False)
        X  = X.astype('float64') #apparently needed
        y = data.iloc[:,-1].ravel() #last column for binary data
        y_strat = y

        
    elif set_up.lower() in ["multi", "multi-label", "multi-l"]:
        stratify = False # not possible (actuyally possible but not worth it)
        df1 = pd.read_csv(datapath + folder +  "/" + str(folder) + "_train_fold_1.csv") 
        df2 = pd.read_csv(datapath + folder +  "/" + str(folder) + "_test_fold_1.csv") 
        data = pd.concat([df1, df2], ignore_index=True)
        del df1, df2
        
        label_cols = [col for col in data.columns if "label" in col]
        X = data.drop(columns=label_cols, axis=1, inplace=False)
        X  = X.astype('float64') #apparently needed
        y = pd.DataFrame(data, columns=label_cols)
        y_strat = None # and stratify is set to False
        
    elif set_up.lower() in ["survival", "surv"]:  #last two columns for event
        
        try:
            df1 = pd.read_csv(datapath + folder.split(".csv")[0] +  "/" + "train1.csv") 
            df2 = pd.read_csv(datapath + folder.split(".csv")[0] +  "/" + "test1.csv") 
            df3 = pd.read_csv(datapath + folder.split(".csv")[0] +  "/" + "val1.csv") 
            df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
        except:
            df1 = pd.read_csv(datapath + folder.split(".csv")[0] +  "/" + "new_train1.csv") 
            df2 = pd.read_csv(datapath + folder.split(".csv")[0] +  "/" + "new_test1.csv") 
            df = pd.concat([df1, df2], axis=0, ignore_index=True)
        #old version (until December 8th)
        #df = pd.read_csv(datapath + folder)
        df[df.columns[-2]] = df[df.columns[-2]].astype('bool') # needed for correct recarray
        X = df[df.columns[:-2]].astype('float64') #astype apparently needed
        
        y = df[df.columns[-2:]]#.to_numpy()
        y = y.to_records(index=False) #builds the structured array, needed for RSF
        y_strat = np.array([i[0] for i in y])
        
        
    elif set_up.lower() in ["regression", "regress", "regr"]:  #last two columns for event
        stratify = False # not possible
        
        df1 = pd.read_csv(datapath + folder + "/old_train1.csv") # <- alirght?
        df2 = pd.read_csv(datapath + folder + "/old_test1.csv")
        data = pd.concat([df1, df2], ignore_index=True)
        del df1, df2
        X = data.drop(data.columns[-1], axis=1, inplace=False)
        #X  = X.astype('float64') #apparently needed        
        y = data.iloc[:,-1].ravel() #last column is regression target
        y = (y - y.min())/(y.max() - y.min()) # scale to [0-1]        
        y_strat = None
        
    elif set_up.lower() in ["multi-target", "multi-t", "mtr"]:
        stratify = False
        
        data_path = os.path.join(datapath, folder, folder +".csv")
        
        data = pd.read_csv(data_path)
        label_cols = [col for col in data.columns if "target_" in col]
        X = data.drop(columns=label_cols, axis=1, inplace=False)
        X  = X.astype('float64') #apparently needed
        print("X size:", X.shape)
        y = pd.DataFrame(data, columns=label_cols)
        print("y size:", y.shape)
        y = (y-y.min())/(y.max()-y.min()) # min-max normalization (per target)
        y_strat = None # and stratify is set to False
    
    else:
        raise ValueError("scenario key not recognised: {}".format(set_up))
        
    if stratify == True:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        inner_folds = kf.split(X, y_strat)
        
    elif stratify == False:      
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        inner_folds = kf.split(X, y)
        
    assert X.isnull().sum().sum() == 0 # cannot handle missing vlaues for now

    return X, y, stratify, inner_folds


from sklearn.metrics import roc_auc_score as auroc
from sksurv.metrics import concordance_index_censored as c_index
from sklearn.metrics import mean_absolute_error as mae

#add brier score (custom version)
def score_method(y_test, y_pred, set_up): #add more methods
    
    if set_up.lower() in ["surv", "survival"]:
        #y_pred = np.array(y_pred).ravel() # list to 2D array to 1D array
        return c_index([i[0] for i in y_test], [i[1] for i in y_test],
                   y_pred)[0]
    elif set_up.lower() in ["multi", "multi-label", "multi-l", "mtc"]:
        return auroc(y_test, y_pred, average="weighted")
    
    elif set_up.lower() in ["regression", "regress", "regr"]:
        return mae(y_test, y_pred)
        #return mse(y_test, y_pred)
    
    elif set_up.lower() in ["bin", "binary"]:
        return auroc(y_test, y_pred, average="weighted")
    elif set_up.lower() in ["multi-t", "mtr", " mt_regress", "multi-target"]:
        return mae(y_test, y_pred, multioutput="uniform_average")
    else:
        raise KeyError("set-up: \"{}\" not recognised".format(set_up))


def plot_my_tree(my_clf, tree_index, feature_names, sample_index, name): # add "data" variale in the future
    #depth = [estimator.tree_.max_depth for estimator in alt_clf.estimators_][tree_index]
    my_depth = my_clf.estimators_[tree_index].tree_.max_depth
    smart_size_1 = 1 + 1.3*np.sqrt(my_clf.estimators_[tree_index].tree_.n_leaves**1.5)
    smart_size_2 = my_depth*1.4
    
    fig, ax = plt.subplots(figsize=(smart_size_1, smart_size_2))
    tree.plot_tree(my_clf[tree_index], feature_names=feature_names, fontsize=8)

    #predict_sample = my_clf[tree_index].predict([the_data.iloc[sample_index]])
    plt.title("Candidate %i predicting sample %i" % (tree_index, sample_index))
    #filename = "Tree{}-sample{}.png".format(tree_index, sample_index)
    #filename = pathlib.PurePath('name')
    plt.savefig(name)
    #plt.show()
    

def plot_no_clustering(plot_data_bunch): # extra info, e.g. for the Figure
    
    PCA_fitted = PCA(n_components=2).fit(plot_data_bunch.proj_data)
    plottable_data = PCA_fitted.transform(plot_data_bunch.proj_data)  # (lambda,2)
    
    fig, ax = plt.subplots(figsize=(4.2, 4.5))
    fig.tight_layout()  # consider this
    fig.subplots_adjust(top=0.86)
    ax.scatter(plottable_data[:,0], plottable_data[:,1], c="black")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.axis('equal') # is it even a good idea? We will see
    ax.set_title('Path projection')
    
    plt.show()
    
    return


def custom_axes_limit(bunch_min_value, bunch_max_value, RF_pred, is_binary):
    
    # works as expected also when RF_pred is np.nan
    v_min = min(bunch_min_value, RF_pred)
    v_max = max(bunch_max_value, RF_pred)
    
    if is_binary:
        # combat counterintuitive colouring when predictions are confident
        v_min = min(v_min, 0.8) # v_min never above 0.8
        v_max = max(v_max, 0.2) # v_max never below 0.2

    # add a bit of extra spacing on the extremes
    v_min = v_min-(v_max-v_min)*0.05
    v_max = v_max+(v_max-v_min)*0.05+0.002 # avoid the case v_min = v_max
    
    
    return v_min, v_max
    

    ## LocalMethod inputs: plot_data_bunch, plot_kmeans, tuned_method, self.clf.n_outputs_
def plot_preselected_trees(plot_data_bunch, kmeans, tuned_method, final_ts_idx,
                           base_font_size=12, show_ax_ticks="auto",
                           plot_dpi=120):
    
    small_size = 40
    big_size = 220
    
    # Custom formatter function for colorabar on ax4
    # not working correctly..
    def custom_formatter(x, pos): # pos paramter to comply with expected signature
        if np.abs(x) < 1e-7: # 0.00 for near zero values
            return f"{x:.2f}"
        if 1e-2 <= np.abs(x) < 1:
            return f"{x:.2f}"  # 2 decimal digits for numbers between -1 and 1
        elif 1 <= np.abs(x) < 10:
            return f"{x:.1f}"  # 1 decimal digit 
        elif 10 <= np.abs(x) < 100:
            return f"{x:.0f}"  # 0 decimal digits (round to nearest integer)
        else: # 1e-7 < np.abs(x) < 1e-2 or  np.abs(x) > 100
            return f"{x:.1e}"  # Scientific notation with 2 significant digits
    
    if show_ax_ticks == "auto":
        show_ax_ticks = False if base_font_size > 15 else True
    
    
    #PCA to 2 dimensions for projected trees
    #(original proj dimension can be > 2)
    PCA_fitted = PCA(n_components=2).fit(plot_data_bunch.proj_data)
    plottable_data = PCA_fitted.transform(plot_data_bunch.proj_data)  # (lambda,2)

    centers = PCA_fitted.transform(kmeans.cluster_centers_)
    class_memb = kmeans.labels_
    
    custom_gridspec = {'width_ratios': [3, 0.2, 3, 0.2]}

    
    fig, (ax1, ax2, ax3, ax4) = pylab.subplots(1, 4, figsize=(10, 4.5), dpi=plot_dpi,
                                    gridspec_kw=custom_gridspec)
    # (scatter1, cb1, scatter2, cb2)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    
    # conditional sizes for trees and candidate trees:
    is_final_candidate = [plot_data_bunch.index[i] in final_ts_idx
                          for i in range(len(plot_data_bunch.index))]
    
    
    #####   LEFT PLOT (cluster memberships)   #####
    
    for i, txt in enumerate(centers): #plot cluster centers
        ax1.annotate(i+1, (centers[i,0], centers[i,1]),
                      bbox={"boxstyle" : "circle", "color": "grey", "alpha": 0.6})
        

    x_normal = plottable_data[:,0][[not x for x in is_final_candidate]]
    y_normal = plottable_data[:,1][[not x for x in is_final_candidate]]
    color_normal = class_memb[[not x for x in is_final_candidate]]
    
    x_selected = plottable_data[:,0][is_final_candidate]
    y_selected = plottable_data[:,1][is_final_candidate]
    color_selected = class_memb[is_final_candidate]
    
    ax1.scatter(x_normal, y_normal,
               c=color_normal,
               cmap=None,
               s=small_size,
               marker="o",
               edgecolors=(1, 1, 1, 0.5))
    
    ax1.scatter(x_selected, y_selected,
               c=color_selected,
               cmap=None,
               s=big_size,
               marker="*",
               edgecolors="black")

    ax1.set_xlabel("PC1", fontdict={'fontsize': base_font_size})
    ax1.set_ylabel("PC2", fontdict={'fontsize': base_font_size})

    ax1.axis('equal') # is it even a good idea? We will see
    ax1.set_title('Cluster membership', fontdict={'fontsize': base_font_size})
    
    # create the map for segmented colorbar (ax2: left colorbar)
    cmap = plt.cm.viridis  # default colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)        
    
    # define the bins and normalize
    freqs = np.bincount(class_memb)
    if np.min(freqs) == 0:
         raise KeyError("There are empty clusters, the scatter and colorbar would differ in color shade")
    norm_bins = list(np.cumsum(freqs))
    norm_bins.insert(0, 0)
    
    if len(norm_bins) == 2:  #color gradient is off, add artificial bin
        # this will create an empty artificial cluster later on, that will be dropped
        norm_bins.insert(-1, norm_bins[1]) 
    
    # scatterplot color does not scale correctly if there are empty classes!
    # transform list to array (ticks location needs arithmentic computation)
    norm_bins = np.array(norm_bins)
    
    # create label names
    labels = []
    for i in np.unique(class_memb):
        labels.append("cl.{:d}".format(i+1))

    # normalizing color, prepare ticks, labels
    norm = mpl.colors.BoundaryNorm(norm_bins, cmap.N)
    tickz = norm_bins[:-1] + (norm_bins[1:] - norm_bins[:-1]) / 2
    
    if tickz.max() == norm_bins.max(): #artificial empty cluster somewhere: drop
        tickz = tickz[:-1] # drop last tick at top of colorbar
    
    # colorab on axis 2 out of 4.
    cb = mpl.colorbar.Colorbar(ax2, cmap=cmap, norm=norm,
        spacing='proportional', ticks=tickz, boundaries=norm_bins,
        format='%1i')
        #label="cluster membership")
    cb.ax.set_yticklabels(labels)  # vertically oriented colorbar
    cb.ax.tick_params(labelsize=base_font_size-1) #ticks font size
    ax2.yaxis.set_ticks_position('left')
    

    #####   RIGHT PLOT (predictions or losses)  #####
    
    # PREDICTIONS when single class output (SurvivalTree included)
    if tuned_method.clf.n_outputs_ == 1 or isinstance(tuned_method.clf,
                                        RandomSurvivalForest): # single output, color on predictions
    
    
        ### right figure scatterplot here (ax3 and ax4):
                
        cmap2 = plt.cm.get_cmap('RdYlBu') # or "viridis", or user choice
       
        is_binary = isinstance(tuned_method.clf, sklearn.ensemble.RandomForestClassifier)
        
        v_min, v_max = custom_axes_limit(np.array(plot_data_bunch.pred).min(),
                                         np.array(plot_data_bunch.pred).max(),
                                         plot_data_bunch.RF_pred, is_binary)
        
        norm_preds = mpl.colors.BoundaryNorm(np.linspace(v_min, v_max, 256),
                                             cmap2.N)
        
        color_indeces = np.zeros(len(plot_data_bunch.pred)) #length = n_trees
        
        for i in range(len(plot_data_bunch.pred)):
            color_indeces[i] = np.argmin([thresh <= plot_data_bunch.pred[i] 
                                           for thresh in norm_preds.boundaries])
        # format as integers, for list comprehension
        color_indeces = [int(x+0.1) for x in color_indeces] 
        
        real_colors = np.array([cmap2(idx) for idx in color_indeces])
        
        ax3.scatter(x_normal, y_normal,
                   c=real_colors[[not x for x in is_final_candidate]],
                   #cmap=cmap2,
                   s=small_size,
                   marker="o",
                   edgecolors=(1,1,1,0.5))
        
        ax3.scatter(x_selected, y_selected,
                   c=real_colors[is_final_candidate],
                   #cmap=cmap2,
                   s=big_size,
                   marker="*",
                   edgecolors="black")
        
        ax3.set_xlabel("PC1", fontdict={'fontsize': base_font_size})
        ax3.yaxis.set_label_position("right")
        ax3.set_ylabel("PC2", fontdict={'fontsize': base_font_size})
        #ax3.yaxis.tick_right()
        ax3.axis('equal') # is it even a good idea? We will see
        
        # add color bar to the side
        pred_tick = np.round(float(tuned_method.local_prediction()), 3)
        
        cb2 = mpl.colorbar.Colorbar(ax4, cmap=cmap2, norm=norm_preds,
                                    format=FuncFormatter(custom_formatter),
                                    label="predicted: " + str(pred_tick))
        ax3.set_title('Rule-path predictions', fontdict={'fontsize': base_font_size})
        
        
        #if isinstance(tuned_method.clf, RandomSurvivalForest)
        
        ## add to colorbar a line corresponding to LTreeX prediction
        cb2.ax.plot([0, 1], [plot_data_bunch.pred]*2, color='grey',
                    linewidth=1)
        cb2.ax.plot([0.02, 0.98], [pred_tick]*2, color='black', linewidth=2.5,
                    marker="P")
                
        
        if isinstance(tuned_method.clf, RandomSurvivalForest):
            cb2.set_label("Cumul.Hazard: "+ str(pred_tick),
                          size=base_font_size-3)

        if isinstance(tuned_method.clf, sklearn.ensemble.RandomForestClassifier):
            cb2.set_label("pred. prob:"+ str(pred_tick),
                          size=base_font_size-3)
        if isinstance(tuned_method.clf, sklearn.ensemble.RandomForestRegressor):
            cb2.set_label("pred. value:"+ str(pred_tick),
                          size=base_font_size-3)
        
    # LOSS  whne multi-output predictions: plot distance from RF preds
    else: 
    
        color_map = plt.cm.get_cmap('RdYlBu')  # or "viridis", or user choice
        #norm = BoundaryNorm(np.linspace(0.2, 0.8, 256), color_map.N)
        # normalise colors min pred.--> blue, max pred. --> red to improve readability
        
        # v_min, v_max = custom_axes_limit(np.array(plot_data_bunch.loss).min(),
        #                                  np.array(plot_data_bunch.loss).max(),
        #                                  RF_pred=np.nan, is_binary=False)

        v_min, v_max = np.array(plot_data_bunch.loss).min(), np.array(plot_data_bunch.loss).max()
        
        norm_preds = BoundaryNorm(np.linspace(v_min, v_max, 256), cmap.N)
        
        final_candidate_loss = np.array(plot_data_bunch.loss)[is_final_candidate]
        normal_rule_loss = np.array(plot_data_bunch.loss)[[not x for x in is_final_candidate]]

        
        ax3.scatter(x_normal, y_normal,
                   c=normal_rule_loss,
                   cmap=color_map,
                   s=small_size,
                   marker="o",
                   edgecolors=(1,1,1,0.5))
        
        ax3.scatter(x_selected, y_selected,
                   c=final_candidate_loss,
                   cmap=color_map,
                   s=big_size,
                   marker="*",
                   edgecolors="black")
        
        cb2 = mpl.colorbar.Colorbar(ax4, cmap=color_map, norm=norm_preds,
                                    label=str(tuned_method.fidelity_measure)+' loss')
        cb2.ax.plot([0, 1], [plot_data_bunch.loss]*2, color='grey', linewidth=1)

    # end indentation single-target vs multi-target case
    
    ticks_to_plot = ax4.get_yticks()
    
    if np.abs(np.min(ticks_to_plot)) < 1e-3 and np.abs(np.max(ticks_to_plot)) > 1e-2:
        min_index = np.argmin(ticks_to_plot)
        ticks_to_plot[min_index] = 0
        ax4.set_yticks(ticks_to_plot)

    ax4.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax4.minorticks_off()
    
    cb2.ax.tick_params(labelsize=base_font_size-3) #ticks font size
    
    
    if show_ax_ticks == False:
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])    

    # plt.show()

    return #plottable_data, labels


def format_targets(y_train, y_test, SETUP, verbose=0):
    
    BINARY_KEYS = ["bin", "binary"]
    SURVIVAL_KEYS = ["surv", "survival"]
    REGRESSION_KEYS = ["regress", "regression", "regr"]
    MTC_KEYS = ["multi-l", "multi-label", "mtc", "multi"]
    MTR_KEYS = ["multi-t", "multi-target", "mtr"]
    
    # This function sets target variable to the correct format depending 
    # on the prediciton scenarios.E.g. sets np.recarray for survival data,
    # or normalises data for single and multi-target regression tasks.

    if SETUP.lower() in BINARY_KEYS+ REGRESSION_KEYS:
        y_train = y_train.values
        y_test = y_test.values
    
    
    if SETUP.lower() in MTC_KEYS + MTR_KEYS:
        if verbose >= 1:
            print("orig. n* labels:", y_test.shape[1])
    
    # drop targets that are not present in the train or not present in the test
    # otherwise measures such as AUROC and AUPR collapse
    if SETUP.lower() in MTC_KEYS + MTR_KEYS:
        y_train.columns = y_test.columns # otherwise, sometimes is a bug
        for col in y_test.columns:
            if len(y_test[col].unique()) == 1 or len(y_train[col].unique()) == 1:
                y_test = y_test.drop(col, inplace=False, axis=1)
                y_train = y_train.drop(col, inplace=False, axis=1)
        
        n_labels = y_test.shape[1]
        if verbose >= 0:
            print("new n* labels:", n_labels)
        
    if SETUP.lower() in BINARY_KEYS+ REGRESSION_KEYS:
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        
    if  SETUP.lower() in SURVIVAL_KEYS:
        y_train = y_train.to_records(index=False)
        y_test = y_test.to_records(index=False)
    # formatting y target complete
    
    return y_train, y_test


def format_RF_preds(rf, X_test, SETUP):
    
    BINARY_KEYS = ["bin", "binary"]
    SURVIVAL_KEYS = ["surv", "survival"]
    REGRESSION_KEYS = ["regress", "regression", "regr"]
    MTC_KEYS = ["multi-l", "multi-label", "mtc", "multi"]
    MTR_KEYS = ["multi-t", "multi-target", "mtr"]
    
    # storing predictions for performance evaluation:
    if SETUP.lower() in SURVIVAL_KEYS + REGRESSION_KEYS:
        y_ens_pred = rf.predict(X_test)
        
    elif SETUP.lower() in BINARY_KEYS: 
        y_ens_pred = rf.predict_proba(X_test)[:,1]

    elif SETUP.lower() in MTC_KEYS:
        # original output is list of length L (n*labels) of prediction (n_samples, n_classes)
        y_ens_pred = rf.predict_proba(X_test)
        y_ens_pred = np.transpose(np.array(y_ens_pred)[:,:,1])
        
    elif SETUP.lower() in MTR_KEYS:
        y_ens_pred = rf.predict(X_test)
        
    return y_ens_pred



def validate_paramater_run(p_grid, EXTRA_NOTES, N_FOLDS):
    
    assert isinstance(p_grid, dict) # validate dictionary ( and keys? not now)
    
    if EXTRA_NOTES not in ["", "noTrees_", "noDims_", "Ablat2_", "trial_"]:
        raise KeyError("Key \"{}\" for EXTRA NOTES not recognized. \n \
                Must be in [\"\", \"noTrees_\", \"noDims_\", \"Ablat2_\", \"trial_\"]".format(EXTRA_NOTES))
    
    if EXTRA_NOTES != "trial_": # K in [1,2,3] in any case
        assert p_grid["n_clusters"] == [1, 2, 3]
        assert N_FOLDS == 5

    if EXTRA_NOTES == "": # default grid, verify
        assert p_grid["n_dims"] == [2, 5, None]
        assert p_grid["n_trees"] == [0.2, 0.5, 0.8]

    
    if EXTRA_NOTES in ["noDims_", "Ablat2_"]:
        assert p_grid["n_dims"] == [None]
        
    if EXTRA_NOTES in ["noTrees_", "Ablat2_"]:
        assert p_grid["n_trees"] == [100]
    


def rename_data_index(df, SETUP): # useful in statistical_analysis_Nemenyi.py
    
    if SETUP == "regress":
        df.rename(index={"concrete_compressive_strength" : "concrete_compress",
                                  "car_imports_1985_imputed": "car imports",
                                  "students_final_math" : "students maths"
                                  }, inplace=True, errors="ignore")
    if SETUP == "bin":
        df.rename(index={"breast_cancer_diagnostic" : "B.C. diagn.",
                                  "breast_cancer_original": "B.C. original",
                                  "breast_cancer_prognostic": "B.C. progn.",
                                  "brest_cancer_coimba": "B.C. coimba",
                                  "Colonoscopy_green" : "Col. Green",
                                  "Colonoscopy_hinselmann" : "Col. Hinselm.",
                                  "Colonoscopy_schiller": "Col. Schiller",
                                  "LSVT_voice_rehabilitation" : "LSVT voice",
                                  "simulation_crashes" : "simul. crashes",
                                  "vertebral_column_data" : "vertebral"
                                  }, inplace=True, errors="ignore")   
        
    if SETUP == "surv":
        df.rename(index={"breast_cancer-survival-imputed" : "B.C. survival",
                                  "FLChain-single_event-imputed": "FLChain",
                                  "NHANES_I-imputed": "NHANES I",
                                  "primary_biliary_cirrhosis": "PBC",
                                  "rotterdam-excl-recurr" : "rotterdam (excl. recurr)",
                                  "rotterdam-incl-recurr" : "rotterdam (incl. recurr)",
                                  }, inplace=True, errors="ignore")
        
    return df

