import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA
import matplotlib as mpl
import pylab

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
# from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from sklearn import tree
import sklearn
import sksurv
from sksurv.ensemble import RandomSurvivalForest

from sklearn.tree import _tree

def rule_print_inline(tree, sample, weight=None):
    
    clf = tree
    print('#'*16, '   SAMPLE   ', '#'*16)
    node_indicator = clf.decision_path(sample.values)
    # Set to store the unique feature indices used in the rule sets
    unique_features = set()

    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    X_test = sample
    sample_id = 0

    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    ]
    
    for node_id in node_index[:-1]: #internal nodes (exclude leaf)
        unique_features.add(feature[node_id])

    # Print only the relevant features
    # print(sample.iloc[:, list(unique_features)].to_string())
    # Print only the relevant features with :.2f
    unique_features_formatted = sample.iloc[:, list(unique_features)].applymap(lambda x: '{:.2f}'.format(x))
    print(unique_features_formatted.to_string())
    print('#'*54)


    node_indicator = clf.decision_path(sample.values)
    #leaf_id = clf.apply(sample)    

    node_weights = clf.tree_.n_node_samples/(clf.tree_.n_node_samples[0])
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    X_test = sample
    sample_id = 0    
    
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    ]
    
    if isinstance(tree, sklearn.tree._classes.DecisionTreeClassifier):
        if tree.tree_.n_outputs == 1:
            partials = tree.tree_.value[:,0,:] # now take average
            partial_preds = partials[:,1] / (partials[:,0] + partials[:,1])
            
        else:
            KeyError("n_classes > 1, not implemented yet.")
 
    elif isinstance(tree, sklearn.tree._classes.DecisionTreeRegressor):
        if tree.tree_.n_outputs == 1:
            partial_preds = tree.tree_.value.ravel()
        else:
            partial_preds = tree.tree_.value[:,:,0]

            # round values, convert to string...
            # or manage to print a ndarray with .__format__
            raise ValueError("n_classes > 1, not implemented yet.")
            
    elif isinstance(tree, sksurv.tree.SurvivalTree):
        # if tree.tree_.n_outputs == 1: # implementation is different!!!
        #else:
        #    KeyError("n_classes > 1, not implemented yet.")            
        # multi-output Survival Tree does not exist yet
        
        # node label, plotting risk score
        # tree.tree_.value: np array of [node, time, [H(node), S(node)]]
        #                                              ^idx 0   ^idx 1
        # currently: 'integral' of the CHF (without taking into account time spacing)
        # coherent with the .predict method
        partial_preds = np.sum(tree.tree_.value[:,:, 0], axis=1)
        # alternative:
        # partial_preds = tree.tree_.value[:,-1, 0] # CHF at last time point
                            
    else:
        raise ValueError('Tree learner not recognized, or not implemented')
    
    if weight == None or weight == 1:
        print('Baseline prediction: {:.4f}'.format(partial_preds[0]))
    else:
        print('Baseline prediction: {:.4f} \t (weight = {:.2f})'.format(partial_preds[0], weight))

    for node_id in node_index[:-1]: #internal nodes (exclude leaf)
        # continue to the next node if it is a leaf node
        #if leaf_id[sample_id] == node_id:
        #    continue
    
        # check if value of the split feature for sample 0 is below threshold
        if X_test.iloc[sample_id, feature[node_id]] <= threshold[node_id]:
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
                #sample=sample_id,
                feature=X_test.columns[feature[node_id]],
                value=X_test.iloc[sample_id, feature[node_id]],
                inequality=threshold_sign,
                threshold=threshold[node_id],
                partial=partial_preds[next_child]
                )
            )
        
            
    X_sample = X_test.iloc[sample_id,:].values # numpy.ndarray
        
    if hasattr(clf, "predict_proba"):
        y_pred = clf.predict_proba([X_sample])[:,1]
    elif hasattr(clf, "predict"):
        y_pred = clf.predict([X_sample])
    else:
        print(clf)
        KeyError("check tree method, is it correct?")

    print(
        "\nleaf {node:3}: predicts:{predict:5.4f}".format(
            node=node_index[-1],
            predict=float(y_pred)
        )
    )



def rule_print_inline_old(tree, X_test, sample_id):
    
    clf = tree
    node_indicator = clf.decision_path(X_test)
    #leaf_id = clf.apply(X_test)
    
    #node_weights = clf.tree_.n_node_samples/(clf.tree_.n_node_samples[0])
    #n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    
    #prop_samples = clf.
    
    
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    ]
    
    if isinstance(tree, sklearn.tree._classes.DecisionTreeClassifier):
        if tree.tree_.n_outputs == 1:
            partials = tree.tree_.value[:,0,:] # now take average
            partial_preds = partials[:,1] / (partials[:,0] + partials[:,1])
            
        else:
            KeyError("n_classes > 1, not implemented yet.")
 
    if isinstance(tree, sklearn.tree._classes.DecisionTreeRegressor):
        if tree.tree_.n_outputs == 1:
            partial_preds = tree.tree_.value.ravel()
        else:
            KeyError("n_classes > 1, not implemented yet.")
    
    
    print('Baseline prediction: {:.4f}'.format(partial_preds[0]))
    
    for node_id in node_index[:-1]: #internal nodes ( exclude leaf)
        # continue to the next node if it is a leaf node
        #if leaf_id[sample_id] == node_id:
        #    continue
    
        # check if value of the split feature for sample 0 is below threshold
        if X_test.iloc[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
            next_child = children_left[node_id]
        else:
            threshold_sign = "> "
            next_child = children_right[node_id]

    
        print(
            #"node {node:3}: w: {weight:1.2f} "
            "{feature:8} {inequality} {threshold:6.2f}" 
            #" ({feature:8} = {value:6.2f})" 
            "  -->  {partial:5.4f}".format(
                #node=node_id,
                #weight=node_weights[node_id],
                #sample=sample_id,
                feature=X_test.columns[feature[node_id]],
                #value=X_test.iloc[sample_id, feature[node_id]],
                inequality=threshold_sign,
                threshold=threshold[node_id],
                partial=partial_preds[next_child]
            )
        )
        
            
    X_sample = X_test.iloc[sample_id,:].values.reshape(1,-1)
    
    if hasattr(clf, "predict_proba"):
        y_pred = clf.predict_proba(X_sample)[:,1]
    elif hasattr(clf, "predict"):
        y_pred = clf.predict(X_sample)
    else:
        print(clf)
        KeyError("check tree method, is it correct?")

    print(
        "leaf {node:3}: predicts:{predict:5.4f}".format(
            node=node_index[-1],
            predict=float(y_pred)
        )
    )




def rule_to_code(tree, scenario, rule_nodes, sample, feature_names, full_save_name):
    learner = tree
    
    if isinstance(learner, sklearn.tree._classes.DecisionTreeClassifier):
        leaf_print = learner.predict_proba([sample.values])[:,1].ravel()
    elif isinstance(learner, sklearn.tree._classes.DecisionTreeRegressor):
        leaf_print = learner.predict([sample.values]).ravel() # risk score
    elif isinstance(learner, sksurv.tree._classes.SurvivalTree):
        leaf_surv_curve = learner.predict_survival_function([sample.values], return_array=False)
        

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
                    if rule_nodes[node] == 1 and sample[tree_.feature[node]] <= threshold:
                        intervals[name][1] = threshold # reduce feature upper bound
                        rule_nodes[node] = 0
                        f.write("node.{}: {}if {} <= {}\n".format(node, indent, name, threshold))
                        
                    recurse(tree_.children_left[node], depth + 1, sample, intervals)
                    
                    if rule_nodes[node] == 1 and sample[tree_.feature[node]] > threshold:
                        intervals[name][0] = threshold # increase feature lower bound 
                        rule_nodes[node] = 0
                        #print("node.{}: {}if {} > {}".format(node, indent, name, threshold))
                        f.write("node.{}: {}if {} > {}\n".format(node, indent, name, threshold))
                    recurse(tree_.children_right[node], depth + 1, sample, intervals)
                else: #it is undefined, it is therefore a leaf (?)
                    if rule_nodes[node] == 1:
                        #print("leafnode.{}: {}return {}".format(node, indent, leaf_print2)) #tree_.value[node].ravel()
                        f.write("leafnode.{}: {}return {}\n".format(node, indent, leaf_print))
                        name_save_plot = full_save_name.split(".")[0] + "-plot.png"
                        
                        if scenario in ["survival", "surv"]:
                            plt.figure()
                            #plt.title("Tree {} predicting sample {}".format()) not available at this level, it's in main!!
                            plt.plot(leaf_surv_curve[0].x, leaf_surv_curve[0].y)
                            plt.ylim(ymin=0, ymax=1)
                            plt.savefig(name_save_plot)
                            plt.show()
                        
                        f.write("predicted:{}\n".format(leaf_print))
            recurse(0, 1, sample, intervals)
            f.close()


    
def rule_to_code_and_intervals(tree, scenario, rule_nodes, sample, feature_names, full_save_name):
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
                    if rule_nodes[node] == 1 and sample[tree_.feature[node]] <= threshold:
                        intervals[name][1] = threshold # reduce feature upper bound
                        rule_nodes[node] = 0
                        f.write("node.{}: {}if {} <= {}\n".format(node, indent, name, threshold))
                        
                    recurse(tree_.children_left[node], depth + 1, sample, intervals)
                    
                    if rule_nodes[node] == 1 and sample[tree_.feature[node]] > threshold:
                        intervals[name][0] = threshold # increase feature lower bound 
                        rule_nodes[node] = 0
                        #print("node.{}: {}if {} > {}".format(node, indent, name, threshold))
                        f.write("node.{}: {}if {} > {}\n".format(node, indent, name, threshold))
                    recurse(tree_.children_right[node], depth + 1, sample, intervals)
                else: #it is undefined, it is therefore a leaf (?)
                    if rule_nodes[node] == 1:
                        #print("leafnode.{}: {}return {}".format(node, indent, leaf_print2)) #tree_.value[node].ravel()
                        f.write("leafnode.{}: {}return {}\n".format(node, indent, leaf_print))
                        name_save_plot = full_save_name.split(".")[0] + "-plot.png"
                        
                        if scenario in ["survival", "surv"]:
                            plt.figure()
                            #plt.title("Tree {} predicting sample {}".format()) not available at this level, it's in main!!
                            plt.plot(leaf_surv_curve[0].x, leaf_surv_curve[0].y)
                            plt.ylim(ymin=0, ymax=1)
                            plt.savefig(name_save_plot)
                            plt.show()
                        
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

    # if filter_out == True and set_up.lower() in ["survival", "surv"]: # very big datasets, drop for now
    #     try:
    #         dnames.remove('ALS-imputed') # this dataset is very big
    #         dnames.remove('ALS-imputed-50') # this dataset is very big
    #         dnames.remove('ALS-imputed-90') # this dataset is very "long" (but not high dimensional)
    #         dnames.remove('aids-single-event') # 90% censoring rate (local method could crash)
    #     except:
    #         print("An error incurred, maybe the datasets to drop do not exist anymore")

    
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
    
    # works as expected also when RF_pred is not provided, that is, when it's np.nan
    v_min = min(bunch_min_value, RF_pred)
    v_max = max(bunch_max_value, RF_pred)
    
    if is_binary:
        # combat counterintuitive colouring when predictions are confident
        v_min = min(v_min, 0.8) # v_min never above 0.8
        v_max = max(v_max, 0.2) # v_min never below 0.2

    # add a bit of extra spacing on the extremes
    v_min = v_min-(v_max-v_min)*0.05
    v_max = v_max+(v_max-v_min)*0.05+0.005 # avoid the case v_min = v_max
    
    
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
    ax1.set_title('Cluster membership', fontdict={'fontsize': base_font_size+2})

    
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
        ax3.set_title('Rule-path predictions', fontdict={'fontsize': base_font_size+2})
        
        
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
        
    
    else: # multi-label/multi-target predictions: plot distance from RF preds
    
        color_map = plt.cm.get_cmap('RdYlBu')  # or "viridis", or user choice
        #norm = BoundaryNorm(np.linspace(0.2, 0.8, 256), color_map.N)
        # normalise colors min pred.--> blue, max pred. --> red to improve readability
        
        v_min, v_max = custom_axes_limit(np.array(plot_data_bunch.loss).min(),
                                         np.array(plot_data_bunch.loss).max(),
                                         RF_pred=np.nan, is_binary=False)
        
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


    plt.show()

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

