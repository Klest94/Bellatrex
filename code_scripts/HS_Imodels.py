from imodels import HSTreeClassifierCV, HSTreeRegressorCV #import any model here
from imodels import RuleFitClassifier, RuleFitRegressor
import numpy as np
import os
import pandas as pd
import warnings

from utilities import score_method
warnings.filterwarnings("ignore", category=UserWarning)

import sklearn
from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sksurv.tree import SurvivalTree


def get_node_depths(tree):
    """ Get the node depths of the decision tree """

    def get_node_depths_(current_node, current_depth, l, r, depths):
        depths += [current_depth]
        if l[current_node] != -1 and r[current_node] != -1:
            get_node_depths_(l[current_node], current_depth + 1, l, r, depths)
            get_node_depths_(r[current_node], current_depth + 1, l, r, depths)

    depths = []
    get_node_depths_(0, 0, tree.children_left, tree.children_right, depths) 
    return np.array(depths)


#%%

LEAVES = 20 # defults seems to be 20 leaves for DTs
SETUP = "bin" #surv kinda works, but without rulkelength part (.apply does not work...)

EXTRA_NOTES = ""

#from joblib import Parallel, delayed

from utilities import get_data_list
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

##########################################################################
#root_folder = "C:/Users/u0135479/Documents/GitHub/Explain-extract-trees/"
root_folder = os.getcwd()

dnames, data_scenario_folder, scenario = get_data_list(SETUP, root_folder) #scenario has extra "/"

testing_dnames = dnames#[:3]#[:4]

print("running on datasets:")
print(testing_dnames)


#probelms with regression (maybe in get_data_list ?)
dnames = list(filter(lambda f: not f.endswith('.csv'), dnames)) # throws away csv-s

binary_key_list = ["bin", "binary"]
survival_key_list = ["surv", "survival"]
multi_label_key_list = ["multi", "multi-l", "multi-label", "mtc"]
regression_key_list = ["regression", "regress", "regr"]
mt_regression_key_list = ["multi-target", "multi-t", "mtr"]
#%%

results_df = pd.DataFrame() # empty datafarme to start with. Append here

df_filename = "HS_" + SETUP.capitalize() + "_" + str(LEAVES) + EXTRA_NOTES + ".csv"


for folder in testing_dnames:
    
    dt_perform_fold = np.zeros(5)
    rf_perform_fold = np.zeros(5)
    rulelength_fold = np.zeros(5)
    rulelength_test_fold = np.zeros(5)
    rulelength_test_fold_rf = np.zeros(5)
    avg_numb_leaves = np.zeros(5)
    
    j = 0  #outer fold index, (we are performing the outer CV "manually")
        
    while j < 5: #old: folder.split(".csv")[0]
        data_single_folder = os.path.join(data_scenario_folder, folder)
        X_train = pd.read_csv(os.path.join(data_single_folder, "X_new_train"+ str(j+1) + ".csv"))
        y_train = pd.read_csv(os.path.join(data_single_folder, "y_new_train"+ str(j+1) + ".csv"))
        X_test = pd.read_csv(os.path.join(data_single_folder, "X_new_test"+ str(j+1) + ".csv"))
        y_test = pd.read_csv(os.path.join(data_single_folder, "y_new_test"+ str(j+1) + ".csv"))
        
        col_names = list(X_train.columns)
        if col_names[:2] == ["0", "1"]:  # if column names are ugly
            col_names = ["X"+str(i) for i in X_train.columns]
        
        X_train = X_train#.values
        y_train = y_train#.values
        X_test = X_test#.values
        y_test = y_test#.values
        X = pd.concat([X_train, X_test], axis=0)

        if SETUP.lower() in survival_key_list:
            y_train = y_train.to_records(index=False)
            y_test = y_test.to_records(index=False)        


        if SETUP.lower() in multi_label_key_list + mt_regression_key_list:
            orig_n_labels = y_test.shape[1]
            y_train.columns = y_test.columns # otherwise, sometimes is a bug
            for col in y_test.columns:
                if len(y_test[col].unique()) == 1 or len(y_train[col].unique()) == 1:
                    y_test.drop(col, inplace=True, axis=1)
                    y_train.drop(col, inplace=True, axis=1)
            
            n_labels = y_test.shape[1]
            print("new n* labels:", n_labels)
        
        
        # fit the model
        if SETUP in binary_key_list + multi_label_key_list:
            model = HSTreeClassifierCV(max_leaf_nodes=LEAVES,
                                       ) #max_leaf_nodes=8  # initialize a tree model and specify only 4 leaf nodes
            rf = RandomForestClassifier(max_leaf_nodes=LEAVES)
            model_rf = HSTreeClassifierCV(estimator_=rf)
            #clf = RuleFitClassifier()
        elif SETUP in regression_key_list + mt_regression_key_list:
            rf = RandomForestRegressor(max_leaf_nodes=LEAVES)
            model = HSTreeRegressorCV(max_leaf_nodes=LEAVES)   #max_leaf_nodes=8# initialize a tree model and specify only 4 leaf nodes
            model_rf = HSTreeClassifierCV(estimator_=rf)

        elif SETUP in survival_key_list:
            rf = RandomSurvivalForest(max_leaf_nodes=LEAVES)
            dst = SurvivalTree(max_leaf_nodes=LEAVES)
            model_rf = HSTreeRegressorCV(estimator_=rf)            
            model = HSTreeRegressorCV(estimator_=dst)   #max_leaf_nodes=8# initialize a tree model and specify only 4 leaf nodes
        

        model.fit(X_train, y_train, feature_names=col_names)   # fit model
        model_rf.fit(X_train, y_train, feature_names=col_names)   # fit model

        if SETUP in ["regress"]:
            y_pred = model.predict(X_test) # discrete predictions: shape is (n_test, 1)
            y_pred_rf = model_rf.predict(X_test) # discrete predictions: shape is (n_test, 1)

        elif SETUP in ["bin"]:
            y_pred = model.predict_proba(X_test)[:,1]
            y_pred_rf = model_rf.predict_proba(X_test)[:,1]

        elif SETUP in ["multi", "mtc"]:
            y_pred0 = [y0[:, 1] for y0 in model.predict_proba(X_test)]
            y_pred = np.transpose(np.array(y_pred0))
            
            y_pred0_rf = [y0[:, 1] for y0 in model_rf.predict_proba(X_test)]
            y_pred_rf = np.transpose(np.array(y_pred0_rf))
        elif SETUP in ["mtr"]:
            y_pred = model.predict(X_test) # discrete predictions: shape is (n_test, 1)
            y_pred_rf = model_rf.predict(X_test) # discrete predictions: shape is (n_test, 1)
        
        elif SETUP in survival_key_list:
            #y_pred = model.predict(X_test) # discrete predictions: shape is (n_test, 1)
            y_pred_rf = model_rf.predict(X_test) # discrete predictions: shape is (n_test, 1)

        else:
            KeyError("Scenario not recognized, or not implemented.")

        print(model) # print the model
        print("n leaves:", model.estimator_.tree_.n_leaves)
        print("----------------")

        
        # masked array, keep only array with indeces of leaf nodes (no left child)
        leaf_nodes = np.where(model.estimator_.tree_.children_left == -1)
        #tree_.n_nodes_samples
        #rule_length_train = 0
        rule_length = 0
        if SETUP not in survival_key_list:
            leaf_samples = model.estimator_.apply(X_test) # leaf memebership
        else:
            #leaf_samples = dst.estimator_.apply(X_test) # leaf memebership
            leaf_samples = np.array([np.nan]*X_test.shape[0])
            
        # of all test samples, repetition of the elements is normal
        assert leaf_samples.shape[0] == X_test.shape[0]
        for i in list(leaf_samples):
            rule_depth = get_node_depths(model.estimator_.tree_)[i]
            rule_length += rule_depth


        ### rulelength for full Random Forest
        ### same as before, but iterating over trees of the RF and summing up
        rulelength_rf = 0
        
        for tree in rf.estimators_:
            leaf_samples = tree.apply(X_test) # leaf memebership
            # of all test samples, repetition of the elements is normal
            assert leaf_samples.shape[0] == X_test.shape[0]
            for i in list(leaf_samples):
                rule_depth_rf = get_node_depths(tree.tree_)[i]
                rulelength_rf += rule_depth_rf
            

        
        rulelength_test_fold[j] = rule_length/(X_test.shape[0])
        rulelength_test_fold_rf[j] = rulelength_rf/(X_test.shape[0])

        
        dt_perform_fold[j] = score_method(y_test, y_pred, SETUP)
        rf_perform_fold[j] = score_method(y_test, y_pred_rf, SETUP)
        avg_numb_leaves[j] = model.estimator_.tree_.n_leaves

        j+=1

        
    performs = pd.DataFrame({
                    "data size" : str(X.shape),
                    "HS_"+str(LEAVES) : np.round(np.nanmean(dt_perform_fold),4),
                    "HS_RF_"+str(LEAVES) : np.round(np.nanmean(rf_perform_fold),4),
                    #"rulelength_train_"+str(LEAVES) : np.round(np.nanmean(rulelength_fold), 4),
                    "rulelength_"+str(LEAVES) : np.round(np.nanmean(rulelength_test_fold), 4),
                    "rulelength_RF"+str(LEAVES) : np.round(np.nanmean(rulelength_test_fold_rf), 4),
                    "n leaves"+str(LEAVES): np.round(np.nanmean(avg_numb_leaves),4),
                    }, 
                     index = [folder.split(".csv", 1)[0]])
        
    results_df = pd.concat([results_df, performs], axis=0) # [brackets] necessary if passing dictionary instead  
    print("DONE with: {}".format(folder).ljust(42))
    
results_df.loc["average"] = results_df.mean(axis=0, numeric_only=True)
        
#results_df.to_csv(os.path.join(root_folder, df_filename))
 
    
    
        