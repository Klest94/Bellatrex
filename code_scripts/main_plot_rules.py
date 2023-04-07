import numpy as np
import os
import pandas as pd
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.ensemble import RandomForestClassifier
from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestRegressor
from utilities import rule_to_code_and_intervals, rule_print_inline
from plot_tree_patch import plot_tree_patched

#from sklearn.metrics import roc_auc_score as auroc
#from sklearn.metrics import mean_absolute_error as mae

from utilities import get_data_list

from LocalMethod_class import LocalTreeExtractor
from TreeDissimilarity_class import TreeDissimilarity
from TreeRepresentation import count_rule_length


#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')


SETUP = "bin"
EXTRA_NOTES = ""   # <-- add to name of stored y-s and y_preds
PLOT_FIG_TREE = False

STRUCTURE = "rules" # or "trees"
FEAT_REPRESENTATION = "by_samples" # "simple" or "by_samples" # or

NOTES = EXTRA_NOTES + "PCA" + "_" + FEAT_REPRESENTATION + "_"

OVERWRITE_CSV = False
N_FOLDS = 1
MAX_TEST_SIZE = 999 # cuts computational costs, change import file
SAVE_PREDS = False # if True, stores predictions and  runs Cox-PH for surv set-up (slow)
#unfortunately, it is not super consistent across scenarios, 
# ( double) trailing underscore for regression and multi-label (?)

root_folder = "C:/Users/u0135479/Documents/GitHub/Explain-extract-trees/"
root_folder = os.getcwd()

TAIL_NOTES = "" # append at the end of .csv file, e.g. when running -p1, -p2 ...

#data_parent_folder = os.path.join(root_folder2, "datasets")
dnames, data_scenario_folder, scenario = get_data_list(SETUP, root_folder) 
dnames = list(filter(lambda f: not f.endswith('.csv'), dnames)) # keeps only dirs


#%%#########################################################################
testing_dnames = dnames[0:1]

##########################################################################

print("running on datasets:")
for data in testing_dnames:
    print(data)
print("##########################################")
print("SETUP: {}, STRUCTURE: {}, PROJ: PCA".format(SETUP, STRUCTURE))
print("FEAT. REPRESENT: {}, EXTRA NOTES: {}".format(FEAT_REPRESENTATION, EXTRA_NOTES))
print("TAIL NOTES: {}".format(TAIL_NOTES))
print("##########################################")

results_df = pd.DataFrame() # empty datafarme to start with. Append here



binary_key_list = ["bin", "binary"]
survival_key_list = ["surv", "survival"]
multi_label_key_list = ["mtc", "multi-l", "multi-label", "multi"]
mt_regression_key_list = ["multi-t", "mtr", "multi-target"]
regression_key_list = ["regression", "regress", "regr"]


for folder in testing_dnames:
    
    ######  store fold performance, and prepare data in the right format
    
    # consdier bulding on this on separate file, to clean the main_method file
    cv_fold_ensemble = np.zeros(N_FOLDS)*np.nan
    cv_n_rules = np.zeros(N_FOLDS)*np.nan

    # this function read the data fomr the correct (sub)-folder and 
    # prepares the data in the right format
    j = 0  #outer fold index, (we are performing the outer CV "manually")
            
    while j < N_FOLDS:
    
        data_single_folder = os.path.join(data_scenario_folder, folder)
        X_train = pd.read_csv(os.path.join(data_single_folder, "X_new_train"+ str(j+1) + ".csv"))
        y_train = pd.read_csv(os.path.join(data_single_folder, "y_new_train"+ str(j+1) + ".csv"))
        X_test = pd.read_csv(os.path.join(data_single_folder, "X_new_test"+ str(j+1) + ".csv"))
        y_test = pd.read_csv(os.path.join(data_single_folder, "y_new_test"+ str(j+1) + ".csv"))
        
        X_train.drop("Unnamed: 0", axis=1, errors="ignore", inplace=True)
        X_test.drop("Unnamed: 0", axis=1, errors="ignore", inplace=True)
        
        col_names = list(X_train.columns)
        if col_names[:2] == ["0", "1"]:  # if column names are ugly
            col_names = ["X"+str(i) for i in X_train.columns]
            
        
        X = pd.concat([X_train, X_test], axis=0, ignore_index=False)
        
        X.columns = col_names
        X_train.columns = col_names
        X_test.columns = col_names
        
        
        # for short runs, test size can be decreased
        if MAX_TEST_SIZE < 100:
            y_test = y_test[:MAX_TEST_SIZE]
            X_test = X_test[:MAX_TEST_SIZE]

 
        ''' prepare correct y format '''
        
        if SETUP.lower() in multi_label_key_list + mt_regression_key_list: ##drop y_labels that do not occur in a train or test fold at all
            orig_n_labels = y_test.shape[1]
            y_train.columns = y_test.columns
            ### drop labels with no positive (or negative) class! (train or test)
            for col in y_test.columns:
                if len(y_test[col].unique()) == 1 or len(y_train[col].unique()) == 1:
                    y_test.drop(col, inplace=True, axis=1)
                    y_train.drop(col, inplace=True, axis=1)
            
            n_labels = y_test.shape[1]
            print('new n labels:', n_labels)
            
        if SETUP.lower() in binary_key_list + regression_key_list:
            y_train = y_train.values.ravel()
            y_test = y_test.values.ravel()
            
        if SETUP.lower() in survival_key_list :
            y_train = y_train.to_records(index=False)
            y_test = y_test.to_records(index=False)
       
        '''data is ready, insantiate simple competing methods:
        D(S)T, R(S)F, MiniRF (again)
        CoxPH (survival only)''' 
        
        ### instantiate original R(S)F estimator
        if SETUP.lower() in survival_key_list:
            rf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
                                        n_jobs=3, random_state=0)

        elif SETUP.lower() in binary_key_list + multi_label_key_list:
            rf = RandomForestClassifier(n_estimators=100, min_samples_split=5,
                                        n_jobs=3, random_state=0)
            
        elif SETUP.lower() in regression_key_list + mt_regression_key_list:
            rf = RandomForestRegressor(n_estimators=100, min_samples_split=5,
                                        n_jobs=3, random_state=0)
        
        
        rf_params = rf.get_params()
        #fit RF here. The hyperparameters are given
        
        #fir the RF that will serve as oracle, store properties and get
        # the Tree Extractor ready
        oracle_fitted = LocalTreeExtractor(rf, SETUP, n_trees=[0.2, 0.5, 0.8], 
                                    n_dims=[2, 5, None],
                                    n_clusters=[1, 2, 3],
                                    proj_method="PCA",
                                    dissim_method="rules",
                                    feature_represent="by_samples",
                                    pre_select_trees="L2",
                                    scoring="L2",
                                    n_jobs=1,
                                    verbose=3).fit(X_train, y_train)
        
        
        # store, for every sample in the test set, the predictions from the
        # local method and the original R(S)F
        N = min(X_test.shape[0], MAX_TEST_SIZE)        
        trees_local_preds = []
        local_dissimil = np.zeros(N)
        # miniRF_dissimil = np.zeros(N)
        tot_n_splits = 0 # total for the whole fold j        
        stored_info = []
        
        if SETUP in binary_key_list + survival_key_list + regression_key_list:
        
            y_sorted = np.argsort(np.array(y_test).ravel())
            
            y_lows = y_sorted[5:20]
            y_highs = y_sorted[70:]
            y_other = y_sorted[47:52]
            y_other = np.append(y_other, [17, 18])
            y_interest = np.concatenate([y_lows, y_highs, y_other], axis=0)
            
            y_interest = [17, 18]
            
        else:
            y_interest = [i for i in range(N)]
        
        
        for i in range(N): #for every sample in the test set: predict
        
            if i in y_interest: # interesting cases, plot trees
                
                sample_score, local_pred, \
                sample_info = oracle_fitted.predict(X_test, i)  #tuning is also done
                
                if sample_info.n_clusters <= 4:
                
                    print("sample i={} fold j={}, data: {}".format(i, j,
                                                                    folder.split(".csv", 1)[0]))
                    
                    
                    # plot_preselected_trees(plot_data_bunch, kmeans, tuned_method, final_ts_idx,
                    #                        base_font_size=14)
                    
                    trees_local_preds.append(local_pred) # store for debuggind and analysis
                    
                    tree_idx = sample_info.final_trees_idx
                    clu_sizes = sample_info.cluster_sizes
                    sample_i = X_test.iloc[i,:].values.reshape(1,-1)
                    
                    
                    tree_dissim_instance = TreeDissimilarity(rf, SETUP,
                                                          sample_info.final_trees_idx,
                                                          dissim_method=STRUCTURE,
                                                          feature_represent=FEAT_REPRESENTATION,
                                                          sample=X_test.iloc[i,:])
                    
                    
                    vecs =  tree_dissim_instance.tree_to_vectors(rf, sample_info.final_trees_idx,
                                                    STRUCTURE, FEAT_REPRESENTATION,
                                                    X_test.iloc[i,:])
                    vecs = pd.DataFrame(vecs, index=tree_idx)
                    
                    vecs.loc[-1,:] = list(X.columns)
                    
                    print("extracted trees: ", tree_idx)
                    print("with weights   : ", clu_sizes/np.sum(clu_sizes))

    
                    for t in tree_idx:
                        if PLOT_FIG_TREE:
                            print(' %%%%%%%% SHOWING TREE {:d} OF FOREST %%%%%%%%'.format(t))
                            plot_tree_patched(rf[t], feature_names=list(X_train.columns),
                                      max_depth=3)
                            plt.show()
                        
                        #tree_rules = export_text(rf[t], feature_names=list(X_train.columns),
                        #                         max_depth=2)
                        #print(tree_rules)
                        
                        node_indicator = rf[t].decision_path(sample_i)
                        nodes_sample = node_indicator.indices
                        leaf_id = rf[t].apply(sample_i)
                        
                        print("Rule {:2} for sample {}:".format(t, i))
                        rule_print_inline(rf[t], X_test, i)
                        
                        print("real value: {:.4f}".format(y_test[i]))
                        print(' '*45)
                        # for i in range(vecs.shape[0]):
                        #     print("vec", vecs.iloc[i,:])
                        # print(' '*45)

                    print("final LTreeX prediction:{:.4f}".format(float(local_pred)))
                    if hasattr(rf, 'predict_proba_'):
                        print("original RF prediction:  {:.4f}".format(float(rf.predict_proba(X_test)[:,1][i])))
                    else:
                        print("original RF prediction:  {:.4f}".format(float(rf.predict(X_test)[i])))
    
                    print('-'*45)

                            #rule_to_code_and_intervals(rf[t], "bin", nodes_sample, X_test.iloc[i,:], list(X_train.columns), None)
                    
                    for idx in sample_info.final_trees_idx:
                        tot_n_splits += count_rule_length(rf, idx, X_test.iloc[i:i+1,:])
                    
                    if SETUP.lower() in multi_label_key_list + mt_regression_key_list:
                        y_true = y_test.iloc[i,:].values
                        y_local_pred =  sample_info.local_predict()
                        
                    elif SETUP.lower() in survival_key_list:
                        y_true = y_test[i]
                        y_local_pred = float(sample_info.local_predict())
        
                    else:
                        y_true = float(y_test[i])
                        y_local_pred = float(sample_info.local_predict())
                ### End condition n_clusters <= 2 for plotting rules
            ### end of the y_extremes loop 

        ### end of the range(N) loop    
        ### models are fitted, call .predict(_proba) and store in right format
            
        if SETUP.lower() in survival_key_list:
            y_ens_pred = rf.predict(X_test)
            
        elif SETUP.lower() in regression_key_list + mt_regression_key_list:
            y_ens_pred = rf.predict(X_test)
  
        elif SETUP.lower() in binary_key_list: #binary or multi-label
            y_ens_pred = rf.predict_proba(X_test)[:,1]


        elif SETUP.lower() in multi_label_key_list:
            y_ens_pred = rf.predict_proba(X_test)  #sometimes array, sometimes list
            

            # list with predictions is complete. Format correctly:
            if type(y_ens_pred) is list:
                for item in y_ens_pred: #check there are no shape errors:
                    assert item.shape == y_ens_pred[0].shape
                
                y_ens_pred = np.array(y_ens_pred)
            # now y_ens_pred is a np.ndarray for sure
            y_ens_pred = np.transpose(y_ens_pred[:,:,1])

        
        elif SETUP.lower() in mt_regression_key_list:
            y_ens_pred = rf.predict(X_test)  #sometimes array, sometimes list
            if type(y_ens_pred) is list:
                for item in y_ens_pred: #check there are no shape errors:
                    assert item.shape == y_ens_pred[0].shape
                
                y_ens_pred = np.array(y_ens_pred)
        
        else:
            raise ValueError("Set-up not recognised:", SETUP)
        
        
        prediction_single_folder = os.path.join(root_folder, "predictions", scenario, folder)
        
        naming_format = NOTES + "fold" + str(j+1) + ".npy"
        # rememeber: NOTES = EXTRA_NOTES + PROJ_METHOD + "_" + FEAT_REPRESENTATION + "_"

        cv_n_rules[j] = 0

        print("data: {}, finished fold j = {}".format(folder.split(".csv", 1)[0], j))
        j +=1  # next fold, close fold now
        
print('done.')

