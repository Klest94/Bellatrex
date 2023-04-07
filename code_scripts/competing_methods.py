import numpy as np
import os
import pandas as pd
from datetime import date
import warnings
import re
today = date.today()
print("Today's date:", today)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestRegressor

from sksurv.tree import SurvivalTree
from sksurv.linear_model import CoxPHSurvivalAnalysis#, CoxnetSurvivalAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor # multi-output Ridge

from sklearn.metrics import roc_auc_score as auroc
from sklearn.metrics import mean_absolute_error as mae

from utilities import get_data_list, score_method
from TreeDissimilarity_class import TreeDissimilarity
from TreeRepresentation import count_rule_length


#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')


SETUP = "bin"
PROJ_METHOD = "PCA"
EXTRA_NOTES = ""   # <-- add to name of stored y-s and y_preds

STRUCTURE = "rules" # or "trees"
FEAT_REPRESENTATION = "by_samples" # "simple" or "by_samples" # or

NOTES = EXTRA_NOTES + PROJ_METHOD + "_" + FEAT_REPRESENTATION + "_"

OVERWRITE_CSV = False
N_FOLDS = 2
MAX_TEST_SIZE = 19 # cuts computational costs, change import file
SAVE_PREDS = False # if True, stores predictions and  runs Cox-PH for surv set-up (slow)
#unfortunately, it is not super consistent across scenarios, 
# ( double) trailing underscore for regression and multi-label (?)
N_JOBS = 1

root_folder = "C:/Users/u0135479/Documents/GitHub/Explain-extract-trees/"
root_folder = os.getcwd()

TAIL_NOTES = "" # append at the end of .csv file, e.g. when running -p1, -p2 ...

#data_parent_folder = os.path.join(root_folder2, "datasets")
dnames, data_scenario_folder, scenario = get_data_list(SETUP, root_folder) 
dnames = list(filter(lambda f: not f.endswith('.csv'), dnames)) # keeps only dirs

try:
    dnames.remove("ALS-imputed-70")
except:
    print("nothing special")

#%%#########################################################################
testing_dnames = dnames[0:4]

##########################################################################

print("running on datasets:")
for data in testing_dnames:
    print(data)
print("##########################################")
print("SETUP: {}, STRUCTURE: {}, PROJ: {}".format(SETUP, STRUCTURE, PROJ_METHOD))
print("FEAT. REPRESENT: {}, EXTRA NOTES: {}".format(FEAT_REPRESENTATION, EXTRA_NOTES))
print("TAIL NOTES: {}".format(TAIL_NOTES))
print("##########################################")

results_df = pd.DataFrame() # empty datafarme to start with. Append here


binary_key_list = ["bin", "binary"]
survival_key_list = ["surv", "survival"]
multi_label_key_list = ["mtc", "multi-l", "multi-label", "multi"]
multi_target_key_list = ["multi-t", "mtr", "multi-target"]
regression_key_list = ["regression", "regress", "regr"]


for folder in testing_dnames:
    
    ######  store fold performance, and prepare data in the right format
    
    # consdier bulding on this on separate file, to clean the main_method file
    cv_fold_ensemble = np.zeros(N_FOLDS)*np.nan
    cv_fold_DT = np.zeros(N_FOLDS)*np.nan
    cv_fold_bestTs = np.zeros(N_FOLDS)*np.nan
    cv_fold_miniRF = np.zeros(N_FOLDS)*np.nan
    
    cv_dissim_bestTs = np.zeros(N_FOLDS)*np.nan
    cv_dissim_miniRF = np.zeros(N_FOLDS)*np.nan

    #cv_dissim_bestTs_fid = np.zeros(N_FOLDS)*np.nan
    cv_n_splits_DT = np.zeros(N_FOLDS)*np.nan
    cv_n_splits_RF = np.zeros(N_FOLDS)*np.nan
    cv_n_splits_bestTs = np.zeros(N_FOLDS)*np.nan
    cv_n_splits_miniRF = np.zeros(N_FOLDS)*np.nan
    cv_n_rules = np.zeros(N_FOLDS)*np.nan
    
    cv_fold_coxPH = np.zeros(N_FOLDS)*np.nan
    cv_fold_lin_model = np.zeros(N_FOLDS)*np.nan
    cv_fold_ridge_model = np.zeros(N_FOLDS)*np.nan

    # this function read the data fomr the correct (sub)-folder and 
    # prepares the data in the right format
    j = 0  #outer fold index, (we are performing the outer CV "manually")
            
    while j < N_FOLDS:

        #params_df = pd.read_csv(hyperparams_path + "Params" + EXTRA_NOTES + "_" + folder.split(".csv")[0] + "_" + str(j+1) + ".csv")
        # C:/Users/u0135479/Documents/GitHub/Explain-extract-trees/tuned_hyperparams/survival/
        params_scenario_folder = os.path.join(root_folder, "tuned_hyperparams", scenario)
        params_single_folder = os.path.join(params_scenario_folder, folder)
        params_name_piece = "Params_" + NOTES + \
                                STRUCTURE + "_fold" + str(j+1) + ".csv"
        params_filename = os.path.join(params_single_folder, params_name_piece)
        params_df = pd.read_csv(params_filename)
        opt_clusters = params_df["n_clusters"]
    
        data_single_folder = os.path.join(data_scenario_folder, folder)
        X_train = pd.read_csv(os.path.join(data_single_folder, "X_new_train"+ str(j+1) + ".csv"))
        y_train = pd.read_csv(os.path.join(data_single_folder, "y_new_train"+ str(j+1) + ".csv"))
        X_test = pd.read_csv(os.path.join(data_single_folder, "X_new_test"+ str(j+1) + ".csv"))
        y_test = pd.read_csv(os.path.join(data_single_folder, "y_new_test"+ str(j+1) + ".csv"))
        
        X = pd.concat([X_train, X_test], axis=0, ignore_index=False)
        #y_test2 = pd.read_csv(datapath + folder.split(".csv")[0] +  "/" + "y_old_test"+ str(j+1) + ".csv")
        
        # for short runs, test size can be decreased
        if MAX_TEST_SIZE < 100:
            y_test = y_test[:MAX_TEST_SIZE]
            X_test = X_test[:MAX_TEST_SIZE]

 
        ''' prepare correct y format '''
        
        if SETUP.lower() in multi_label_key_list + multi_target_key_list: ##drop y_labels that do not occur in a train or test fold at all
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
        

        scaler = StandardScaler().fit(X_train)
        X_train2 = scaler.transform(X_train)
        X_test2 = scaler.transform(X_test)
        
        # train as well
        if SETUP.lower() in survival_key_list:
            rf = RandomSurvivalForest(min_samples_split=10,
                                       n_jobs=N_JOBS, oob_score=True,
                                       random_state=0).fit(X_train, y_train)
        
            

            clf_tree = SurvivalTree(min_samples_split=10, random_state=0)
            clf_tree.fit(X_train, y_train)
            #y_ens_pred = rf.predict(X_test)
            if SAVE_PREDS == True:
                
                alpha_cox = 1e-4 if X_train.shape[0] > X_train.shape[1] else 1e-2
                print("alpha regul:", alpha_cox)
                # consider adding this to clf_cox
                clf_cox = CoxPHSurvivalAnalysis(alpha=alpha_cox, n_iter=2e4,
                                            ties="efron")
                
            # try:# If it doesn' work, use Coxnet survival analysis
            #     clf_cox.fit(X_train2, y_train)
            #     y_cox = clf_cox.predict(X_test2)
                
                result = None
                while result == None:
                    try:
                        clf_cox.fit(X_train2, y_train)
                        y_cox = clf_cox.predict(X_test2)
                        result = True # if it works iwhtout issue, break the while loop
                        print("Fit successful with alpha={}".format(alpha_cox))
                    except:
                        print("failed with alpha={}, trying with alpha={}:".format(alpha_cox, alpha_cox*10))
                        alpha_cox = alpha_cox*10
                        clf_cox = CoxPHSurvivalAnalysis(alpha=alpha_cox, n_iter=2e4,
                                                    ties="efron")
                    # increase regularisation by a factor of 10 and try again
            
            
            # except: # ValueError: use CoxNet with increasing regularization
            #     print("exception raised!")
            #     alpha_cox = alpha_cox*10
            #     #clf_cox = CoxnetSurvivalAnalysis(n_alphas=10) #weird plan b, but it works...
            #     CoxPHSurvivalAnalysis(alpha=alpha_cox, n_iter=2e4,
            #                                 ties="efron")
            #     clf_cox.fit(X_train2, y_train)
            #     # still to test
            #     for a in [clf_cox.alphas_[1]]: #get 2nd alpha only (dumb workaround)
            #         print('new regul:', a)
            #         y_cox = clf_cox.predict(X_test2)
        
        elif SETUP.lower() in regression_key_list + multi_target_key_list:
            rf = RandomForestRegressor(min_samples_split=5, oob_score=True,
                                        n_jobs=-3,
                                        random_state=0).fit(X_train, y_train)

            clf_tree = DecisionTreeRegressor(min_samples_leaf=5, random_state=0)
            clf_tree.fit(X_train, y_train)
            
            if SETUP.lower() in multi_target_key_list:
            
                clf_ridge = MultiOutputRegressor(Ridge(alpha=1, random_state=0)).fit(X_train2, y_train)            
            
            elif SETUP.lower() in regression_key_list:
                
                clf_ridge = Ridge(alpha=1, random_state=0).fit(X_train2, y_train)            

            
        elif SETUP.lower() in binary_key_list:
            rf = RandomForestClassifier(min_samples_split=5, oob_score=True,
                                        n_jobs=-3,
                                        random_state=0).fit(X_train, y_train)
            
            clf_tree = DecisionTreeClassifier(min_samples_leaf=5, random_state=0)
            clf_tree.fit(X_train, y_train)
            
            clf_linear = LogisticRegression(penalty='elasticnet', solver='saga',
                                            max_iter=1e6, l1_ratio=0.5)
            clf_linear.fit(X_train2, y_train)


        elif SETUP.lower() in multi_label_key_list:
            rf = RandomForestClassifier(min_samples_split=5, oob_score=True,
                                        n_jobs=-3,
                                        random_state=0).fit(X_train, y_train)
            
            clf_tree = DecisionTreeClassifier(min_samples_leaf=5, random_state=0)
            clf_tree.fit(X_train, y_train)
            
            clf_linear = LogisticRegression(penalty='elasticnet', solver='saga',
                                            max_iter=1e6, l1_ratio=0.5)
            
            clf_mlc = [] # list of predicting models, one per target variable
            
            # iterate over all labels :(
            for i in range(y_train.shape[1]):
                y_train_i = y_train.iloc[:,i]
                clf_mlc.append(clf_linear.fit(X_train2, y_train_i)) # why doesn't it work!?
            
            
        ## models are fitted, call .predict(_proba) and store in right format
            
        if SETUP.lower() in survival_key_list:
            y_ens_pred = rf.predict(X_test)
            DTree_pred = clf_tree.predict(X_test)
            
        elif SETUP.lower() in regression_key_list + multi_target_key_list:
            y_ens_pred = rf.predict(X_test)
            DTree_pred = clf_tree.predict(X_test)
            ridge_pred = clf_ridge.predict(X_test2) # does this work in MTR setting?
            
        elif SETUP.lower() in binary_key_list: #binary or multi-label
            y_ens_pred = rf.predict_proba(X_test)[:,1]
            DTree_pred = clf_tree.predict_proba(X_test)[:,1] #
            lin_pred = clf_linear.predict_proba(X_test2)[:,1]

        elif SETUP.lower() in multi_label_key_list:
            y_ens_pred = rf.predict_proba(X_test)  #sometimes array, sometimes list
            
            # need to iterate over all labels
            lin_pred_raw = []
            for i in range(y_train.shape[1]):
                lin_pred_raw.append(clf_mlc[i].predict_proba(X_test2)[:,1])

            # list with predictions is complete. Format correctly:
            lin_pred = np.array(lin_pred_raw).T # shape (N,d)

            if type(y_ens_pred) is list:
                for item in y_ens_pred: #check there are no shape errors:
                    assert item.shape == y_ens_pred[0].shape
                
                y_ens_pred = np.array(y_ens_pred)
            # now y_ens_pred is a np.ndarray for sure
            y_ens_pred = np.transpose(y_ens_pred[:,:,1])

            DT_orig_pred = clf_tree.predict_proba(X_test) #sometimes array, sometimes list
            label_pred = np.array([lab[:,1] for lab in DT_orig_pred]) #get prob of posit label
            DTree_pred = np.transpose(label_pred)
        
        elif SETUP.lower() in multi_target_key_list:
            y_ens_pred = rf.predict(X_test)  #sometimes array, sometimes list
            if type(y_ens_pred) is list:
                for item in y_ens_pred: #check there are no shape errors:
                    assert item.shape == y_ens_pred[0].shape
                
                y_ens_pred = np.array(y_ens_pred)
            # now y_ens_pred is a np.ndarray for sure
            # y_ens_pred = np.transpose(y_ens_pred[:,:,1])
            # DT_orig_pred
            DTree_pred = clf_tree.predict(X_test) #sometimes array, sometimes list
            # label_pred = np.array([lab[:,1] for lab in DT_orig_pred]) #get prob of posit label
            # DTree_pred = np.transpose(label_pred)
        
        else:
            raise ValueError("Set-up not recognised:", SETUP)
        
            
        ### best trees here:
        
        from sklearn.ensemble._forest import _generate_unsampled_indices
        #from sklearn.forest import _generate_unsampled_indices

        n_samples = X_train.shape[0]
        oob_score_trees = []
        for tree in rf.estimators_:
            
            tree.feature_names_in_ = rf.feature_names_in_
            # Here at each iteration we obtain out of bag samples for every tree.
            np_index = _generate_unsampled_indices(
                                        tree.random_state, n_samples, n_samples)
            df_index = X_train.iloc[np_index,:].index

            X_oob = X_train.loc[df_index]
            if y_train.ndim == 1: # single target
                y_oob = y_train[np_index]
            else:
                y_train.index = X_train.index
                y_oob = y_train.loc[df_index]

            if SETUP in survival_key_list:
                y_pred = tree.predict(X_oob)
                oob_score = ([i[0] for i in y_oob], [i[1] for i in y_oob],
                                         y_pred)[0] #c_index(y_oob, y_pred)
            
            
            elif SETUP in regression_key_list:
                y_pred = tree.predict(X_oob)
                try: # the higher the better (score instead of loss)
                    oob_score = (-1)*mae(y_oob, y_pred)
                except ValueError:
                    oob_score = -1e5
            
            elif SETUP in binary_key_list:
                y_pred = tree.predict_proba(X_oob)[:,1]
                try: 
                    oob_score = auroc(y_oob, y_pred, average="weighted")
                except ValueError:
                    oob_score = 0.5
                
            elif SETUP in multi_label_key_list:
                y_pred0 = tree.predict_proba(X_oob)
                y_pred = np.transpose(np.array(y_pred0)[:,:,1])
                try:
                    oob_score = auroc(y_oob, y_pred, average="weighted")
                except ValueError:
                    oob_score = 0.5            
            
            else: #multi-target
                y_pred0 = tree.predict(X_oob)
                #y_pred0 = np.transpose(np.array(y_pred0)[:,:,1])
                try: # the higher the better (score instead of loss)
                    oob_score = (-1)*mae(y_oob, y_pred0, multioutput='uniform_average')
                except ValueError:
                    oob_score = -1e5

            oob_score_trees.append(oob_score)


        # all oob scores are now stored
        oob_score_trees = np.array(oob_score_trees)
        args_trees = oob_score_trees.argsort()[::-1] # from highest to lowest
        
        best_tree_indeces = args_trees[:4]
        ## in theory, it can handle all five scanrios! We will test it soon
        
        def local_predict(clf, final_trees_idx, sample): #self clf, X, sample, 
            
            if final_trees_idx is None:
                N_trees = clf.n_estimators
                final_trees_idx = range(N_trees) # update to [0, 1, ... N_trees]
            #else:
            N_trees = len(final_trees_idx)
            #print('"selected indeces:', final_trees_idx)
            
            if hasattr(clf, "predict_proba"):
                preds_list = np.sum([clf[t].predict_proba([X_test.iloc[sample,:]]) 
                                     for t in final_trees_idx], axis=0)
                try: #multi-label case
                    y_pred_multi = np.transpose([pred[:, 1][0] for pred in preds_list]) # (n_labels,)
                except IndexError: #binary case
                    y_pred_multi = np.transpose([pred[1] for pred in preds_list]) # (1,)
                
                my_prediction = y_pred_multi
        
            elif hasattr(clf, "predict"): #survival set-up (to be reniewed)
                for t in final_trees_idx:
                    my_prediction = np.sum([clf[t].predict([X_test.iloc[sample,:]])
                                           for t in final_trees_idx], axis=0)
            else:
                raise ValueError("oracle doesn't have .predict nor .predict_proba method")
    
            return (1/N_trees)*my_prediction  #float in case .predict gives an array
        
        # provisional fix here:
        N = min(X_test.shape[0], MAX_TEST_SIZE)

        if SETUP.lower() in multi_label_key_list + multi_target_key_list:
            BestTs_pred = np.zeros([N, n_labels])
            Mini_rf_pred = np.zeros([N, n_labels])

            for i in range(N):
                BestTs_pred[i,:] = local_predict(rf, best_tree_indeces[:opt_clusters[i]], i)
                #Mini_rf_pred[i, :] = local_predict(rf, None, i)

        
        else: # if set other than multi-label:
            BestTs_pred = np.zeros(N)
            Mini_rf_pred = np.zeros(N)

            
            for i in range(N):
                BestTs_pred[i] = local_predict(rf, best_tree_indeces[:opt_clusters[i]], i)
                #Mini_rf_pred[i] = local_predict(rf, None, i)

        
        ''' Mini-RF now , with n_estimators= n_clusters '''
        from sklearn.base import clone
        mini_rf = clone(rf)
        
        Mini_rf_dissim = np.zeros(N)

        n_splits_miniRF = 0 # total splits along folder
        n_splits_DT = 0
        n_splits_RF = 0
        
        for i in range(N):
            mini_rf = mini_rf.set_params(n_estimators=opt_clusters[i], 
                                         oob_score=False, random_state=99999*j+i)
            mini_rf.fit(X_train, y_train)
                
            Mini_rf_dissim[i], _ = TreeDissimilarity(mini_rf, SETUP, None, 
                                            STRUCTURE,
                                            sample=X_test.iloc[i,:],
                                            feature_represent=FEAT_REPRESENTATION,
                                            ).compute_dissimilarity()
                 
            Mini_rf_pred[i] = local_predict(mini_rf, None, i)
            
            
            
            for idx in range(opt_clusters[i]):
                n_splits_miniRF  += count_rule_length(rf, idx, X_test.iloc[i:i+1,:])
            
            
            n_splits_DT  += count_rule_length(clf_tree, idx, X_test.iloc[i:i+1,:])
            
            for idx in range(rf.n_estimators):
                n_splits_RF  += count_rule_length(rf, idx, X_test.iloc[i:i+1,:])
                
        
        cv_n_splits_miniRF[j] = n_splits_miniRF/N # fold average n of splits in the explanation
        cv_n_splits_RF[j] = n_splits_RF/N # fold average n of splits in the explanation
        cv_n_splits_DT[j] = n_splits_DT/N # fold average n of splits in the explanation

        cv_n_rules[j] = np.nanmean(opt_clusters) # fold average n of trees/rulepaths
        


        ''' saving all stored predictions now '''
                    
        # predictions are stored in the correct format, hopefully:
            
            
        cv_fold_DT[j] = score_method(y_test, DTree_pred, #list or array, right?
                                        SETUP)
        
        cv_fold_ensemble[j] = score_method(y_test, y_ens_pred, #list or array, right?
                                        SETUP)
        
        assert y_test.shape[0] == BestTs_pred.shape[0]
   
        
        cv_fold_bestTs[j] = score_method(y_test, BestTs_pred, #list or array, right?
                                        SETUP)    
        cv_fold_miniRF[j] = score_method(y_test, Mini_rf_pred, #list or array, right?
                                        SETUP)

        best_T_dissim = np.zeros(N)

        n_splits_fold_bestTs = 0 # total for the whole fold j

        for i in range(X_test.shape[0]):
            
            best_T_dissim[i], _ = TreeDissimilarity(rf, SETUP,
                                                    best_tree_indeces[:opt_clusters[i]],
                                                    dissim_method=STRUCTURE,
                                                    sample=X_test.iloc[i,:],
                                                    feature_represent=FEAT_REPRESENTATION
                                                    ).compute_dissimilarity()
            
            #tree_idx = list(map(int, re.findall(r'\d+', df_trees[i])))
            for idx in best_tree_indeces[:opt_clusters[i]]:
                n_splits_fold_bestTs  += count_rule_length(rf, idx, X_test.iloc[i:i+1,:])
                
        
        cv_n_splits_bestTs[j] = n_splits_fold_bestTs/N
                                    
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cv_dissim_bestTs[j] = np.nanmean(best_T_dissim)
            cv_dissim_miniRF[j] = np.nanmean(Mini_rf_dissim)

        
        prediction_single_folder = os.path.join(root_folder, "predictions", scenario, folder)
        
        naming_format = NOTES + "fold" + str(j+1) + ".npy"
        # rememeber: NOTES = EXTRA_NOTES + PROJ_METHOD + "_" + FEAT_REPRESENTATION + "_"

        if SAVE_PREDS == True: # create directories and store numpy predictions

            if not os.path.exists(prediction_single_folder):
                os.makedirs(prediction_single_folder)
                
            np.save(os.path.join(prediction_single_folder, "BestTs_preds_" + STRUCTURE + "_" + naming_format), BestTs_pred)
            np.save(os.path.join(prediction_single_folder, "BestTs_dissim_" + STRUCTURE + "_" + naming_format), best_T_dissim)
            np.save(os.path.join(prediction_single_folder, "MiniRF_preds_" + STRUCTURE + "_" + naming_format), Mini_rf_pred)
            np.save(os.path.join(prediction_single_folder, "MiniRF_dissim_" + STRUCTURE + "_" + naming_format), Mini_rf_dissim)
            # if not os.path.exists(prediction_single_folder):
            #     os.makedirs(prediction_single_folder)
        
        if SETUP.lower() in survival_key_list and SAVE_PREDS == True:
            cox_file_fold = os.path.join(prediction_single_folder, "Cox_pred_" + naming_format)
            np.save(cox_file_fold, y_cox)
            cv_fold_coxPH[j] = score_method(y_test, y_cox, #list or array, right?
                                        SETUP)    

        elif SETUP.lower() in binary_key_list + multi_label_key_list:
            lin_reg_file_fold = os.path.join(prediction_single_folder, "LR_pred_" + naming_format)
            np.save(lin_reg_file_fold, lin_pred)
            cv_fold_lin_model[j] = score_method(y_test, lin_pred, #list or array, right?
                                        SETUP)
        
        elif SETUP.lower() in regression_key_list + multi_target_key_list: 
            ridge_file_fold = os.path.join(prediction_single_folder, "Ridge_pred_" + naming_format)
            np.save(ridge_file_fold, ridge_pred)
            cv_fold_ridge_model[j] = score_method(y_test, ridge_pred, #list or array, right?
                                        SETUP)
        
        np.save(os.path.join(prediction_single_folder, "SingleDT_preds_" + STRUCTURE + "_" + naming_format), DTree_pred)

        # count splits for MiniRF and BestTs now:
        params_scenario_folder = os.path.join(root_folder, "tuned_hyperparams", scenario, folder)
        params_filename = "Params_" + NOTES + STRUCTURE + "_fold" + str(j+1) + ".csv"
        df_trees = pd.read_csv(os.path.join(params_scenario_folder, params_filename))["trees_idx"]

        print("data: {}, finished fold j = {}".format(folder.split(".csv", 1)[0], j))
        j +=1  # next fold, close fold now
        
        
    #storing average performances in dataframe here:
    performances = pd.DataFrame({"data size" : str(X.shape),
                    "Single D(S)T" : np.round(np.nanmean(cv_fold_DT),4),
                    "Single D(S)T n_splits" : np.round(np.nanmean(cv_n_splits_DT),4),
                    "orig. R(S)F" : np.round(np.nanmean(cv_fold_ensemble),4),
                    "R(S)F n_splits" : np.round(np.nanmean(cv_n_splits_RF),4),
                    "--": "",
                    "Best (S)T" : np.round(np.nanmean(cv_fold_bestTs),4),
                    "BestTs diss" : np.round(np.nanmean(cv_dissim_bestTs),4),
                    "BestT n_splits" : np.round(np.nanmean(cv_n_splits_bestTs),4),
                    "---": "",
                    "Mini R(S)F " : np.round(np.nanmean(cv_fold_miniRF),4),
                    "Mini R(S)F diss" : np.round(np.nanmean(cv_dissim_miniRF),4),
                    "MiniRF n_splits" : np.round(np.nanmean(cv_n_splits_miniRF),4),
                    "-": "",
                    "n_rules" : np.round(np.nanmean(cv_n_rules), 4),
                    }, 
                     index = [folder.split(".csv", 1)[0]])
    if SETUP.lower() in multi_label_key_list:
        performances.insert(1, "n labels", orig_n_labels) # add extra info
    
    if SETUP.lower() in survival_key_list and SAVE_PREDS == True:
        performances.insert(3, "Cox-PH", np.round(np.nanmean(cv_fold_coxPH),4))
    elif SETUP.lower() in binary_key_list + multi_label_key_list:
        performances.insert(3, "Logistic-Regr", np.round(np.nanmean(cv_fold_lin_model),4))
    
    elif SETUP.lower() in regression_key_list + multi_target_key_list:
        performances.insert(3, "Ridge-Regr", np.round(np.nanmean(cv_fold_ridge_model),4))

    results_df = pd.concat([results_df, performances], axis=0) # [brackets] necessary if passing dictionary instead  

    print("DONE with: {}".format(folder).ljust(42))

#averages = pd.DataFrame(np.round(results_df.mean(),4), columns= ["avg. perform"]).T
#results_df = results_df.append(averages, ignore_index=False)

results_df.loc["average"] = np.round(results_df.mean(axis=0, numeric_only=True),4)

if OVERWRITE_CSV == True:    
    name_info = "Draft_" if SAVE_PREDS == False else ""
    name_info = name_info + NOTES + STRUCTURE.capitalize() + "_"
    df_filename = name_info + SETUP.capitalize() + "_Competing_" + str(today) + TAIL_NOTES + ".csv"
    results_df.to_csv(os.path.join(root_folder, df_filename))
