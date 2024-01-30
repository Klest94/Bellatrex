import numpy as np
import pandas as pd
import os
# avoids memory leak UserWarning caused by KMeans (DOES IT EVEN WORK?)
os.environ["OMP_NUM_THREADS"] = "1" 
import datetime
import warnings

from datetime import date
from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from code_scripts.utilities import get_data_list, score_method
from code_scripts.utilities import format_targets
from code_scripts.TreeDissimilarity_class import TreeDissimilarity
from code_scripts.TreeRepresentation_utils import count_rule_length

from code_scripts.LocalMethod_class import Bellatrex
from code_scripts.wrapper_class import tree_list_to_model, DT_to_dict

SETUP = "surv"
PROJ_METHOD = "PCA"

SAVE_PREDS = False # if True: save: predictions, tuned_hyperparams and dataframe
# otherwise if False, the final file is stored with the "Draft_" prefix
OVERWRITE_DF = False # save performance df as csv, potentially overwriting smth
# reduce MAX_TEST_SIZE for quick code testing
MAX_TEST_SIZE = 3 #if set >= 100, it takes the (original) value X_test.shape[0]
VERBOSE = 4
PLOT_GUI = True
N_FOLDS = 1

p_grid = {
    "n_trees": [0.8, 1.0], # [100] for "noTrees_" ablation
    "n_dims": [2, None], #None = no dim reduction   #Ablation: noDims_
    "n_clusters": [2, 3]
    }


STRUCTURE = "rules" # accepted keys: "trees" or "rules"
FEAT_REPRESENTATION = "weighted" # "simple" or "by_samples" # or
FINAL_SCORING = "L2" # L2 or cosine

##########################################################################
root_folder = os.getcwd()

dnames, data_scenario_folder, scenario = get_data_list(SETUP, root_folder) #scenario has extra "/"
#hyperparams_scenario_folder =  os.path.join(root_folder, "datasets", scenario[:-1])
# <-- add to name of stored y-s and y_preds ??
EXTRA_NOTES = "trial_"  # noDims_ or noTrees_ or Ablat2_ or trial_
TAIL_NOTES = "p1" # ONLY FOR FINAL CSV FILE

NOTES = EXTRA_NOTES + PROJ_METHOD + "_" + FEAT_REPRESENTATION + "_"

testing_dnames = dnames[4:5]#[:4]
#testing_dnames = [dnames[i] for i in [4, 6, 8, 10, 11]]
##########################################################################

from code_scripts.utilities import validate_paramater_run
validate_paramater_run(p_grid, EXTRA_NOTES, N_FOLDS)

print("running on datasets:")
print(testing_dnames)

# PARALLELISATION NOT WORKING SUPER WELL YET. GAINS ARE MARGINAL
JOBS = 1 # n_jobs for R(S)F learner (and now also ETrees candidate choice!)
'''  levels of verbosity in this script:
    - >= 1.0: print best params, their achieved fidelity,
              and the scoring method used to compute such performance
    - >= 2.0: print final tree idx cluster sizes
              and store txt files with the extracted rule-paths
    - >= 3.0: plot representation of the extracted trees (two plots)
    - >= 4.0: plot trees with GUI (if PLOT_GUI == True)
    - >= 4.0: plot trees without GUI (if PLOT_GUI == False)
    - >= 5.0: print params and performance during GridSearch
'''

'''
SET keywords to be used across the 5 scenarios implemented so far.
Needed because:
    - different data subfolders
    - different manipulation steps within Bellatrex 
'''

binary_key_list = ["bin", "binary"]
survival_key_list = ["surv", "survival"]
multi_label_key_list = ["multi", "multi-l", "multi-label", "mtc"]
regression_key_list = ["regression", "regress", "regr"]
mt_regression_key_list = ["multi-target", "multi-t", "mtr"]
multi_sa_key_list = ["multi-sa", "multi-variate-sa", "mvsa"] #not implemented yet


print("##########################################")
print("SETUP: {}, STRUCTURE: {}, PROJ: {}".format(SETUP, STRUCTURE, PROJ_METHOD))
print("FEAT. REPRESENT: {}, EXTRA NOTES: {}".format(FEAT_REPRESENTATION, EXTRA_NOTES))
print("TAIL NOTES: {}".format(TAIL_NOTES))
print("##########################################")


results_df = pd.DataFrame() # empty datafarme to start with. Append here


# custom filename, use date to avoid overwriting potentially useful data
name_info = "Draft_" if SAVE_PREDS == False else ""
name_info = name_info + NOTES + STRUCTURE.capitalize() + "_"
df_filename = name_info + SETUP.capitalize() + str(date.today())[4:] + TAIL_NOTES + ".csv"

# loop across the list of datasets
for dataset in testing_dnames:
    
    ######  store fold performance
    cv_fold_ensemble = np.zeros(N_FOLDS)*np.nan
    cv_fold_shallow = np.zeros(N_FOLDS)*np.nan
    cv_fold_local = np.zeros(N_FOLDS)*np.nan
    cv_local_dissimil = np.zeros(N_FOLDS)*np.nan
    avg_splits_fold = np.zeros(N_FOLDS)*np.nan

    t1 = datetime.datetime.now()    
    
    j = 0  #outer fold index, (we are performing the outer CV "manually")
    
    # read the (maximum 5) datafolds, generated in advance
        
    while j < N_FOLDS:
        data_single_folder = os.path.join(data_scenario_folder, dataset)
        X_train = pd.read_csv(os.path.join(data_single_folder, "X_new_train"+ str(j+1) + ".csv"))
        y_train = pd.read_csv(os.path.join(data_single_folder, "y_new_train"+ str(j+1) + ".csv"))
        X_test = pd.read_csv(os.path.join(data_single_folder, "X_new_test"+ str(j+1) + ".csv"))
        y_test = pd.read_csv(os.path.join(data_single_folder, "y_new_test"+ str(j+1) + ".csv"))
        
        X_train.drop("Unnamed: 0", axis=1, errors="ignore", inplace=True)
        X_test.drop("Unnamed: 0", axis=1, errors="ignore", inplace=True)
        X = pd.concat([X_train, X_test], axis=0, ignore_index=False)
        assert X.isnull().sum().sum() < 1 #make sure there are no null values
        # NaNs are supposed to be filled in, e.g. with MICE
        print(X_train.mean())
        
        # for quick testing, set a small MAX_TEST_SIZE
        # X_test = X_test[:MAX_TEST_SIZE]
        y_test = y_test[:MAX_TEST_SIZE]
        
        orig_n_labels = y_test.shape[1] #meaningful only in multi-output

        # set target variable to correct format depending on the prediciton
        # scenarios.E.g. set np.recarray fo survival data, or normalise data 
        # in case of single and multi-target regression
        
        y_train, y_test = format_targets(y_train, y_test, SETUP, VERBOSE)
            

        ### instantiate original R(S)F estimator
        if SETUP.lower() in survival_key_list:
            rf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
                                        n_jobs=4, random_state=0)

        elif SETUP.lower() in binary_key_list + multi_label_key_list:
            rf = RandomForestClassifier(n_estimators=100, min_samples_split=5,
                                        n_jobs=4, random_state=0)
            
        elif SETUP.lower() in regression_key_list + mt_regression_key_list:
            rf = RandomForestRegressor(n_estimators=100, min_samples_split=5,
                                        n_jobs=4, random_state=0)
        
        # dictionary testing for BELLATREX here
        
        rf.fit(X_train, y_train)
        
        tree_list = []
            
        for t in range(rf.n_estimators):
               
            tree_dict = DT_to_dict(rf, t, 'auto',
                                       T_to_bin=None)
            tree_list.append(tree_dict)
               
        clf1 = tree_list_to_model(tree_list) #normally, you load the dict in this format
        # this is run under the hood
        # clf2 = EnsembleWrapper(clf1)
        
        #%%

        #fit RF here. The hyperparameters are given
        # input either original trained model rf, or dictionary clf1.       
        # EnsembleWrapper is called internally, and will try to adapt the dict into something compatible with Bellatrex


        Btrex_fitted = Bellatrex(rf, SETUP,
                                  p_grid=p_grid,
                                  proj_method=PROJ_METHOD,
                                  dissim_method=STRUCTURE,
                                  feature_represent=FEAT_REPRESENTATION,
                                  n_jobs=JOBS,
                                  verbose=VERBOSE,
                                  show=True,
                                  colormap='RdYlGn_r',
                                  plot_GUI=PLOT_GUI).fit(X_train, y_train)
        
        # store, for every sample in the test set, the predictions from the
        # local method and the original R(S)F
        N = min(X_test.shape[0], MAX_TEST_SIZE)
        
        local_dissimil = np.zeros(N)
        tot_n_splits = 0 # total for the whole fold j        
        stored_info = []
        
        # store final Bellatrex predictions here.
        # It will be a (n_samples, n_outputs_) ndarray, or (n_samples,) for single output (??)
        y_pred = np.empty((0, 0))
        
        # y_pred has different (and inconsistent shape) with different R(S)F models
        # we are harmonising everythin with the next two functions 
        from code_scripts.utilities import predict_helper, concatenate_helper
        
        for i in range(N): #for every sample in the test set: predict
            
            if VERBOSE >= 0:
                print("sample i={} fold j={}," 
                      " data: {}".format(i, j, dataset.split(".csv", 1)[0]))
                  
            # call the .predict method. The hyperparamters were given and
            # and tested within .fit. Now  they are actively used
            
            FILE_OUT = "Rules_"+str(dataset)+"_f"+str(j)+'_id'+str(i)+'.txt'

            sample_info, fig, ax = Btrex_fitted.explain(X_test, i, 
                                    os.path.join(root_folder, 'example-explanations',
                                    FILE_OUT)) #tuning is also done
            
            y_local_pred = sample_info.local_prediction()
            y_pred = concatenate_helper(y_pred, y_local_pred, axis=0)

            # count total number of rulesplits to compute "complexity" as
            # in the paper. Sum over all test samples, average will be taken
            
            for idx in sample_info.final_trees_idx:
                tot_n_splits += count_rule_length(rf, idx, X_test.iloc[i:i+1,:])
                        
            
            ''' input for TreeDissimilarity:
                - original fitted ensemble estimator (not the Bellatrex)
                - indeces of the final extracted trees
                - dissimilarity measure to be used.
            '''
                
            tree_dissim, _ = TreeDissimilarity(rf, SETUP,
                                               sample_info.final_trees_idx,
                                               dissim_method=STRUCTURE,
                                               feature_represent=FEAT_REPRESENTATION,
                                               sample=X_test.iloc[i,:],
                                               ).compute_dissimilarity()
            
            local_dissimil[i] = tree_dissim
            
            if SETUP.lower() in multi_label_key_list + mt_regression_key_list:
                y_true = y_test.iloc[i,:].values
                
            elif SETUP.lower() in survival_key_list:
                y_true = y_test[i]

            else: #binary and (single-target) regression
                y_true = float(y_test[i])
                
            ### binary and survival need a float()
            #consider adding y_oracle prediction
            
            stored_info_iteration = [sample_info.n_trees, str(sample_info.n_dims), 
                                     sample_info.n_clusters,
                                     sample_info.final_trees_idx,
                                     sample_info.cluster_sizes,
                                     y_local_pred,
                                     y_true, # post_processed y_test[i]
                                     sample_info.sample_score, tree_dissim]
                        
            # other attributes are: cluster_sizes, trees (indices)
            stored_info.append(stored_info_iteration) #append to store in csv file
            # CONSIDER SAVING EVERYTHING TO AN OBJECT INSTEAD (pickle)

        # adapt to numpy array (N, L) where N = samples, L = labels (if L>1)
        y_ens_pred = predict_helper(Btrex_fitted.clf, X_test)
                  
        prediction_single_folder = os.path.join(root_folder, "predictions", scenario, dataset)

        if not os.path.exists(prediction_single_folder): #if the dataset name does not exists (e.g. new dataset being used)
            os.makedirs(prediction_single_folder) # then create the new dataset folder
            
        naming_format = NOTES + "fold" + str(j+1) + ".npy" # NOTES = EXTRA_NOTES + PROJ_METHOD + FEAT_REPRESENT
        shorter_naming = "fold" + str(j+1) + ".npy" #for y_labels, no extra info needs to be stored
        
        if SAVE_PREDS == True: # name info: rules or trees, extra notes, proj method, feat_repres., fold. 
            np.save(os.path.join(prediction_single_folder, "local_preds_" + STRUCTURE + "_" + naming_format), y_pred)
            np.save(os.path.join(prediction_single_folder, "RF_preds_" + STRUCTURE + "_" + naming_format), y_ens_pred)
            np.save(os.path.join(prediction_single_folder, "y_true_labels_" + shorter_naming), y_test)
            np.save(os.path.join(prediction_single_folder, "local_dissim_" + STRUCTURE + "_" + naming_format), local_dissimil)
        ## plus, store .txt file with configuration!!!
        
        if j == 0 and N_FOLDS >= 5 and EXTRA_NOTES == "": # not running a draft/trial 
            p_grid["pre-scoring"] = "L2"
            p_grid["FINAL_SCORING"] = FINAL_SCORING            
            config_file = os.path.join(prediction_single_folder,
                                       "config_info.txt")
            
            with open(config_file, 'w', encoding='utf-8') as f:
                #print(p_grid, file=f)
                f.write(str(p_grid))
            
        
        y_test = y_test[:MAX_TEST_SIZE]
        y_ens_pred = y_ens_pred[:MAX_TEST_SIZE]

        
        avg_splits_fold[j] = tot_n_splits/N

        cv_fold_local[j] = list(score_method(y_test, y_pred,SETUP).values())[0]

        cv_fold_ensemble[j] = list(score_method(y_test, y_ens_pred,SETUP).values())[0]

        print("ensemble score: {:.4f}".format(cv_fold_ensemble[j]))
        print("local fold {:.4f}".format(cv_fold_local[j]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cv_local_dissimil[j] = np.nanmean(local_dissimil)
        
        print("data: {}, finished fold j = {}".format(dataset.split(".csv", 1)[0], j))
                
        # storing (fold specific) results and details such as 
        # optimal parameters for Bellatrex

        df_params = pd.DataFrame(stored_info, columns=["n_trees", "n_dims", "n_clusters",
                                                       "trees_idx", "clust_sizes", "pred",
                                                       "y_true", "fidelity_score", "dissim"])
        
        if SETUP.lower() in multi_label_key_list + mt_regression_key_list:
            y_ens_store = [y_ens_pred[i,:] for i in range(len(y_ens_pred))]
            df_params.insert(6, "y_ensemble", y_ens_store)
        else:
            df_params.insert(6, "y_ensemble", y_ens_pred)
        
        #root_folder + "predictions/" + scenario + dataset.split(".csv", 1)[0]
        #params_scenario_folder = os.path.join(params_scenario_folder, scenario)
        params_single_folder = os.path.join(root_folder, "tuned_hyperparams", scenario, dataset)
        
        if not os.path.exists(params_single_folder): #if the dataset folder does not exists (e.g. new dataset being used)
            os.makedirs(params_single_folder) # then create the new folder        

        if SAVE_PREDS == True:
            params_filename = "Params_" + NOTES + STRUCTURE + "_fold" + str(j+1) + ".csv"
            df_params.to_csv(os.path.join(params_single_folder, params_filename))

        j +=1  # next fold, close fold now
        
    t2 = datetime.datetime.now()    # time elapsed for single dataset

    
    # storing average performances in dataframe here:
    performances = pd.DataFrame({"data size" : str(X.shape),
                    "Original R(S)F" : np.round(np.nanmean(cv_fold_ensemble),4),
                    "--": "",
                    "Local perf": np.round(np.nanmean(cv_fold_local),4),
                    "Local dissim": np.round(np.nanmean(cv_local_dissimil),4),
                    "elapsed_time": int((t2- t1).total_seconds()) #in seconds
                    }, 
                     index = [dataset.split(".csv", 1)[0]])
            
    if SETUP.lower() in multi_label_key_list + mt_regression_key_list:
        performances.insert(1, "n labels", orig_n_labels) # add extra info

    
    results_df = pd.concat([results_df, performances], axis=0) # [brackets] necessary if passing dictionary instead  
    print("DONE with: {}".format(dataset).ljust(42))
    
results_df.loc["average"] = results_df.mean(axis=0, numeric_only=True)


if OVERWRITE_DF == True:    
    name_info = "Draft_" if SAVE_PREDS == False else ""
    #NOTES = EXTRA_NOTES + PROJ_METHOD + FEAT_REPRESENT
    name_info = name_info + NOTES + STRUCTURE.capitalize() + "_"
    if FINAL_SCORING.lower() == "cosine":
        name_info = name_info + "cosine_"
    df_filename = name_info + SETUP.capitalize() + str(date.today())[4:] + TAIL_NOTES + ".csv"
    results_df.to_csv(os.path.join(root_folder, df_filename))

# %%
