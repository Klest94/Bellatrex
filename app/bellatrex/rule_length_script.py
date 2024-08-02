import numpy as np
import os
import pandas as pd
import datetime
import warnings
import re
warnings.filterwarnings("ignore", category=UserWarning)
from datetime import date
today = date.today()
print("Today's date:", today)
from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

#from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from utilities import get_data_list
from TreeDissimilarity_class import TreeDissimilarity
# from LocalMethod_class import LocalTreeExtractor
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

SETUP = "surv"
PROJ_METHOD = "PCA"
# "noDims_"
# Ablation study so far only with Jaccard_trees + PCA (simplest set-up)

DRAFT = False # if False: save: predictions, tuned_hyperparams and dataframe
OVERWRITE_DF = True # save performance df as csv, potentially overwriting smth
MAX_SAMPLES = 999 #if set > 100, it takes the (original) value X_test.shape[0]

STRUCTURE = "rules" # accepted keys: "trees" or "rules"
FEAT_REPRESENTATION = "by_samples" # "simple" or "by_samples" # or
FINAL_SCORING = "L2" # L2 or cosine

##########################################################################
root_folder = "C:/Users/u0135479/Documents/GitHub/Explain-extract-trees/"
root_folder = os.getcwd()

dnames, data_scenario_folder, scenario = get_data_list(SETUP, root_folder) #scenario has extra "/"
#hyperparams_scenario_folder =  os.path.join(root_folder, "datasets", scenario[:-1])
# <-- add to name of stored y-s and y_preds ??
EXTRA_NOTES = ""  # noDims_ or noTrees_ or noBoth_
TAIL_NOTES = "" #ONLY FOR FINAL CSV FILE
# think it through...
NOTES = EXTRA_NOTES + PROJ_METHOD + "_" + FEAT_REPRESENTATION + "_"

testing_dnames = dnames#[:3]
#testing_dnames = [dnames[i] for i in [4, 6, 8, 10, 11]]
##########################################################################
#%%

p_grid = {
    "trees": [0.2, 0.5, 0.8], # [100] for "noTrees_" ablation
    "dims": [2, 5, None], #None = no dim reduction   #Ablation: noDims_
    "clusters": [1, 2, 3]
    }

N_FOLDS = 5

# N JOBS NOT WORKING CORRECTLY!!!!
JOBS = 1 # n_jobs for R(S)F learner (and now also ETRees candidate choice!)
VERBOSE = 1
'''         ------  levels of verbosity in this script:
- from 0.0: sample i and fold j print, ensemble and local score.
- from 1.0: print best params, their achieved performance,
          and the scoring method used to compute such performance
- from 2.0: plot info on the extracted trees (two plots)
- from ?.?: trees idx per hyperparam setting
            oracle sample is None or not?
    NOTE: might be innacurate when n_jobs > 1
- from 3.0: print params and performance during GridSearch
- from 4.0: print single local prediction, and R(S)F prediction  (? kinda...)
- from 5.0: y_local and y_pred prediction shape (debugging)
--------------------'''


#probelms with regression (maybe in get_data_list ?)
dnames = list(filter(lambda f: not f.endswith('.csv'), dnames)) # throws away csv-s

binary_key_list = ["bin", "binary"]
survival_key_list = ["surv", "survival"]
multi_label_key_list = ["multi", "multi-l", "multi-label", "mtc"]
regression_key_list = ["regression", "regress", "regr"]
mt_regression_key_list = ["multi-target", "multi-t", "mtr"]
#%%


print("running on datasets:")
for data in testing_dnames:
    print(data)

results_df = pd.DataFrame() # empty datafarme to start with. Append here


#for PROJ_METHOD in PROJ_METHODS:
for folder in testing_dnames:

    ######  store fold performance, and prepare data in the right format
    avg_splits_fold = np.zeros(N_FOLDS)*np.nan

    j = 0  #outer fold index, (we are performing the outer CV "manually")

    while j < 5: #old: folder.split(".csv")[0]
        data_single_folder = os.path.join(data_scenario_folder, folder)
        X_train = pd.read_csv(os.path.join(data_single_folder, "X_new_train"+ str(j+1) + ".csv"))
        y_train = pd.read_csv(os.path.join(data_single_folder, "y_new_train"+ str(j+1) + ".csv"))
        X_test = pd.read_csv(os.path.join(data_single_folder, "X_new_test"+ str(j+1) + ".csv"))
        y_test = pd.read_csv(os.path.join(data_single_folder, "y_new_test"+ str(j+1) + ".csv"))


        if SETUP.lower() in binary_key_list + regression_key_list:
            y_train = y_train.values # why this? needed to .fit the train labels?
            y_test = y_test.values

        X = pd.concat([X_train, X_test], axis=0, ignore_index=False)
        assert X.isnull().sum().sum() < 1

        if SETUP.lower() in multi_label_key_list + mt_regression_key_list:
            orig_n_labels = y_test.shape[1]
            y_train.columns = y_test.columns # otherwise, sometimes is a bug
            for col in y_test.columns:
                if len(y_test[col].unique()) == 1 or len(y_train[col].unique()) == 1:
                    y_test.drop(col, inplace=True, axis=1)
                    y_train.drop(col, inplace=True, axis=1)

            n_labels = y_test.shape[1]
            print("new n* labels:", n_labels)

        if SETUP.lower() in binary_key_list + regression_key_list:
            y_train = y_train.ravel()
            y_test = y_test.ravel()

        if  SETUP.lower() in survival_key_list:
            y_train = y_train.to_records(index=False)
            y_test = y_test.to_records(index=False)


        ### instantiate original R(S)F estimator
        if SETUP.lower() in survival_key_list:
            rf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
                                        n_jobs=JOBS, random_state=0)

        elif SETUP.lower() in binary_key_list + multi_label_key_list:
            rf = RandomForestClassifier(n_estimators=100, min_samples_split=5,
                                        n_jobs=JOBS, random_state=0)

        elif SETUP.lower() in regression_key_list + mt_regression_key_list:
            rf = RandomForestRegressor(n_estimators=100, min_samples_split=5,
                                        n_jobs=JOBS, random_state=0)

        rf.fit(X_train, y_train)

        N = min(X_test.shape[0], MAX_SAMPLES)
        #n_splits = np.zeros(N)

        params_scenario_folder = os.path.join(root_folder, "tuned_hyperparams", scenario, folder)
        params_filename = "Params_" + NOTES + STRUCTURE + "_fold" + str(j+1) + ".csv"

        df_trees = pd.read_csv(os.path.join(params_scenario_folder, params_filename))["trees_idx"]

        from TreeRepresentation import count_rule_length

        tot_n_splits = 0 # total for the whole fold j
        for i in range(N):
            # reformat the output, form a string to a proper list
            tree_idx = list(map(int, re.findall(r'\d+', df_trees[i])))

            for idx in tree_idx:
                tot_n_splits += count_rule_length(rf, idx, X_test.iloc[i:i+1,:])

            #n_splits[i] = tot_n_splits

        print('data: {}, fold {:d}, tot n splits: {:d}'.format(folder, j, tot_n_splits))
        avg_splits_fold[j] = tot_n_splits/N
        j += 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        #storing average performances in dataframe here:
        performances = pd.DataFrame({"data size" : str(X.shape),
                        "Avg. rule length" : np.round(np.nanmean(avg_splits_fold),4),
                        },
                         index = [folder.split(".csv", 1)[0]])

    results_df = pd.concat([results_df, performances], axis=0) # [brackets] necessary if passing dictionary instead
    print("DONE with: {}".format(folder).ljust(42))

results_df.loc["average"] = results_df.mean(axis=0, numeric_only=True)


if OVERWRITE_DF == True:
    name_info = "Draft_" if DRAFT == True else ""
    name_info = name_info + NOTES + STRUCTURE.capitalize() + "_"
    df_filename = name_info + "l_rules_" + SETUP.capitalize() + str(today)[4:]
    df_filename = df_filename + TAIL_NOTES + ".csv"
    results_df.to_csv(os.path.join(root_folder, df_filename))
