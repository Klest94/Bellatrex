# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:22:59 2022

@author: u0135479
"""
import numpy as np
import os
import pandas as pd
import warnings

from utilities import score_method
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.ensemble import RandomForestClassifier
from rulecosi import RuleCOSIClassifier

SETUP = "bin"
N_FOLDS = 5
TAIL_NOTES = ""
from utilities import get_data_list
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

##########################################################################
#root_folder = "C:/Users/u0135479/Documents/GitHub/Explain-extract-trees/"
root_folder = os.getcwd()

dnames, data_scenario_folder, scenario = get_data_list(SETUP, root_folder) #scenario has extra "/"

testing_dnames = dnames#[:3]

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

df_filename = "RuleCOSI_" + SETUP.capitalize() + TAIL_NOTES + ".csv"


def calculate_intervals(full_rule):
    
    if not isinstance(full_rule, str):
        full_rule = full_rule.str
    splits = full_rule.split(chr(708))

    conditions_dict = {} # dict with all conditions of the rule
    
    for splitted in splits:
        
        '''calculate intervals '''
        
        if splitted.count(chr(62)) == 1: # if we have a ">"
            col_name = splitted.split(chr(62))[0] # strip first char
            col_name = col_name[col_name.find('(')+1:]
            col_name = col_name[:col_name.find(')')]
    
            col_value = splitted.split(chr(62))[1] 
            # strip outside brackets
            col_value = col_value[:col_value.find(')')]
            interval = [float(col_value), np.inf]
            conditions_dict[str(col_name)] = interval
            
        elif splitted.count(chr(8804)) == 1:
            col_name = splitted.split(chr(8804))[0] # strip first char
            col_name = col_name[col_name.find('(')+1:]
            col_name = col_name[:col_name.find(')')]
            
            col_value = splitted.split(chr(8804))[1][1:] # strip space char
            # strip brackets
            col_value = col_value[:col_value.find(')')]
            interval = [-np.inf, float(col_value)]
            conditions_dict[str(col_name)] = interval

        elif splitted.count(chr(8804)) == 0 and splitted.count(chr(62)) == 0:
            pass # empty dict remains
        else:
            raise ValueError("Something went wrong")
            
    return conditions_dict
    

def conditions_satisfied(sample, conditions):
    # sample: pd.Series, # conditions: dict
    check = True
    for col in sample.index:
        if col in conditions.keys():
            update = sample[col] < conditions[col][1] and sample[col] > conditions[col][0]
            check = check and update
    
    return check



for folder in testing_dnames:
    
    cv_fold_ruleCosi = np.zeros(N_FOLDS)
    
    cv_ruleCosi_length = np.zeros(N_FOLDS)
    cv_ruleCosi_num = np.zeros(N_FOLDS)
    
    cv_reading_complexity = np.zeros(N_FOLDS)
    cv_num_rules_read = np.zeros(N_FOLDS)
    
    j = 0  #outer fold index, (we are performing the outer CV "manually")
        
    while j < N_FOLDS: #old: folder.split(".csv")[0]
        data_single_folder = os.path.join(data_scenario_folder, folder)
        X_train = pd.read_csv(os.path.join(data_single_folder, "X_new_train"+ str(j+1) + ".csv"))
        y_train = pd.read_csv(os.path.join(data_single_folder, "y_new_train"+ str(j+1) + ".csv"))
        X_test = pd.read_csv(os.path.join(data_single_folder, "X_new_test"+ str(j+1) + ".csv"))
        y_test = pd.read_csv(os.path.join(data_single_folder, "y_new_test"+ str(j+1) + ".csv"))
        
        col_names = list(X_train.columns)
        if col_names[:2] == ["0", "1"]:  # if column names are ugly
            col_names = ["X"+str(i) for i in X_train.columns]
            
        X_train.columns = col_names
        X_test.columns = col_names
        
        X = pd.concat([X_train, X_test], axis=0)
        
        X_train = X_train#.values
        y_train = y_train#.values
        X_test = X_test#.values
        y_test = y_test#.values
        
        if SETUP.lower() in binary_key_list + multi_label_key_list:
            ens = RandomForestClassifier(n_estimators=100, min_samples_split=5,
                                     n_jobs=-1, random_state=0)
            
        else:
            raise KeyError("Not yet")
            
        rc = RuleCOSIClassifier(base_ensemble=ens, 
                        metric='roc_auc', tree_max_depth=5, 
                        conf_threshold=0.9, cov_threshold=0.0,
                        random_state=0, column_names=X_train.columns)
        
        rc.fit(X_train, y_train)
        
        rc.simplified_ruleset_.print_rules(heuristics_digits=4, condition_digits=1)

        if SETUP.lower() in binary_key_list:
            y_pred = rc.predict_proba(X_test)[:,1]
        elif SETUP.lower() in multi_label_key_list:
            y_pred0 = [y0[:, 1] for y0 in ens.predict_proba(X_test)]
            y_pred = np.transpose(np.array(y_pred0))
        
        
        # global measures (printed explanations):
        rules = rc.simplified_ruleset_.rules # list of Rule object
        for rule in rules: # count occcurences of conjunctions "˄" for each rule
            cv_ruleCosi_length[j] += rule.str.count(chr(708)) + 1
        
        # local measure ( sample dependent). Loop along X_test:
        #repeat per fold ( index j)
        reading_complexity = 0
        tot_rules_read = 0
        
        for i in range(X_test.shape[0]):
            
            X_test_i = X_test.iloc[i,:]
            
            reading_complex_i = reading_complexity
            
            #reading = True
            #while reading == True:
            # OLD SET-UP: reading until rule is found
            # for rule in rules:
            #     rule_as_dict = calculate_intervals(rule)
            #     reading_complexity += rule.str.count(chr(708)) + 1
            #     tot_rules_read += 1
            #     if conditions_satisfied(X_test_i, rule_as_dict):
            #         break
                # all rules have been read here, exit anyway
                
            # NEW SET-UP: read and count only the verified rule
            for rule in rules[:-1]:
                rule_as_dict = calculate_intervals(rule)
                if conditions_satisfied(X_test_i, rule_as_dict):
                    tot_rules_read += 1
                    reading_complexity += rule.str.count(chr(708)) + 1
                    break
                     
            if reading_complexity == reading_complex_i: #then no rule was activated (only the last one remains)
                for rule in rules:
                    tot_rules_read += 1
                    rule_as_dict = calculate_intervals(rule)
                    reading_complexity += rule.str.count(chr(708)) + 1                
            
            # continue per every sample, sum them up

            
        #take average across fold
        cv_reading_complexity[j] = reading_complexity/(X_test.shape[0])
        cv_num_rules_read[j] = tot_rules_read/(X_test.shape[0])                
        cv_ruleCosi_num[j] =  len(rules)

    
        cv_fold_ruleCosi[j] = score_method(y_test, y_pred, #list or array, right?
                                        SETUP)
        
        ### TODO would be nice to compare dissimilarity of the final rules
                
        j += 1
    
    print('total rulelength:', cv_ruleCosi_length)
    print('finished with:', folder)
    print("==========================")
    
    performs = pd.DataFrame({
                    "data size" : str(X.shape),
                    "RuleCOSI" : np.round(np.nanmean(cv_fold_ruleCosi),4),
                    "all ruleslength": np.round(np.nanmean(cv_ruleCosi_length), 4),
                    "n. rules": np.round(np.nanmean(cv_ruleCosi_num), 4),
                    "ruleslength read": np.round(np.nanmean(cv_reading_complexity), 4),
                    "n. rules read": np.round(np.nanmean(cv_num_rules_read), 4)
                    },
                    index = [folder.split(".csv", 1)[0]])
        
    results_df = pd.concat([results_df, performs], axis=0) # [brackets] necessary if passing dictionary instead  
    print("DONE with: {}".format(folder).ljust(42))
    
results_df.loc["average"] = results_df.mean(axis=0, numeric_only=True)
        
results_df.to_csv(os.path.join(root_folder, df_filename))

#%%
# import joblib
# import numpy as np
# import pandas as pd

# chr708 is the "et"
# chr 62 is the ">"
# chr 8804 is the "<="

# X_test_i = X_test.iloc[1,:]
# rules = joblib.load("./ruleCOSI-load.joblib")

# rule1 = rules[0].str

# reading_complexity = 0
# tot_rules_read = 0
# for rule in rules:
#     rule_as_dict = calculate_intervals(rule)
#     if not conditions_satisfied(X_test_i, rule_as_dict):
#         reading_complexity += rule.str.count(chr(708)) + 1
#         tot_rules_read += 1
    

    
    
    
