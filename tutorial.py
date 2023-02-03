import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1" # avoids memory leak UserWarning caused by KMeans
import pandas as pd

from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from utilities import score_method, output_X_y
from utilities import format_targets, format_RF_preds
#from plot_tree_patch import plot_tree_patched

from LocalMethod_class import Bellatrex

# reduce MAX_TEST_SIZE for quick code testing
MAX_TEST_SIZE = 10 #if set >= 100, it takes the (original) value X_test.shape[0]

p_grid = {
    "n_trees": [0.2, 0.5, 0.8], # [100] for "noTrees_" ablation
    "n_dims": [2, 5, None], #None = no dim reduction   #Ablation: noDims_
    "n_clusters": [1, 2, 3]
    }

##########################################################################
root_folder = os.getcwd()

data_folder = os.path.join(root_folder, "datasets")


SETUP = "mtr" # "bin", or "mtr", 

# running different RFs or different performance measures according to the 
# prediction scenarios. So far we have implemented the following 5 cases:
binary_key_list = ["bin", "binary"]
survival_key_list = ["surv", "survival"]
multi_label_key_list = ["multi", "multi-l", "multi-label", "mtc"]
regression_key_list = ["regression", "regress", "regr"]
mt_regression_key_list = ["multi-target", "multi-t", "mtr"]

##########################################################################



#%%
VERBOSE = 3
PLOT_GUI = False
'''  levels of verbosity in this script:
    - >= 1.0: print best params, their achieved fidelity,
              and the scoring method used to compute such performance
    - >= 2.0 print final tree idx cluster sizes
    - >= 3.0: plot representation of the extracted trees (two plots)
    - >= 4.0 plot trees with GUI (if PLOT_GUI == True)
    - >= 4.0 plot trees without GUI (if PLOT_GUI == False)
    - >= 5.0: print params and performance during GridSearch
'''

 
df_train = pd.read_csv(os.path.join(data_folder, SETUP + '_tutorial_train.csv'))
df_test = pd.read_csv(os.path.join(data_folder, SETUP + '_tutorial_test.csv'))

X_train, y_train = output_X_y(df_train, SETUP)
X_test, y_test = output_X_y(df_test, SETUP)

X_train = X_train.drop("Unnamed: 0", axis=1, errors="ignore", inplace=False)
X_test = X_test.drop("Unnamed: 0", axis=1, errors="ignore", inplace=False)

assert X_train.isnull().sum().sum() < 1 #make sure there are no null values
assert X_test.isnull().sum().sum() < 1 #make sure there are no null values

# for quick testing, set a small MAX_TEST_SIZE
X_test = X_test[:MAX_TEST_SIZE]
y_test = y_test[:MAX_TEST_SIZE]

orig_n_labels = y_test.shape[1] #meaningful only in multi-output

#%%
# set target variable to correct format depending on the prediciton
# scenarios.E.g. set np.recarray fo survival data, or normalise data 
# in case of single and multi-target regression

y_train, y_test = format_targets(y_train, y_test, SETUP, VERBOSE)


### instantiate original R(S)F estimator
if SETUP.lower() in survival_key_list:
    rf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
                              random_state=0)

elif SETUP.lower() in binary_key_list + multi_label_key_list:
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=5,
                                random_state=0)
    
elif SETUP.lower() in regression_key_list + mt_regression_key_list:
    rf = RandomForestRegressor(n_estimators=100, min_samples_split=5,
                               random_state=0)


#fit RF here. The hyperparameters are given      
Bellatrex_fitted = Bellatrex(rf, SETUP,
                            p_grid=p_grid,
                            proj_method="PCA",
                            dissim_method="rules",
                            feature_represent="weighted",
                            n_jobs=1,
                            verbose=VERBOSE,
                            plot_GUI=PLOT_GUI).fit(X_train, y_train)


# store, for every sample in the test set, the predictions from the
# local method and the original R(S)F
N = min(X_test.shape[0], MAX_TEST_SIZE)        
y_pred = []

stored_info = [] #store extra info such as optimal hyperparameters (for each instance)

for i in range(N): #for every sample in the test set: call .predict
          
    # call the .predict method. The hyperparamters were given in the .fit.
    # Now they are actively used and tuned for every instance
    '''
    the .predict ouputs:
        1) the local prediction 
        2) info about the Bellatrex instance: optimal parameters,
                    final extracted trees/rules, their weight in the prediction, etc... 
    
    '''
    y_local_pred, sample_info = Bellatrex_fitted.predict(X_test, i)  #tuning is also done
    
    # append all test sample predictions in y_pred
    y_pred.append(y_local_pred) # store for debuggind and analysis
    
        
y_ens_pred = format_RF_preds(rf, X_test, SETUP)

# adapt to numpy array (N, L) where N = samples, L = labels (if L>1)
#y_ens_pred = np.transpose(np.array(y_ens_pred)[:,:,1])
if SETUP.lower() not in multi_label_key_list + mt_regression_key_list :
    y_pred = np.array(y_pred).ravel() #force same array format
          

y_pred = np.array(y_pred)
#now it's not weird anymore and is just a (N,l)-shaped array

#in case of quick testing with few samples (less than 100)
y_test = y_test[:MAX_TEST_SIZE]
y_ens_pred = y_ens_pred[:MAX_TEST_SIZE]



#### performance results here ####


cv_fold_local = score_method(y_test, y_pred, #list or array, right?
                                SETUP)     #workaround for now

cv_fold_ensemble = score_method(y_test, y_ens_pred, #list or array, right?
                                SETUP)


performances = pd.DataFrame({
                "Original R(S)F" : np.round(np.nanmean(cv_fold_ensemble),4),
                "Bellatrex": np.round(np.nanmean(cv_fold_local),4),
                }, index=[0] 
                )
        
if SETUP.lower() in multi_label_key_list + mt_regression_key_list:
    performances.insert(1, "n labels", orig_n_labels) # add extra info

print(performances)

