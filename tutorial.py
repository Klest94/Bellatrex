"""
@author:       Klest Dedja
@institution:  KU Leuven
"""
import os
os.environ["OMP_NUM_THREADS"] = "1" # avoids memory leak caused by K-Means
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sksurv.ensemble import RandomSurvivalForest

from code_scripts.utilities import score_method, output_X_y
from code_scripts.utilities import format_targets, format_RF_preds

from code_scripts.LocalMethod_class import Bellatrex


# reduce MAX_TEST_SIZE for quick code testing
MAX_TEST_SIZE = 3 #if set >= 100, it takes the (original) value X_test.shape[0]

# the p_grid used for tuning during the predict phase.
# n_trees refers to the number of preselected trees ( and corresponding rules)
# this step is followed by the the rule vectorisation step
# n_dims refers to the number of dimensions after PCA , performed after the vectorization step
# n_clusters refers to the number of clusters in whoch to group the projected
# for each cluster, one representative rule is extracted as a final explanation.
p_grid = {
    "n_trees": [20, 50, 80],
    "n_dims": [2, 5, None],
    "n_clusters": [1, 2, 3]
    }

##########################################################################
root_folder = os.getcwd()

data_folder = os.path.join(root_folder, "datasets")

'''
choose appropriate learning task wth SETUP parameter
for this tutorial, choose between
- 'bin' for explaining binary classification prediction
performed by a RandomForestClassifier
- 'surv' for explaining time-to-event predictions
made by a RandomSurvivalForest
- 'mtr' for explainiing multi-outoput regression prediction
performed by a RandomForestRegressor
'''

SETUP = "bin"
VERBOSE = 3
PLOT_GUI = True

'''  levels of verbosity in this script:
    - >= 1.0: print best params, their achieved fidelity,
              and the scoring method used to compute such performance
    - >= 2.0 print final tree idx cluster sizes

    GRAPHICAL EXPLANATIONS START FROM HERE:

    - >= 3.0: plot representation of the extracted trees (two plots)
    - >= 4.0 plot trees without GUI (if PLOT_GUI == False)
    - >= 4.0 plot trees with GUI (if PLOT_GUI == True)
    - >= 5.0: print params and performance during GridSearch
'''


#%%
df_train = pd.read_csv(os.path.join(data_folder, SETUP + '_tutorial_train.csv'))
df_test = pd.read_csv(os.path.join(data_folder, SETUP + '_tutorial_test.csv'))

X_train, y_train = output_X_y(df_train, SETUP)
X_test, y_test = output_X_y(df_test, SETUP)

# for quick testing, set a small MAX_TEST_SIZE
X_test = X_test[:MAX_TEST_SIZE]
y_test = y_test[:MAX_TEST_SIZE]

orig_n_labels = y_test.shape[1] #meaningful only in multi-output


y_train, y_test = format_targets(y_train, y_test, SETUP, VERBOSE)


### instantiate original R(S)F estimator
if SETUP.lower() in ['surv']:
    rf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
                              random_state=0)

elif SETUP.lower() in ['bin','mlc']:
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=5,
                                random_state=0)

elif SETUP.lower() in ['regress', 'mtr']:
    rf = RandomForestRegressor(n_estimators=100, min_samples_split=5,
                               random_state=0)

#%%

#fit RF here. The hyperparameters are given

Bellatrex_fitted = Bellatrex(rf, SETUP,
                            p_grid=p_grid,
                            proj_method="PCA",
                            dissim_method="rules",
                            feature_represent="weighted",
                            n_jobs=1,
                            verbose=VERBOSE,
                            plot_GUI=PLOT_GUI,
                            plot_max_depth=5,
                            dpi_figure=100).fit(X_train, y_train)


# store, for every sample in the test set, the Bellatrex predictions
N = min(X_test.shape[0], MAX_TEST_SIZE)
y_pred = []

for i in range(N): #for every sample in the test set: call .predict
    # call the .predict method. The hyperparamters were given in the .fit.
    # and are now tuned for every instance, the .predict method ouputs:
    #     1) the local prediction
    #     2) info about the Bellatrex instance: optimal parameters,
    #         final extracted trees/rules, their weight in the prediction etc.

    y_pred_i, _ = Bellatrex_fitted.predict(X_test, i)  #tuning is also done

    y_pred.append(y_pred_i)

y_ens_pred = format_RF_preds(rf, X_test, SETUP)

# adapt to numpy array (N, L) where N = samples, L = labels (if L>1)
#y_ens_pred = np.transpose(np.array(y_ens_pred)[:,:,1])
if SETUP.lower() not in ['mlc', 'mtr']:
    y_pred = np.array(y_pred).ravel() #force same array format

# make sure y_pred is a (N,l)-shaped array (and not a list of arrays)
y_pred = np.array(y_pred)


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

if SETUP.lower() in ['mlc', 'mtr']:
    performances.insert(1, "n labels", orig_n_labels) # add extra info

print(performances)
