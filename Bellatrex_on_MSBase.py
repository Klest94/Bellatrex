import numpy as np
import os
import pandas as pd
import datetime

from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
#%matplotlib inline
from LocalMethod_class import LocalTreeExtractor 

root_folder = os.path.dirname(os.path.realpath('__file__'))
os.chdir(root_folder)


FOLDER_NAME = "MS_BASE_static"

SETUP = "binary" # "binary" or "survival"
PROJ_METHOD = "PCA"
EXTRA_NOTES = "" # <-- add to name of stored y-s and y_preds
# "noDims_"
JOBS = 1
NOTES = EXTRA_NOTES + PROJ_METHOD + "_"

DRAFT = True # if False: save: predictions, tuned_hyperparams and dataframe,
OVERWRITE_DF = False # overwrite performance df and compute results
PREDS = True  #run all predictions?

PRE_SCORING = "L2"
FINAL_SCORING = "L2" # L2 or cosine
test_idx_to_explain = [0, 3] # samples to consider per test fold (normally: N = X_test.shape[0])
#test_idx_to_explain = np.arange(25)

#%%
##########################################################################


p_grid = {
    "trees": [20, 50, 80],
    "dims": [2, 5, None], #None = no dim reduction
    "clusters": [1, 2, 3, 4],
    }

VERBOSE = 2.5

'''         ------ GUIDE TO THE VERBOSITY LEVELS in this script:
- from 0.0: sample index i print. Store .txt-s with rulepath and leaf intervals
- from 1.0: print best params, their achieved performance,
          and the scoring method used to compute such performance
- from 2.0: plot info on the extracted trees (two plots)
- from 2.5: plot final extracted trees, in full
- from ?.?: trees idx per hyperparam setting
            oracle sample is None or not?
    NOTE: might be innacurate when n_jobs > 1
- from 3.0: print params and performance during GridSearch
- from 4.0: print single local prediction, and R(S)F prediction  (? kinda...)
- from 5.0: y_local and y_pred prediction shape (debugging)
--------------------'''

binary_key_list = ["bin", "binary"]
survival_key_list = ["surv", "survival"]
multi_label_key_list = ["multi", "multi-l", "multi-label"]
regression_key_list = ["regression", "regress", "regr"]

### import the example dataset

cols_drop = ["Unnamed: 0", "clinic_i", "country_i", "PATIENT_ID"] #slice_id
data_train0 = pd.read_csv(os.path.join(root_folder, FOLDER_NAME, SETUP, "MS_train.csv")) 
data_train = data_train0.drop(cols_drop, axis=1, errors="ignore")

data_test0 = pd.read_csv(os.path.join(root_folder, FOLDER_NAME, SETUP, "MS_test.csv"))
data_test = data_test0.drop([col for col in cols_drop if col in cols_drop], axis=1, errors="ignore")



###### prepare data in the right format (TO BE IMPROVED)

t1 = datetime.datetime.now()

if SETUP.lower() not in survival_key_list:
    # float64 --> float32 ??

    #X_cols = data_train.columns[:-1]
    X_cols = data_train.drop("label", axis=1).columns
    X_train, y_train = data_train.drop("label", axis=1), data_train["label"]
    X_test, y_test = data_test.drop("label", axis=1),  data_test["label"]

else:
    X_cols = data_train.drop(["event_time", "event_status"], axis=1).columns
    X_train  = data_train.drop(["event_time", "event_status"], axis=1)
    y_train = data_train[["event_time", "event_status"]]
    X_test = data_test.drop(["event_time", "event_status"], axis=1)
    y_test = data_test[["event_time", "event_status"]]
    
    
# fixing column names (either original ones, or name them "X0", "X1" etc. etc.)
if X_cols[0] == str(0) and X_cols[-1] == str(len(X_cols)-1):
    X_cols = ["X"+ col for col in X_cols]        


X = pd.concat([X_train, X_test], axis=0, ignore_index=False)
X.columns = X_cols[:X.shape[1]]

if SETUP.lower()  in survival_key_list:
    # use float32 instead of float64 (compatibility with scikit-survival)
    X_train = pd.DataFrame.astype(X_train, np.float32)
    X_test = pd.DataFrame.astype(X_test, np.float32)
    
    '''   further post processing:
    - event_type == 2 indicates already observed event --> drop these
    - event_time == 0 is the last visit, drop them
    - consider dropping all visits after the last observed event, except for the
    first visit of such series (gives repeated information, with smaller and 
                                smaller time to censoring)
    
    '''
    X_train.loc[X_train["event_stats"] == 2]["event_stats"] = 1
    X_test.loc[X_test["event_stats"] == 2]["event_stats"] = 1
    
    #X_train = X_train.loc[(y_train["event_status"] != 2) & ((y_train["event_status"] == 1) | (y_train["event_time"] > 1))]
    #y_train = y_train.loc[(y_train["event_status"] != 2) & ((y_train["event_status"] == 1) | (y_train["event_time"] > 1))]
    #X_test = X_test.loc[(y_test["event_status"] != 2) & ((y_test["event_status"] == 1) | (y_test["event_time"] > 1))]
    #y_test = y_test.loc[(y_test["event_status"] != 2) & ((y_test["event_status"] == 1) | (y_test["event_time"] > 1))]

    print("censoring rate: {:.3f}".format(1 - y_test["event_status"].mean())) # VERY HIGH CENSORING, DOUBLE CHECK!
    
    


# check no missing data:
assert X_train.isnull().sum().sum() + X_test.isnull().sum().sum() < 1

## drop unexisting labels in y_train or y_test
if SETUP.lower() in multi_label_key_list:
    orig_n_labels = y_test.shape[1]
    #y_train = pd.DataFrame(y_train, columns=y_test.columns) # they got lost
    #y_test = pd.DataFrame(y_test)
    ### drop labels with no positive (or negative) class! (train or test)
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
    y_train = y_train.to_records(index=False,
                column_dtypes={"event_status": "<?", "event_time": "float32"})
    y_test = y_test.to_records(index=False,
                column_dtypes={"event_status": "<?", "event_time": "float32"})
    
    # swap order??
    



### instantiate original R(S)F estimator
if SETUP.lower() in survival_key_list:
    rf = RandomSurvivalForest(min_samples_split=20, min_samples_leaf=10,
                                n_jobs=-3, max_depth=20, random_state=0)

elif SETUP.lower() in binary_key_list + multi_label_key_list:
    rf = RandomForestClassifier(min_samples_split=10, n_jobs=-3,
                                max_depth=20, class_weight=None,
                                random_state=0)
    
elif SETUP.lower() in regression_key_list:
    rf = RandomForestRegressor(min_samples_split=10, n_jobs=-3,
                               max_depth=10,
                               random_state=0)

rf_params = rf.get_params()
#fit RF here. The hyperparameters are given
LTreeX = LocalTreeExtractor(rf, SETUP, n_trees = p_grid["trees"], 
                  n_dims = p_grid["dims"],
                  n_clusters = p_grid["clusters"],
                  proj_method=PROJ_METHOD,
                  dissim_method="Jaccard_trees",
                  pre_select_trees=PRE_SCORING,
                  scoring=FINAL_SCORING,
                  n_jobs=JOBS,
                  verbose=VERBOSE)

LTreeX_fitted = LTreeX.fit(X_train, y_train)

# store, for every sample in the test set, the predictions from the
# local method and the original R(S)F
y_local_pred = []

stored_info = []


for i in test_idx_to_explain : #for every sample in the test set: predict
    if VERBOSE >= 0:
        print("\nsample i={}".format(i))
    # call the .predict method. The hyperparamters were given already
    # with .fit, but will be used only now

    #new output: (binary set-up)
    #float, array (1,), TreeExtraction object with:
            #- final_trees_idx
            #- local_prediction
            # [sample_info.n_trees, str(sample_info.n_dims),
            #                   sample_info.n_clusters,
            #                   y_local_pred, y_true,
            #                   sample_score, tree_dissim]
    sample_score, local_pred, \
    sample_info = LTreeX_fitted.predict(X_test, i)  #tuning is also done
        
    #y_local_pred.append(local_pred) # store for debuggind and analysis

    #assert = local pred is
    
    if SETUP.lower() in multi_label_key_list:
        y_true = y_test.iloc[i,:].values
        y_local_pred.append(sample_info.local_predict())
        
    elif SETUP.lower() in survival_key_list:
        y_true = y_test[i]
        y_local_pred.append(sample_info.local_predict())

    else: # binary or regression
        y_true = float(y_test[i])
        y_local_pred.append(sample_info.local_predict())
        
    ### binary and survival need a float()
    #consider adding y_oracle prediction

    stored_info_iteration = [sample_info.n_trees, str(sample_info.n_dims), 
                              sample_info.n_clusters,
                              sample_info.final_trees_idx,
                              sample_info.cluster_sizes,
                              sample_info.local_predict(),
                              y_true, # post_processed y_test[i]
                              sample_score] # dropped "tree_dissim" 
                
    # other attributres are: cluster_sizes, trees (indeces)
    stored_info.append(stored_info_iteration) #append to store in csv file
    
    # CONSIDER SAVING EVERYTHING TO AN OBJECT INSTEAD (pickle)
    
    prints_single_folder = os.path.join(root_folder, FOLDER_NAME, SETUP)
    if not os.path.exists(prints_single_folder): #if the data folder does not exists (e.g. new dataset being used)
        os.makedirs(prints_single_folder) # then create the new folder


    sample_info.print_rules(prints_single_folder, i, y_test[i])


    ### close loop around samples here ###
    
# storing predictions for performance evaluation:
if SETUP.lower() in survival_key_list + regression_key_list:
    y_ens_pred = rf.predict(X_test)
    
elif SETUP.lower() in binary_key_list: 
    y_ens_pred = rf.predict_proba(X_test)[:,1]
    
elif SETUP.lower() in multi_label_key_list:
    y_ens_pred = rf.predict_proba(X_test)
    y_ens_pred = np.transpose(np.array(y_ens_pred)[:,:,1])

# adapt to numpy array (N, L) where N = samples, L = labels (if L>1)
#y_ens_pred = np.transpose(np.array(y_ens_pred)[:,:,1])
if SETUP.lower() not in multi_label_key_list:
    y_local_pred = np.array(y_local_pred).ravel() #force same array format
  
  
t2 = datetime.datetime.now()    # time elapsed for single dataset

print("DONE. Elapsed time: {} seconds".format(int((t2- t1).total_seconds())))
print("samples explained:", len(test_idx_to_explain))

#%%
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import CalibrationDisplay

#rf_calibr.fit(X_train, y_train)
rf_calibr = CalibratedClassifierCV(base_estimator=rf,
                                        cv="prefit")
rf_calibr.fit(X_test, y_test)

# LTreeX_calibr = CalibratedClassifierCV(base_estimator=LTreeX_fitted,
#                                         cv="prefit")

# LTreeX_calibr.fit(X_test, y_test)



# compare this with uy_ens_pred (original rf preds)
y_ens_pred_calib = rf_calibr.predict_proba(X_test)[:,1]
# y_local_pred_calib = LTreeX_calibr.predict_proba(X_test)[:,1]

print("let's try calibration")
print(" ##### RF original ##### ")


'''what is ahppening here, exactly? Study CalibrationClassifierCV carefully '''

#plt.figure(figsize=(5,4))
disp = CalibrationDisplay.from_estimator(rf, X_test, y_test, n_bins=20)
disp.figure_.figure.dpi = 80
plt.title("Test set RF calibration")
plt.show()

# disp = CalibrationDisplay.from_estimator(rf, X_train, y_train, n_bins=20)
# disp.figure_.figure.dpi = 80
# plt.title("Train set RF calibration")
# plt.show()


print(" ##### RF calibrated ##### ")

disp2 = CalibrationDisplay.from_estimator(rf_calibr, X_test, y_test, n_bins=20)
disp2.figure_.figure.dpi = 80
plt.title("Test set calibr_RF calibration")
plt.show()


# disp2 = CalibrationDisplay.from_estimator(rf_calibr, X_train, y_train, n_bins=20)
# disp2.figure_.figure.dpi = 80
# plt.title("Train set calibr_RF calibration")
# plt.show()


#%%

''' trying a more proper calibration routine ''' 

X = X_train.iloc[:20000]
y = y_train[:20000]

weights = np.array([5 if y[i] == True else 1 for i in range(len(y))])

weights = {0: 1, 1: 9}

#%%

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1, max_iter=1000)

lr_weights = LogisticRegression(C=1, max_iter=1000, class_weight=weights)


rf = RandomForestClassifier(max_depth=10,
                            random_state=0)

rf_weights = RandomForestClassifier(max_depth=10, random_state=0,
                            class_weight=weights)

rf_isotonic = CalibratedClassifierCV(rf, cv=5,
                                      method="isotonic") 

rf_sigmoid = CalibratedClassifierCV(rf, cv=5,
                                    method="sigmoid")

#LTreeX = LocalTreeExtractor(rf, "binary", 0)

clf_list = [
    (lr, "Logistic"),
    (lr_weights, "Logistic + weights"),
    (rf, "Random Forest"),
    (rf_weights, "Random Forest + weights"),
    (rf_isotonic, "Random Forest + Isotonic"),
    (rf_sigmoid, "Random Forest + Sigmoid"),
]

fig = plt.figure(figsize=(15, 10))
gs = GridSpec(4, 3) # create GridSpace for multiple subplots (convenient!)
colors = plt.cm.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :3])
calibration_displays = {}
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=20,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots")

# Add histogram
grid_positions = [(2, 0), (3, 0), (2, 1), (2, 2), (3, 1), (3, 2)]
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()



#%%
''' calibration status:
    
    Performed some calibration test with different set-ups, playing with
    class_weights for example (since the dataset is imbalanced).
    - Logistic Regression is a baseline for typically calibrated models
    
    - Random Forest is investigated, since our model imitates it
    (store y_preds and run a plot in the future?)
    In particular:
        - normal RF ( no intervention at all)
        - RF with balanced weights
        - RF with isotonic or sigmoid calibration
        
        
    OBSERVATIONS:
        - non weighted classifiers tend to predict 0 or at most 0.4
        - weighted LR and RF have nice gaussians in their predictions,
            but they are not well calibrated
        - the best calibarted model is unweighted RF, but calibration is 
            not defined after 0.4 (too few preditions!)
            
        TRY with all dataset, and more bins.
        
        Trying with max_depth=15 or 20, maybe leads to more extreme predictions?
    
    '''

# from sklearn.calibration import calibration_curve

# prob_true, prob_pred = calibration_curve(y_test, y_ens_pred, n_bins=20)
    

#%%

sorted_idx_imp = np.argsort(-1*rf.feature_importances_)

features_plot = rf.feature_names_in_[sorted_idx_imp][:6]

import matplotlib.pyplot as plt
from time import time
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay

print("Computing partial dependence plots...")
tic = time()

display = PartialDependenceDisplay.from_estimator(
    rf,
    X_train,
    features_plot,
    kind="both",
    subsample=50,
    n_jobs=-2,
    grid_resolution=20,
    random_state=0,
    ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
    pd_line_kw={"color": "tab:orange", "linestyle": "--"},
)
display.figure_.dpi = 90
#display.figure_.figsize=(21, 15) does not work
print(f"done in {time() - tic:.3f}s")
display.figure_.suptitle(
    "Partial dependence of EDSS progression on most \n"
    "important features, predicted with RF classifier"
)
display.figure_.subplots_adjust(hspace=0.3)




