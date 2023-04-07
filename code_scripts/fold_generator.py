import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from utilities import get_data_list



''' 
this script has been used to pre-process prepare the data folds, 
as indicated in Section 5 of the manuscript
'''

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

SETUP = "mt_regress"

N_FOLDS = 5
MAX_TEST_SIZE= 100

root_folder = os.getcwd()
# "C:\Users\u0135479\Documents\GitHub\Explain-extract-trees\"

dnames, datapath, hyperpar_folder = get_data_list(SETUP, root_folder)

dnames = list(filter(lambda f: not f.endswith('.csv'), dnames)) # keeps only .csv


binary_key_list = ["bin", "binary"]
survival_key_list = ["surv", "survival"]
multi_label_key_list = ["multi", "multi-l", "multi-label"]
regression_key_list = ["regression", "regress", "regr"]
mt_regression_key_list = ["multi-target", "multi-t", "mtr", "mt_regress"]


##########################################################################
testing_dnames = dnames#[4:5]
#testing_dnames = [dnames[i] for i in [-1]]
##########################################################################
#%%
print("running on datasets:")
for data in testing_dnames:
    print(data)

results_df = pd.DataFrame() # empty datafarme to start with. Append here

for folder in testing_dnames:
    # not working as expected, manual fixes
    #X, y, stratify_status, inner_folds = preparing_data(SETUP, datapath, folder, N_FOLDS, True)
    #f folder.split(".")[0] == 'wine_quality_white':
    if SETUP.lower() in  regression_key_list:
        stratify = False # not possible
        data = pd.read_csv(datapath + folder + ".csv")
        assert data.isnull().sum().sum() < 1

        X = data.drop(data.columns[-1], axis=1, inplace=False)
        #X  = X.astype('float64') #apparently needed        
        y = data.iloc[:,-1].ravel() #last column is regression target
        y = (y - y.min())/(y.max() - y.min()) # scale to [0-1]        
        y_strat = None
    
    elif  SETUP.lower() in binary_key_list:
        stratify = True
        df1 = pd.read_csv(datapath + folder + "/train1.csv") # <- to changee
        df2 = pd.read_csv(datapath + folder + "/valid1.csv")
        df3 = pd.read_csv(datapath + folder + "/test1.csv")
        data = pd.concat([df1, df2, df3], ignore_index=True)
        assert data.isnull().sum().sum() < 1
        
        X = data.drop(data.columns[-1], axis=1, inplace=False)
        #X  = X.astype('float64') #apparently needed        
        y = data.iloc[:,-1].ravel() #last column is regression target
        y_strat = y

    elif SETUP.lower() in survival_key_list:  #last two columns for event
        stratify = True
        data = pd.read_csv(datapath + folder + ".csv") # <- to changee
        data[data.columns[-2]] = data[data.columns[-2]].astype('bool') # needed for correct recarray
        assert data.isnull().sum().sum() < 1

        X = data[data.columns[:-2]].astype('float64') #astype apparently needed
        y = data[data.columns[-2:]]#.to_numpy()
        y = y.to_records(index=False) #builds the structured array, needed for RSF
        y_strat = np.array([i[0] for i in y])
        
    elif SETUP.lower() in multi_label_key_list:
        stratify = False # not possible
        try:
            df1 = pd.read_csv(datapath + folder + "/" + str(folder) + "_train_fold_1.csv")
            df2 = pd.read_csv(datapath + folder + "/" + str(folder) + "_test_fold_1.csv")
            data = pd.concat([df1, df2], ignore_index=True)
        except: # new datasets are stored differently
            data = pd.read_csv(datapath + folder + "/" + str(folder) + ".csv")

        assert data.isnull().sum().sum() < 1
        y_cols = [col for col in data.columns if "label" in col or "tag_" in col]
        x_cols = [col for col in data.columns if col not in y_cols]
        X = data.drop(y_cols, axis=1, inplace=False)
        #X  = X.astype('float64') #apparently needed        
        y = data.drop(x_cols, axis=1, inplace=False)
        print("y_shape:", y.shape)
        y_strat = None
        
    elif SETUP.lower() in mt_regression_key_list:
        stratify = False
        data = pd.read_csv(os.path.join(datapath, folder, folder + ".csv"))
        
        assert data.isnull().sum().sum() < 1
        y_cols = [col for col in data.columns if "target_" in col]
        x_cols = [col for col in data.columns if col not in y_cols]
        X = data.drop(y_cols, axis=1, inplace=False)
        #X  = X.astype('float64') #apparently needed        
        y = data.drop(x_cols, axis=1, inplace=False)
        print("X_shape:", X.shape)
        print("y_shape:", y.shape)
        
        y = (y-y.min())/(y.max()-y.min()) # min-max normalization (per target)

        
        y_strat = None

    j = 0  #outer fold index, (we are performing the outer CV "manually")
    
    if stratify == True:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
        inner_folds = kf.split(X, y_strat)
        
    elif stratify == False:      
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
        inner_folds = kf.split(X, y)
    
    for train_index, val_index in inner_folds:
        
        X_train, X_val = X.iloc[train_index,:], X.iloc[val_index,:] # N-1 + 1 split
        try:
            y_train, y_val = y[train_index], y[val_index]
        except KeyError:
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            
        custom_test_size = np.minimum(MAX_TEST_SIZE, 
                                      int(X_train.shape[0]/(N_FOLDS-1)))
        
        if SETUP in survival_key_list:
            y_strat = np.array([i[0] for i in y_train])
                               
        elif SETUP in binary_key_list:
            y_strat = y_train
        else: #no stratification for regression, MTR, MTC
             y_strat = None     
        
        X_train, X_test, y_train, y_test = \
        train_test_split(X_train, y_train, test_size= custom_test_size,
                        shuffle=True, random_state=0,
                        stratify=y_strat)
        
        # delete validation set (useless at this stage, GridSearchCV will do the job)
        X_train = pd.concat([X_train, X_val])
        y_train = np.concatenate([y_train, y_val])
        del X_val, y_val
        assert X_train.isnull().sum().sum() < 1

        
        folder_string = folder.split(".csv")[0]
        df_cols = X.columns
        
        X_train = pd.DataFrame(X_train, columns=df_cols).reset_index(drop=True)
        X_test = pd.DataFrame(X_test, columns=df_cols).reset_index(drop=True)
        y_train = pd.DataFrame(y_train).reset_index(drop=True)
        y_test = pd.DataFrame(y_test).reset_index(drop=True)
        
        dir_name = os.path.join(datapath, folder_string)

        X_train.to_csv(os.path.join(dir_name, "X_new_train" + str(j+1) + ".csv"),index=False)
        X_test.to_csv(os.path.join(dir_name, "X_new_test" + str(j+1) + ".csv"), index=False)
        y_train.to_csv(os.path.join(dir_name, "y_new_train" + str(j+1) + ".csv"), index=False)
        y_test.to_csv(os.path.join(dir_name, "y_new_test" + str(j+1) + ".csv"), index=False)
        
        data_train = pd.concat([X_train, y_train], axis=1, ignore_index=True)
        data_test = pd.concat([X_test, y_test], axis=1, ignore_index=True)
        
        data_train.to_csv(os.path.join(dir_name, "new_train" + str(j+1) + ".csv"), index=False)
        data_test.to_csv(os.path.join(dir_name, "new_test" + str(j+1) + ".csv"), index=False)
        
        print("done for j:", j)
        j+=1
    print("done for dataset:", folder)
    print("-------------------")

