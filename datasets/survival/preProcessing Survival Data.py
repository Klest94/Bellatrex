# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:12:33 2021

@author: u0135479
"""

"""----------------------------------------------------------------
Author: Klest Dedja
        u0135479
---------------------------------------------------------------"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

github_root = "C:/Users/u0135479/Documents/GitHub/Explain-extract-trees/" #Klest's path
datapath = github_root + "datasets/survival/"
dnames = os.listdir(datapath)   #consider dropping dnames[2] and dnames[6]
testing_dnames = dnames[5]
#%%
print(testing_dnames)

df0 = pd.read_csv(datapath + testing_dnames)
df = df0

print("number of NaNs:", df.isnull().sum().sum(), "\nin a df size of", df.shape)
print("In %:", (100*df.isnull().sum().sum())/df.size)
#print(df.dtypes)


## drop when too many missing values, and prepare imputer function:

def iter_imputer(the_df):
    #new_data = the_df
    imp = IterativeImputer(max_iter=20, random_state=1)
    imp.fit(the_df)
    new_data = pd.DataFrame(imp.transform(the_df), columns=the_df.columns)
    #new_data.columns = df.columns
    return new_data

def purity_cols(data, thresh_percent): 
    N = data.shape[0]
    my_X2 = data.dropna(axis='columns', thresh=thresh_percent*N*0.01+0.5)
    return my_X2

def purity_rows(data, thresh_percent): 
    N = data.shape[1]
    my_X2 = data.dropna(axis='rows', thresh=thresh_percent*N*0.01+0.5)
    return my_X2

thresh = 0.2

df = purity_cols(df, thresh)  #drop cols with at least x% missing
df = purity_rows(df, thresh)  #drop rows with at least x% missing

print("new size: ", df.shape)


# identifying categorical features and dummy_transforming them

categ_cols = df.select_dtypes(include=['object'])
categ_cols = categ_cols.columns

#%%

#### One-Hot encode (all) categorical columns (before imputing):
if len(categ_cols) > 0:
    df2 = df
    for cols in categ_cols:
        df2 = pd.concat([df2.iloc[:,:-2], pd.get_dummies(df2[cols], prefix=cols, dtype='int64'),
                         df2.iloc[:,-2:]], axis=1)    # ensuring Event and Status are at the end...
        #get_dummies(drop_first = False ?)
        # drop_first = False is behaving properly with missing values!
        df2 = df2.drop(columns=cols) #drop old object (categorical) column
else: #nothing to encode
    df2 = df
    
#surv_cols = df0.columns[-2:]    
#df2 = df2[[col for col in dfd if col not in surv_cols] + surv_cols]

print("df size after encoding: ", df2.shape)


#%%
#now impute through MICE
df2 = iter_imputer(df2)

df2.to_csv(testing_dnames[:-4]+'-imputed-'+ str(thresh) + 'b.csv', index=False)

print("Done. Final size:")
print(df2.shape)
#from sksurv.datasets import load_gbsg2
#X_alt, y_alt = load_gbsg2()

#df3 = pd.concat([X_alt, y_alt], ignore_index=True)


#%%
status_col = df2.columns[-2]
df2[status_col] = df2[status_col].astype('bool')

y2 = df2[df2.columns[-2:]]#.to_numpy()
X2 = df2[df2.columns[:-2]]
        
y_events = y2
y3 = list(zip(y2.iloc[:5,0], y2.iloc[:5,1]))
y2 = y2.to_records(index=False) #builds the structured array, needed for RSF



#%%
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.25, random_state=0)

rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=15,
                           min_samples_leaf=10, max_features="sqrt", n_jobs=2,
                           random_state=0, verbose=0)
rsf.fit(X_train, y_train)
print("C-index score: ", rsf.score(X_test, y_test))
y_pred = rsf.predict_survival_function(X_test, return_array=False)
y_pred9 = rsf.predict_survival_function(X_test.iloc[9:12], return_array=True)

step_funct_y = y_pred[0]
times = y_pred[9].x # universal
probs = y_pred[9].y
alef = y_pred[0]

plt.plot(times, probs)
plt.xlabel("time")
plt.ylabel("P(survival)")
plt.ylim(0,1)
#plt.savefig('Survi2Trial.png', dpi=100)
plt.show()
#%%


med_time = np.zeros(np.shape(y_test))
AU_SurvCurve = np.zeros(np.shape(y_test))

print("----")
j = 0
for curve in y_pred:
    idx = sum(1 for i in curve.y if i > 0.5) # sample index
    try:
        med_time[j] = (curve.x[idx-1] + curve.x[idx])/2
    except IndexError:
        med_time[j] = curve.x[idx-1]*np.log(0.5)/np.log(curve.y[-1]) # exponential fit
    
    AU_SurvCurve[j] = np.trapz(curve.y, curve.x, axis=-1)
    j += 1 

#%%

tree = rsf.estimators_[0]
tree_pred = tree.predict_survival_function(X_test, return_array=False)
tree_pred2= tree.predict_survival_function(X_test, return_array=True).T

alef = tree_pred[0]
#alef = rsf.estimators_[0]
#beta = alef.tree_.predict(X_test.to_numpy().astype('float32'))  (n-samples, n_times, 2)

step_funct_y = tree_pred[0]
times = tree_pred[49].x # possible time events are universal
probs = tree_pred[49].y # probability curvces are defined on the times


#%%


from plot_tree_patched import plot_tree

def plot_my_surv_tree(my_clf, tree_idx, sample_idx, the_data): # add "data" variale in the future
    #depth = [estimator.tree_.max_depth for estimator in alt_clf.estimators_][tree_idx]
    my_surv_tree = my_clf.estimators_[tree_idx]

    my_depth = my_surv_tree.tree_.max_depth
    smart_size_1 = 1 + 1.2*(1.4**my_depth)
    smart_size_2 = my_depth*1.1
    fig, ax = plt.subplots(dpi=200)
    
    plot_tree(my_clf[tree_idx], feature_names=the_data.columns, 
             impurity= False, label= "none", fontsize=9)
    plt.title("Tree n* %i predicting sample %i" % (tree_idx, sample_idx) )
    plt.savefig('SurvTree2Trial.png')

    plt.show()
    
    predict_curve_tree = my_surv_tree.predict_survival_function([the_data.iloc[sample_idx]], return_array=True).T
    
    fig, ax = plt.subplots(figsize=(3.7,1.7), dpi=240)
    plt.plot(my_surv_tree.event_times_, predict_curve_tree)
    plt.title("Tree n*%i predicting sample %i" % (tree_idx, sample_idx), fontsize=9)
    plt.xlabel("time")
    plt.ylabel("P(survival)", fontsize=9)
    plt.ylim(0,1)
    plt.savefig('Survi2Trial.png')
    plt.show()
    #print("predicted:", predict_sample_curve)

plot_my_surv_tree(rsf, 1, 4, X_train)

#my_clf = rsf
#sample_idx = 1
#the_data = X_train

def PlotSurvivalEstimate(my_clf, sample_idx, the_data):
    N = my_clf.n_estimators
    tree_curves = np.zeros([N, my_clf.event_times_.shape[0]])
    
    for (tree, i) in zip(rsf.estimators_, range(N)):
        tree_curves[i,:] = tree.predict_survival_function([the_data.iloc[sample_idx]], return_array=True)
    
    #tree_confidences = np.zeros([2, my_clf.event_times_.shape[0]])
    #surv_curve = my_clf.predict_survival_function([the_data.iloc[sample_idx]], return_array=True)
    #store rsf prediction curve
    # plt.plot
    #plt.plot(my_surv_tree.event_times_, predict_curve_tree)
    return tree_curves
    
    

PlotSurvivalEstimate(rsf, 1, X_train)
    

#rsf = RandomSurvivalForest()
#rsf.fit(X_train, y_train)
#plot_tree(rsf.estimators_[0], feature_names=X_train.columns.tolist(), 
#          impurity=False, label="none")

def Jaccard_Survtrees(my_clf, idx1, idx2): # it's a SIMILARITY measure
    the_DTree1 = my_clf.estimators_[idx1].tree_
    the_DTree2 = my_clf.estimators_[idx2].tree_
    
    # in leafs the "feature split" is = -2, and leafs must be dropped
    fe1 = np.bincount(the_DTree1.feature[the_DTree1.feature > -1 ]) 
    fe2 = np.bincount(the_DTree2.feature[the_DTree2.feature > -1 ])
    # PROBLEM: some features might be missing (at the tail), fill with zeros
    base1 = np.zeros(my_clf.n_features_)
    base2 = np.zeros(my_clf.n_features_)
    base1[:len(fe1)] += fe1
    base2[:len(fe2)] += fe2

    # Jaccardi formula as of Eq.3 in C443 Suppl. material    
    #take "elementwise" min and max along indeces and compute ratio

    return np.sum(np.minimum(base1, base2))/np.sum(np.maximum(base1, base2))

trial = Jaccard_Survtrees(rsf, 1, 2)

plot_my_surv_tree(rsf, 0, 3, X_train)

#%%

def Label_Surv_tree(my_clf, my_method, tree_idx):
    return 0    
my_clf = rsf
idx1 = 2
idx2 = 4
dataset = X_train

def LogMedian_Surv_Tree(my_clf, tree_idx, dataset):
  
    N = dataset.shape[0]
    med_times = np.zeros(dataset.shape[0])
    pred_curves = my_clf.estimators_[tree_idx].predict_survival_function(dataset, return_array=False)

    for (curve, j) in zip(pred_curves, range(N)): #loop around data samples 
        idx = sum(1 for i in curve.y if i > 0.5) # sample index
        try:
            med_times[j] = (curve.x[idx-1] + curve.x[idx])/2
        except IndexError:
            med_times[j] = curve.x[idx-1]*np.log(0.5)/np.log(curve.y[-1]) # exponential fit
        
    return np.log(med_times)

def Surv_trees_as_vects(my_clf, my_data): # add probabs = True, False
    n, k = my_clf.n_estimators, my_data.shape[0]
    my_trees_vects = np.zeros([n,k])
    for i in range(n): # for tree i:
        my_trees_vects[i,:] = LogMedian_Surv_Tree(my_clf, i, my_data)

    return pd.DataFrame(my_trees_vects, columns=my_data.index) #(i,j) is prediction  tree i on sample j




alef = LogMedian_Surv_Tree(rsf, 1, X_train)
beth = LogMedian_Surv_Tree(rsf, 2, X_train)
gamma =  Surv_trees_as_vects(rsf, X_test)
delta =  Surv_trees_as_vects(rsf, X_train)

def trees_surv_loss(my_clf, my_X, my_y): # the smaller-the-better
    my_tree_loss = np.zeros(my_clf.n_estimators)
    for i in range(my_clf.n_estimators):
        my_tree_loss[i] = my_clf.estimators_[i].score(my_X, my_y) 
    
    return my_tree_loss

alef = trees_surv_loss(rsf, X_test, y_test)
beta = trees_surv_loss(rsf, X_train, y_train)

# use pieces fomr here (ensemble part)
# AU_SurvCurve = np.zeros(np.shape(y_test))

# print("----")
# j = 0
# for curve in y_pred:
#     idx = sum(1 for i in curve.y if i > 0.5) # sample index
#     try:
#         med_time[j] = (curve.x[idx-1] + curve.x[idx])/2
#     except IndexError:
#         med_time[j] = curve.x[idx-1]*np.log(0.5)/np.log(curve.y[-1]) # exponential fit
    
#     AU_SurvCurve[j] = np.trapz(curve.y, curve.x, axis=-1)
#     j += 1 


