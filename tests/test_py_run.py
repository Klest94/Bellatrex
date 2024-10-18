'''
Author: Klest Dedja
Here we manually test most of the features
'''
import os
import matplotlib.pyplot as plt

import bellatrex as btrex
print("Bellatrex version:", btrex.__version__)

MAX_TEST_SAMPLES = 2
PLOT_GUI = False

root_folder = os.getcwd()
print(root_folder)

from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split


from bellatrex.datasets import load_mtr_data, load_mlc_data
from bellatrex.datasets import load_survival_data, load_binary_data, load_regression_data
from bellatrex.utilities import get_auto_setup

# X, y = load_binary_data(return_X_y=True)
X, y = load_regression_data(return_X_y=True)
# X, y = load_survival_data(return_X_y=True)
# X, y = load_mlc_data(return_X_y=True)
# X, y = load_mtr_data(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

SETUP = get_auto_setup(y)
print('Detected prediction task \'SETUP\':', SETUP)


### instantiate original R(S)F estimator, works best with some pruning.
if SETUP.lower() in 'survival':
    clf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
                                n_jobs=-2, random_state=0)

elif SETUP.lower() in ['binary', 'multi-label']:
    clf = RandomForestClassifier(n_estimators=100, min_samples_split=5,
                                n_jobs=-2, random_state=0)

elif SETUP.lower() in ['regression', 'multi-target']:
    clf = RandomForestRegressor(n_estimators=100, min_samples_split=5,
                                n_jobs=-2, random_state=0)
else:
    raise ValueError(f"Detection task {SETUP} not compatible with Bellatrex (yet)")


from bellatrex import BellatrexExplain
from bellatrex.wrapper_class import pack_trained_ensemble

clf.fit(X_train, y_train)
print('Model fitting complete.')

clf_packed = pack_trained_ensemble(clf)

# Load the pretrained model and make it compatible with Bellatrex through EnsembleWrapper()
# clf2 = EnsembleWrapper(clf_packed)

# fit RF model here. The hyperparameters are given
# compatible with trained model clf, and with a wrapped dictionary as in clf_packed

Btrex_fitted = BellatrexExplain(clf_packed, set_up='auto',
                                p_grid={"n_clusters": [1, 2, 3]},
                                verbose=3).fit(X_train, y_train)

from bellatrex.utilities import predict_helper

for i in range(MAX_TEST_SAMPLES): # iterate for the first few samples in the test set

    print(f"Explaining sample i={i}")

    # Store the extracted rules in dedicated folder, if needed:
    explan_dir = os.path.join(root_folder, "explanations-out")
    os.makedirs(explan_dir, exist_ok=True)
    FILE_OUT = os.path.join(explan_dir, "Rules_"+str(SETUP)+'_id'+str(i)+'.txt')

    y_train_pred = predict_helper(clf, X_train)

    tuned_method = Btrex_fitted.explain(X_test, i)
    # tuned_method.plot_overview(show=True)

    tuned_method.plot_visuals(plot_max_depth=5,
                              preds_distr=y_train_pred,
                              conf_level=0.9,
                              tot_digits=4)
    plt.show()

    # tuned_method.plot_overview(show=True)

# if using Qt5Agg backend, the figures will be deleted unless we call the following:
# plt.show()

# %%
