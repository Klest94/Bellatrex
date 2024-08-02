
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

X, y = load_binary_data(return_X_y=True)
# X, y = load_regression_data(return_X_y=True)
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


from bellatrex import BellatrexExplain
from bellatrex.wrapper_class import EnsembleWrapper, tree_list_to_model, DT_to_dict

clf.fit(X_train, y_train)
print('Model fitting complete.')

#fit RF here. The hyperparameters are given
# compatible with trained model clf, and with a wrapped dictionary as in clf1
Btrex_fitted = BellatrexExplain(clf, set_up='auto',
                                p_grid={"n_clusters": [1, 2, 3]},
                                verbose=3).fit(X_train, y_train)


# from bellatrex.visualisation import read_rules, plot_rules

for i in range(MAX_TEST_SAMPLES): # iterate for the first few samples in the test set

    print(f"Explaining sample i={i}")

    # Store the extracted rules in dedicated folder, if needed:
    explan_dir = os.path.join(root_folder, "explanations-out")
    os.makedirs(explan_dir, exist_ok=True)
    FILE_OUT = os.path.join(explan_dir, "Rules_"+str(SETUP)+'_id'+str(i)+'.txt')

    # call the .explain method. The hyperparameters are given from the .fit step
    # sample_info, fig, axis = Btrex_fitted.explain(X_test, i, show=True, plot_gui=False,
                                                #   out_file=FILE_OUT)

    # test passed:
    # Btrex_fitted.explain(X_test, i).plot_overview(plot_gui=False, show=True)

    # test passed:
    # fig, ax = Btrex_fitted.explain(X_test, i).plot_overview(plot_gui=False, show=False)
    # fig.suptitle(f"Plot overview for sample {i}", fontsize=16)
    # fig.subplots_adjust(top=0.85)
    # fig.savefig(f"Trial n_{i}.png")
    # plt.show()

    # test passed:
    # tuned_method = Btrex_fitted.explain(X_test, i).plot_overview(plot_gui=False,
    #                                                              plot_max_depth=7,
    #
    # test passed:
    # tuned_method = Btrex_fitted.explain(X_test, i).plot_visuals(plot_max_depth=5,
    #                                                             keep_files=True,
    #                                                             show=True)

    y_train_pred = clf.predict_proba(X_train)[:,1]
    y_test_pred = clf.predict_proba(X_test)[:,1]
    # y_train_pred = clf.predict(X_train)
    # y_test_pred = clf.predict(X_test)

    # currently testing:
    fig, ax = Btrex_fitted.explain(X_test, i).plot_visuals(plot_max_depth=5,
                                                           preds_distr=None,
                                                           conf_level=0.9,
                                                           tot_digits=4,
                                                           b_box_pred=None,
                                                           keep_files=False,
                                                           show=False)
    fig.suptitle(f"Plot overview trial, sample {i}", fontsize=16)
    plt.show()

    # tuned_method.plot_overview(show=True)

# if using Qt5Agg backend, the figures will be deleted unless we call the following:
# plt.show()

# %%
