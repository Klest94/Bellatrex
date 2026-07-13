import numpy as np
import pytest
from bellatrex.wrapper_class import (
    EnsembleWrapper,
    pack_trained_ensemble,
    tree_list_to_model,
    tree_to_dict,
)
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sksurv.ensemble import RandomSurvivalForest
# from sksurv.tree import SurvivalTree

RNG = np.random.default_rng(0)
X = RNG.random((20, 3))
Y_BINARY = np.array([0, 1] * 10)
Y_CONTIN = RNG.uniform(0.0, 5.0, size=20)


# --- Helper for mock tree dicts ---
def make_tree_dict(learner_class="DecisionTreeClassifier", n_outputs=1):
    # Minimal tree dict for EnsembleWrapper
    return {
        "features": [0, 1, -1],
        "n_leaves": 2,
        "node_sample_weight": [10, 5, 5],
        "children_left": [1, -1, -1],
        "children_right": [2, -1, -1],
        "thresholds": [0.5, -2, -2],
        "values": (
            np.array([[0.4, 0.6], [0.6, 0.4], [0.2, 0.8]])
            if n_outputs == 2
            else np.array([[0.6], [0.4], [0.8]])
        ),
        "base_offset": 0.6,
        "max_depth": 1,
        "feature_names_in_": ["f1", "f2"],
        "n_features_in_": 2,
        "learner_class": learner_class,
        "output_format": "auto",
        "ensemble_class": "RandomForestClassifier",
    }


def test_pack_trained_ensemble_classifier():
    clf = RandomForestClassifier(n_estimators=2, random_state=0).fit(X, Y_BINARY)
    packed = pack_trained_ensemble(clf, set_up="auto")
    assert isinstance(packed, dict)
    assert "trees" in packed
    assert packed["ensemble_class"] == "RandomForestClassifier"


def test_pack_trained_ensemble_regressor():
    clf = RandomForestRegressor(n_estimators=2, random_state=0).fit(X, Y_CONTIN)
    packed = pack_trained_ensemble(clf, set_up="auto")
    assert isinstance(packed, dict)
    assert "trees" in packed
    assert packed["ensemble_class"] == "RandomForestRegressor"


def test_pack_trained_ensemble_invalid():
    with pytest.raises(ValueError):
        pack_trained_ensemble(object(), set_up="auto")


def test_ensemble_wrapper_predict_and_decision_path():
    tree_dict = make_tree_dict()
    model_dict = tree_list_to_model([tree_dict, tree_dict])
    wrapper = EnsembleWrapper(model_dict)
    preds = wrapper.predict(X)
    assert preds.shape == (20, wrapper.n_outputs_)
    paths, ptr = wrapper.decision_path(X)
    assert isinstance(paths, csr_matrix)
    assert isinstance(ptr, np.ndarray)
    # __getitem__
    est = wrapper[0]
    assert hasattr(est, "predict")


def test_estimator_predict_and_decision_path():
    tree_dict = make_tree_dict()
    est = EnsembleWrapper.Estimator(tree_dict)
    preds = est.predict(X)
    assert preds.shape[0] == 20
    path = est.decision_path(X)
    assert isinstance(path, csr_matrix)


def test_tree_to_dict_binary():
    clf = RandomForestClassifier(n_estimators=1, random_state=0).fit(X, Y_BINARY)
    d = tree_to_dict(clf, 0, output_format="auto")
    assert isinstance(d, dict)
    assert "values" in d


def test_tree_to_dict_regression():
    clf = RandomForestRegressor(n_estimators=1, random_state=0).fit(X, Y_CONTIN)
    d = tree_to_dict(clf, 0, output_format="auto")
    assert isinstance(d, dict)
    assert "values" in d


def test_tree_to_dict_invalid():
    class Dummy:
        pass

    with pytest.raises(Exception):
        tree_to_dict(Dummy(), 0, output_format="auto")


def test_tree_list_to_model_consistency():
    tree_dict1 = make_tree_dict()
    tree_dict2 = make_tree_dict()
    tree_dict1["output_format"] = tree_dict2["output_format"] = "auto"
    tree_dict1["ensemble_class"] = tree_dict2["ensemble_class"] = "RandomForestClassifier"
    model = tree_list_to_model([tree_dict1, tree_dict2])
    assert isinstance(model, dict)
    assert model["ensemble_class"] == "RandomForestClassifier"
    assert model["output_format"] == "auto"
    assert "trees" in model


def test_tree_to_dict_error_branches():
    # Test error for unrecognized combination
    class DummyTree:
        n_outputs_ = 99

        def __init__(self):
            self.value = np.zeros((3, 3, 3))
            self.children_left = [1, -1, -1]
            self.children_right = [2, -1, -1]
            self.n_leaves = 2
            self.max_depth = 1
            self.feature = [0, 1, -1]
            self.threshold = [0.5, -2, -2]
            self.weighted_n_node_samples = [10, 5, 5]
            self.tree_ = self

    class DummyClf:
        def __getitem__(self, idx):
            return DummyTree()

        __class__ = type("Dummy", (), {})

    with pytest.raises(ValueError):
        tree_to_dict(DummyClf(), 0, output_format="unknown")
