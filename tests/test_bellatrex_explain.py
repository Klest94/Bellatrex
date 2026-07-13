import os
from importlib import import_module

import numpy as np
import pandas as pd
import pytest

try:  # TODO: Paths to be updated: this workaround makes tests work across different setups.
    from app.bellatrex.bellatrex_explain import BellatrexExplain
except ImportError:
    from bellatrex.bellatrex_explain import BellatrexExplain
from sklearn.ensemble import RandomForestClassifier
from sksurv.ensemble import RandomSurvivalForest


@pytest.fixture
def mock_clf():
    return RandomForestClassifier(n_estimators=10, random_state=0)


@pytest.fixture
def mock_survival_clf():
    return RandomSurvivalForest(n_estimators=10, random_state=0)


@pytest.fixture
def mock_data():
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
    y = np.random.randint(0, 2, size=100)  # random binary target
    return X, y


@pytest.fixture
def mock_survival_data():
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])

    times = np.random.exponential(scale=5, size=100).astype(float)
    status = np.random.binomial(1, 0.7, size=100).astype(bool)
    # order of columns is: status, times
    y = np.rec.fromarrays([status, times], dtype=[("event", "?"), ("time", "f8")])

    return X, y


def test_bellatrex_explain_fit(mock_clf, mock_data):
    X, y = mock_data
    explainer = BellatrexExplain(mock_clf, verbose=1)
    explainer.fit(X, y)
    assert explainer.is_fitted() is True


def test_partial_grid_preserves_unspecified_defaults(mock_clf, mock_data):
    X, y = mock_data
    explainer = BellatrexExplain(mock_clf, p_grid={"n_clusters": [1]}).fit(X, y)

    assert explainer.n_trees == [6, 8, 10]
    assert explainer.n_dims == [2, None]
    assert explainer.n_clusters == [1]


@pytest.mark.parametrize("n_jobs", [0, -1, True, 1.5])
def test_n_jobs_must_be_a_positive_integer(mock_clf, n_jobs):
    with pytest.raises(ValueError, match="positive integer"):
        BellatrexExplain(mock_clf, n_jobs=n_jobs)


def test_parallel_grid_search_selects_best_candidate(monkeypatch, mock_clf, mock_data):
    X, y = mock_data
    mock_clf.fit(X, y)
    module = import_module(BellatrexExplain.__module__)

    class FakeTreeExtraction:
        def __init__(
            self,
            proj_method,
            dissim_method,
            feature_represent,
            n_trees,
            n_dims,
            n_clusters,
            pre_select_loss,
            fidelity_measure,
            clf,
            oracle_sample,
            set_up,
            sample,
            verbose,
            output_explain=False,
        ):
            self.proj_method = proj_method
            self.dissim_method = dissim_method
            self.feature_represent = feature_represent
            self.n_trees = n_trees
            self.n_dims = n_dims
            self.n_clusters = n_clusters
            self.pre_select_loss = pre_select_loss
            self.fidelity_measure = fidelity_measure
            self.clf = clf
            self.oracle_sample = oracle_sample
            self.set_up = set_up
            self.sample = sample
            self.verbose = verbose
            self.output_explain = output_explain
            self.final_trees_idx = [0]
            self.cluster_sizes = np.array([1])

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

        def main_fit(self):
            return self

        def score(self, fidelity_measure, oracle_sample):
            return float(self.n_clusters)

    monkeypatch.setattr(module, "TreeExtraction", FakeTreeExtraction)
    explainer = BellatrexExplain(
        mock_clf,
        n_jobs=2,
        p_grid={"n_trees": [3], "n_dims": [None], "n_clusters": [1, 2]},
    ).fit(X, y)

    explainer.explain(X, 0)

    assert explainer.tuned_method.n_clusters == 2
    assert explainer.tuned_method.sample_score == 2.0


def test_numpy_oracle_values_are_indexed_positionally(monkeypatch, mock_clf, mock_data):
    X, y = mock_data
    oracle = np.linspace(0.0, 1.0, len(X))
    seen_oracles = []
    module = import_module(BellatrexExplain.__module__)
    original_score = module.TreeExtraction.score

    def recording_score(self, fidelity_measure, oracle_sample):
        seen_oracles.append(oracle_sample)
        return original_score(self, fidelity_measure, oracle_sample)

    monkeypatch.setattr(module.TreeExtraction, "score", recording_score)
    explainer = BellatrexExplain(
        mock_clf,
        ys_oracle=oracle,
        p_grid={"n_trees": [3], "n_dims": [None], "n_clusters": [1]},
    ).fit(X, y)

    explainer.explain(X, 2)

    assert seen_oracles
    assert all(value == oracle[2] for value in seen_oracles)


def test_projection_can_be_disabled(mock_clf, mock_data):
    X, y = mock_data
    explainer = BellatrexExplain(
        mock_clf,
        proj_method=None,
        p_grid={"n_trees": [3], "n_dims": [2], "n_clusters": [1]},
    ).fit(X, y)

    result = explainer.explain(X, 0)

    assert result.tuned_method.proj_method is None


# def test_bellatrex_explain_explain(mock_clf, mock_data):
#     X, y = mock_data
#     explainer = BellatrexExplain(mock_clf, verbose=0)
#     explainer.fit(X, y)
#     explanation = explainer.explain(X, idx=0)
#     assert explanation is not None


def test_bellatrex_explain_plot_overview(mock_clf, mock_data):
    X, y = mock_data
    explainer = BellatrexExplain(mock_clf, verbose=1)
    explainer.fit(X, y)
    fig, axes = explainer.explain(X, idx=0).plot_overview(show=False)
    # fig, axes = explainer.plot_overview(show=False)
    assert fig is not None
    assert axes is not None


@pytest.mark.gui
def test_plot_overview_gui_show_false_does_not_launch_window(monkeypatch, tmp_path):
    pytest.importorskip("nicegui", reason="Install Bellatrex[gui] to run GUI tests")

    class FakeTunedMethod:
        final_trees_idx = []
        cluster_sizes = []
        clf = object()
        sample = pd.DataFrame([[1.0]], columns=["feature"])

        def preselect_represent_cluster_trees(self):
            return object(), object()

    explainer = object.__new__(BellatrexExplain)
    explainer.sample = pd.DataFrame([[1.0]], columns=["feature"])
    explainer.tuned_method = FakeTunedMethod()
    explainer.sample_index = 0
    explainer.surrogate_pred_str = "0.0"
    explainer.verbose = -1
    explainer.clf = object()

    module_root = BellatrexExplain.__module__.rsplit(".", 1)[0]
    nicegui_plots_code = import_module(f"{module_root}.nicegui_plots_code")

    def fail_launch(*args, **kwargs):
        raise AssertionError("show=False should not launch a NiceGUI window")

    monkeypatch.setattr(nicegui_plots_code, "plot_with_interface", lambda *a, **k: ["plot"])
    monkeypatch.setattr(nicegui_plots_code, "launch_nicegui_window", fail_launch)

    fig, axes = explainer.plot_overview(
        show=False,
        plot_gui=True,
        temp_gui_dir=str(tmp_path),
    )

    assert fig is not None
    assert axes == ["plot"]


@pytest.mark.xfail(raises=NotImplementedError, reason="Not implemented yet")
def test_predict_survival_curve(mock_survival_clf, mock_survival_data):
    X, y = mock_survival_data
    explainer = BellatrexExplain(mock_survival_clf, verbose=1, set_up="survival")
    explainer.fit(X, y)
    survival_curve = explainer.predict_survival_curve(X, 0)  # pylint: disable=E1111

    assert survival_curve is not None
    assert isinstance(survival_curve, pd.DataFrame)


@pytest.mark.xfail(raises=NotImplementedError, reason="Not implemented yet")
def test_predict_median_surv_time(mock_survival_clf, mock_survival_data):
    X, y = mock_survival_data
    explainer = BellatrexExplain(mock_survival_clf, verbose=1, set_up="survival")
    explainer.fit(X, y)
    survival_curve = explainer.predict_median_surv_time(X, 0)  # pylint: disable=E1111
    assert survival_curve is not None
    assert isinstance(survival_curve, pd.DataFrame)


# BIG DUMMY TREE TO MAKE EVERYTHING WORK.
# Better refactgor code than go through this pain.
# class DummyTree(dict):
#     def __init__(self):
#         super().__init__()
#         self.value = [[0]]
#         self.n_outputs_ = 1
#         self["feature_names_in_"] = ["f1", "f2"]
#         self["n_features_in_"] = 2

#     def get(self, *a, **k):
#         return self.value


# def test_is_fitted_dict():
#     # Should wrap dict in EnsembleWrapper and return True

#     dict_trees = {"trees": [DummyTree(), DummyTree(), DummyTree()]}
#     explainer = BellatrexExplain(dict_trees)
#     assert explainer.is_fitted() is True


# def test_is_fitted_ensemblewrapper():
#     from bellatrex.wrapper_class import EnsembleWrapper

#     dict_trees = {"trees": [DummyTree(), DummyTree(), DummyTree()]}
#     ew = EnsembleWrapper(dict_trees)
#     explainer = BellatrexExplain(ew)
#     assert explainer.is_fitted() is True


# def test_fit_force_refit_verbose(monkeypatch, mock_clf, mock_data):
#     X, y = mock_data
#     explainer = BellatrexExplain(mock_clf, force_refit=True, verbose=2)
#     called = {}

#     def fake_fit(X_, y_, n_jobs):
#         called["fit"] = True

#     monkeypatch.setattr(mock_clf, "fit", fake_fit)
#     explainer.fit(X, y)
#     assert called["fit"]


# def test_explain_verbose(monkeypatch, mock_clf, mock_data):
#     X, y = mock_data
#     explainer = BellatrexExplain(mock_clf, verbose=5)
#     explainer.fit(X, y)

#     # Patch TreeExtraction to avoid heavy computation
#     class DummyTE:
#         def __init__(self, *a, **k):
#             pass

#         def set_params(self, **params):
#             return self

#         def main_fit(self):
#             class Dummy:
#                 final_trees_idx = [0]
#                 cluster_sizes = [1]
#                 score = lambda self, *a, **k: 1.0

#             return Dummy()

#     monkeypatch.setattr("app.bellatrex.bellatrex_explain.TreeExtraction", DummyTE)
#     explainer.explain(X, idx=0)
#     assert hasattr(explainer, "tuned_method")


def test_create_rules_txt_file(monkeypatch, mock_clf, mock_data, tmp_path):
    X, y = mock_data
    explainer = BellatrexExplain(mock_clf)
    explainer.fit(X, y)

    class DummyModel:
        final_trees_idx = [0]
        cluster_sizes = [1]
        sample = X.iloc[[0]]

    explainer.tuned_method = DummyModel()
    explainer.sample = X.iloc[[0]]
    explainer.sample_index = 0
    explainer.surrogate_pred_str = "0.0"
    explainer.clf = mock_clf
    # Monkeypatch the symbols as imported into bellatrex_explain to avoid file IO
    monkeypatch.setattr("bellatrex.bellatrex_explain.rule_to_file", lambda *a, **k: None)
    monkeypatch.setattr(
        "bellatrex.bellatrex_explain.read_rules", lambda **k: ([1], [1], [1], [1], [1])
    )
    monkeypatch.setattr("bellatrex.bellatrex_explain._input_validation", lambda *a, **k: None)

    explanation_out_dir_path = tmp_path
    out_file, file_extra = explainer.create_rules_txt(
        out_dir=str(explanation_out_dir_path), out_file="testing_rules.txt"
    )
    assert os.path.exists(out_file)
    assert os.path.exists(file_extra)

    os.remove(out_file)
    os.remove(file_extra)
