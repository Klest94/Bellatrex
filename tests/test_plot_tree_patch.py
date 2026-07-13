import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

try:
    from bellatrex.plot_tree_patch import _color_brew, _MPLTreeExporter, plot_tree_patched
except ImportError:
    from app.bellatrex.plot_tree_patch import _color_brew, _MPLTreeExporter, plot_tree_patched


def test_color_brew_length_and_type():
    n = 5
    colors = _color_brew(n)
    assert isinstance(colors, list)
    assert len(colors) == n
    for color in colors:
        assert isinstance(color, list)
        assert len(color) == 3
        assert all(isinstance(c, int) for c in color)
        assert all(0 <= c <= 255 for c in color)


def test_plot_tree_patched_runs():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    fig, ax = plt.subplots()
    anns = plot_tree_patched(clf, ax=ax)
    assert isinstance(anns, list)
    assert all(hasattr(a, "get_window_extent") for a in anns)
    plt.close(fig)


def test_plot_tree_patched_with_feature_names():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X, y)
    fig, ax = plt.subplots()
    anns = plot_tree_patched(clf, feature_names=iris.feature_names, ax=ax)
    assert isinstance(anns, list)
    plt.close(fig)


def test_exporter_invalid_precision():
    with pytest.raises(ValueError):
        _MPLTreeExporter(precision=-1)
    with pytest.raises(ValueError):
        _MPLTreeExporter(precision="bad")


@pytest.mark.skipif(
    pytest.importorskip("sksurv", reason="sksurv not installed") is None,
    reason="sksurv not available",
)
def test_plot_tree_patched_survivaltree():
    from sksurv.tree import SurvivalTree

    X = np.random.rand(20, 2)
    y = np.array([(True, 1.0 + i) for i in range(20)], dtype=[("event", "?"), ("time", "f8")])
    clf = SurvivalTree(max_depth=1, random_state=0)
    clf.fit(X, y)
    fig, ax = plt.subplots()
    anns = plot_tree_patched(clf, ax=ax)
    assert isinstance(anns, list)
    plt.close(fig)


class DummyTree:
    def __init__(
        self,
        n_classes=2,
        n_outputs=1,
        value=None,
        impurity=None,
        threshold=None,
        feature=None,
        children_left=None,
        children_right=None,
        n_node_samples=None,
        weighted_n_node_samples=None,
    ):
        self.n_classes = [n_classes]
        self.n_outputs = n_outputs
        self.value = value if value is not None else np.array([[[10, 5]]])
        self.impurity = impurity if impurity is not None else np.array([0.5])
        self.threshold = threshold if threshold is not None else np.array([0.5])
        self.feature = feature if feature is not None else np.array([0])
        self.children_left = children_left if children_left is not None else np.array([-1])
        self.children_right = children_right if children_right is not None else np.array([-1])
        self.n_node_samples = n_node_samples if n_node_samples is not None else np.array([15])
        self.weighted_n_node_samples = (
            weighted_n_node_samples if weighted_n_node_samples is not None else np.array([15.0])
        )


def test_get_color_and_fill_color_classification():
    exporter = _MPLTreeExporter()
    exporter.colors = {"rgb": _color_brew(2), "bounds": None}
    tree = DummyTree()
    color = exporter.get_color([0.7, 0.3])
    assert isinstance(color, str) and color.startswith("#")
    fill = exporter.get_fill_color(tree, 0)
    assert isinstance(fill, str) and fill.startswith("#")


def test_get_color_and_fill_color_regression():
    exporter = _MPLTreeExporter()
    exporter.colors = {"rgb": _color_brew(1), "bounds": (0, 1)}
    tree = DummyTree(n_classes=1, value=np.array([[[0.5]]]))
    color = exporter.get_color(0.5)
    assert isinstance(color, str) and color.startswith("#")
    fill = exporter.get_fill_color(tree, 0)
    assert isinstance(fill, str) and fill.startswith("#")


def test_node_to_str_various_options():
    exporter = _MPLTreeExporter(node_ids=True, label="all", proportion=True)
    exporter.colors = {"rgb": _color_brew(2), "bounds": None}
    exporter.characters = ["#", "[", "]", "<=", "\n", "", ""]
    tree = DummyTree(
        n_classes=2,
        value=np.array([[[10, 5]]]),
        impurity=np.array([0.5]),
        feature=np.array([0]),
        threshold=np.array([0.5]),
        children_left=np.array([-1]),
        n_node_samples=np.array([15]),
        weighted_n_node_samples=np.array([15.0]),
    )
    s = exporter.node_to_str(tree, 0, criterion="gini")
    assert isinstance(s, str)

    # Test with regression
    tree_reg = DummyTree(n_classes=1, value=np.array([[[0.5]]]), impurity=np.array([0.2]))
    s2 = exporter.node_to_str(tree_reg, 0, criterion="mse")
    assert isinstance(s2, str)


def test_node_to_str_invalid_node_plot():
    exporter = _MPLTreeExporter()
    exporter.colors = {"rgb": _color_brew(1), "bounds": None}
    exporter.node_plot = "invalid"
    tree = DummyTree(n_classes=1, value=np.zeros((1, 1, 1)), impurity=np.array([0.1]))
    # Simulate logrank criterion and label
    with pytest.raises(ValueError):
        exporter.node_to_str(tree, 0, criterion="logrank")


def test_node_to_str_label_root():
    exporter = _MPLTreeExporter(node_ids=True, label="root")
    exporter.colors = {"rgb": _color_brew(2), "bounds": None}
    exporter.characters = ["#", "[", "]", "<=", "\n", "", ""]
    tree = DummyTree(
        n_classes=2,
        value=np.array([[[10, 5]]]),
        impurity=np.array([0.5]),
        feature=np.array([0]),
        threshold=np.array([0.5]),
        children_left=np.array([-1]),
        n_node_samples=np.array([15]),
        weighted_n_node_samples=np.array([15.0]),
    )
    s = exporter.node_to_str(tree, 0, criterion="gini")
    assert isinstance(s, str)
