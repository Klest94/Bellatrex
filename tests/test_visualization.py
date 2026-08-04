import matplotlib
import numpy as np

matplotlib.use("Agg")  # for headless testing
import matplotlib.pyplot as plt
from bellatrex.visualization import parse, plot_rules, read_rules


# --- plot_rules ---
def test_plot_rules_basic():
    rules = [["feature_1 <= 0.5", "feature_2 > 1.0"]]
    preds = [[0.2, 0.8]]
    baselines = [[0.1]]
    weights = [1.0]
    fig, axs = plot_rules(rules, preds, baselines, weights)
    assert fig is not None
    assert axs is not None
    plt.close(fig)


def test_plot_rules_with_other_preds():
    rules = [["feature_1 <= 0.5", "feature_2 > 1.0"]]
    preds = [[0.2, 0.8]]
    baselines = [[0.1]]
    weights = [1.0]
    other_preds = [[0.15, 0.85]]
    fig, axs = plot_rules(rules, preds, baselines, weights, other_preds=other_preds)
    assert fig is not None
    plt.close(fig)


def test_plot_rules_with_preds_distr():
    rules = [["feature_1 <= 0.5", "feature_2 > 1.0"]]
    preds = [[0.2, 0.8]]
    baselines = [[0.1]]
    weights = [1.0]
    preds_distr = np.random.normal(0.5, 0.1, 100)
    fig, axs = plot_rules(rules, preds, baselines, weights, preds_distr=preds_distr)
    assert fig is not None
    plt.close(fig)


def test_plot_rules_with_conf_level():
    rules = [["feature_1 <= 0.5", "feature_2 > 1.0"]]
    preds = [[0.2, 0.8]]
    baselines = [[0.1]]
    weights = [1.0]
    other_preds = [[0.15, 0.85]]
    fig, axs = plot_rules(rules, preds, baselines, weights, other_preds=other_preds, conf_level=0.9)
    assert fig is not None
    plt.close(fig)


def test_plot_rules_with_bbox_pred():
    rules = [["feature_1 <= 0.5", "feature_2 > 1.0"]]
    preds = [[0.2, 0.8]]
    baselines = [[0.1]]
    weights = [1.0]
    fig, axs = plot_rules(rules, preds, baselines, weights, b_box_pred=0.7)
    assert fig is not None
    plt.close(fig)


# --- parse ---
def test_parse_latex_and_parenthesis():
    s = "feature_1 <= 0.5 (val=0.3)"
    out = parse(s)
    assert r"$\\leq$" in out or "$\\leq$" in out
    s2 = "feature_2 >= 1.0 (val=1.2)"
    out2 = parse(s2)
    assert r"$\\geq$" in out2 or "$\\geq$" in out2


def test_parse_removes_after_paren():
    s = "feature_1 <= 0.5 (val=0.3)"
    out = parse(s)
    assert "(" not in out or out.endswith("(")


# --- read_rules ---
def test_read_rules(tmp_path):
    rule_file = tmp_path / "rules.txt"
    extra_file = tmp_path / "rules_extra.txt"
    rule_file.write_text("""
RULE WEIGHT: 0.5
Baseline prediction: 0.1
node: feature_1 <= 0.5 --> 0.2
node: feature_2 > 1.0 --> 0.8
leaf
Bellatrex prediction: 0.8
""")
    extra_file.write_text("""
Baseline prediction: 0.1
node: feature_1 <= 0.5 --> 0.2
node: feature_2 > 1.0 --> 0.8
leaf
""")
    rules, preds, baselines, weights, other_preds = read_rules(str(rule_file), str(extra_file))
    assert isinstance(rules, list)
    assert isinstance(preds, list)
    assert isinstance(baselines, list)
    assert isinstance(weights, list)
    assert isinstance(other_preds, list)
    # Test without extra file
    rules2, preds2, baselines2, weights2, other_preds2 = read_rules(str(rule_file))
    assert other_preds2 is None
