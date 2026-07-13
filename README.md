<a name="logo-anchor"></a>
<p align="center">
<img src="https://github.com/KlestDedja/Bellatrex/blob/main/app/bellatrex-logo.png?raw=true" alt="Bellatrex Logo" width="60%"/>
</p>

[![Python Versions](https://img.shields.io/pypi/pyversions/bellatrex)](https://pypi.org/project/bellatrex/)
[![Downloads](https://static.pepy.tech/badge/bellatrex)](https://pepy.tech/project/bellatrex)
[![License](https://img.shields.io/github/license/KlestDedja/Bellatrex)](https://github.com/KlestDedja/Bellatrex/blob/main/LICENSE.txt)
[![Cross OS integration](https://github.com/KlestDedja/Bellatrex/actions/workflows/ci-matrix.yaml/badge.svg?branch=main)](https://github.com/KlestDedja/Bellatrex/actions/workflows/ci-matrix.yaml)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2023.3268866-blue)](https://doi.org/10.1109/ACCESS.2023.3268866)
[![PyPI version](https://img.shields.io/pypi/v/bellatrex.svg)](https://pypi.org/project/bellatrex/)
[![codecov](https://codecov.io/github/KlestDedja/Bellatrex/branch/main/graph/badge.svg)](https://app.codecov.io/github/KlestDedja/Bellatrex)
[![Roadmap](https://img.shields.io/badge/roadmap-open-blue)](./ROADMAP.md)



# Bellatrex: Explain your Random Forest predictions

Bellatrex is a Python library designed to generate concise, interpretable, and visually appealing explanations for predictions made by Random Forest models. The name says it all: Bellatrex stands for **B**uilding **E**xplanations through a **L**ocal**L**y **A**ccura**T**e **R**ule **EX**tractor.

Curious about the details and inner mechanisms of Bellatrex? Check out [our paper](https://ieeexplore.ieee.org/abstract/document/10105927) and jump into the [reproducibility branch](https://github.com/KlestDedja/Bellatrex/tree/archive/reproduce-Dedja2023) to dive into the experiments.

## Table of Contents

- [How Bellatrex works](#how-bellatrex-works)
- [Supported models and tasks](#supported-models-and-tasks)
- [Installation](#set-up)
- [Quickstart](#quickstart)
- [API Overview](#api-overview)
- [Support and Contributions](#support-and-contributions)
- [References](#references)

## How Bellatrex works

When explaining a prediction for a specific test instance, Bellatrex:
1) pre-selects a subset of the rules used to make the prediction;
2) creates a vector representation of such rules and (optionally) projects them into a low-dimensional space
3) clusters such representations to pick a rule from each cluster to explain the instance prediction.
4) Shows the selected rule through visually appealing plots, and the tool's GUI allows users to explore similar rules to those extracted.

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/KlestDedja/Bellatrex/blob/main/app/illustration-Bellatrex.png?raw=true" alt="Bellatrex image" width="90%"/>
    </td>
  </tr>
  <tr>
    <td align="left">
      <em>Overview of Bellatrex, starting from top left, proceeding clockwise, we reach the output with related explanations on the bottom left. </em>
    </td>
  </tr>
</table>


## Supported models and tasks

The current support of Bellatrex focuses on Random Forest models implemented via `scikit-learn` and `scikit-survival`:

- Classification tasks and multi-label classification via `RandomForestClassifier`
- Regression tasks and multi-target regression via `RandomForestRegressor`
- Survival Analysis (time-to-event predictions with censoring) via `RandomSurvivalForest`


# Set-up

To install Bellatrex, create an Anaconda environment using Python 3.12, the standard
development and quick-CI version:

```
conda create -n bellatrex python=3.12
```

Then install the package:

```
pip install bellatrex
```

Bellatrex remains compatible with, and release-tested on, Python 3.10 through 3.14.

If this step fails and you don't find a solution immediately, please [open an issue](https://github.com/KlestDedja/Bellatrex/issues). In the meantime, you can also try to [clone](https://github.com/KlestDedja/Bellatrex) the repository manually.


## Interactive GUI mode

The NiceGUI-based interactive frontend is installed by default with:
```
pip install bellatrex
```

The legacy command still works as a compatibility alias:
```
pip install bellatrex[gui]
```

or you can install the browser GUI dependency manually:
```
pip install nicegui
```

**Note:** Bellatrex installs the browser GUI everywhere, and adds native window support automatically where the platform backend is available. On Windows with Python 3.14, Bellatrex currently falls back to browser-based GUI mode because the native `pythonnet` backend does not yet support that Python version. When running Bellatrex with the GUI for multiple test samples, the program will generate an interactive window in your browser. The user can explore the generated rules by clicking on the corresponding representation. To show the Bellatrex explanation for the next sample, interact with the interface and wait until Bellatrex generates the explanation for the new sample.

# Quickstart

The following example explains individual predictions from a `RandomForestClassifier` on the breast cancer dataset:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from bellatrex import BellatrexExplain

# 1. Train a Random Forest
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)

# 2. Fit the Bellatrex explainer on training data
explainer = BellatrexExplain(clf).fit(X_train, y_train)

# 3. Explain the prediction for the first test instance (method-chainable)
explainer.explain(X_test, idx=0).plot_overview()           # cluster overview plot
explainer.explain(X_test, idx=0).plot_visuals()            # rule-level detail plot
explainer.explain(X_test, idx=0).create_rules_txt()        # save explanation as text
explainer.explain(X_test, idx=0).print_rules_txt()         # print explanation to console
```

For a step-by-step walkthrough covering all supported tasks (regression, survival analysis, multi-label classification), see [`tutorial.ipynb`](https://github.com/KlestDedja/Bellatrex/blob/main/tutorial.ipynb).

# API Overview

## `BellatrexExplain`

The main class for generating explanations.

```python
from bellatrex import BellatrexExplain

explainer = BellatrexExplain(
    clf,                          # trained (or untrained) RF / RSF model, or a packed dict
    set_up="auto",                # task type: "auto", "binary", "regression", "survival",
                                  #            "multi-label", "multi-target"
    p_grid={                      # hyperparameter search grid
        "n_trees":    [0.6, 0.8, 1.0],   # fraction (or count) of trees to pre-select
        "n_dims":     [2, None],          # PCA dimensions; None = no projection
        "n_clusters": [1, 2, 3],          # number of explanation rules to return
    },
    proj_method="PCA",            # dimensionality reduction: "PCA" (default) or None
    dissim_method="rules",        # tree dissimilarity metric
    feature_represent="weighted", # feature representation strategy
    n_jobs=1,                     # parallelism (experimental)
    verbose=0,                    # verbosity: 0 = silent, higher = more output
)
```

| Method | Description |
|--------|-------------|
| `.fit(X_train, y_train)` | Fit the explainer (trains the RF if not yet fitted). Returns `self`. |
| `.explain(X_test, idx)` | Run Bellatrex for the sample at positional index `idx`. Returns `self`. |
| `.plot_overview(plot_gui=False)` | Cluster overview: representations, selected rules, and tree plots. |
| `.plot_visuals(...)` | Rule-level detail plot with optional prediction distribution and confidence bands. Single-output tasks only. |
| `.create_rules_txt(out_dir, out_file)` | Write the explanation rules to a `.txt` file. Returns the file paths. |
| `.print_rules_txt(out_dir, out_file)` | Print the explanation rules to stdout. |

## `pack_trained_ensemble`

Converts a trained `scikit-learn` / `scikit-survival` forest into a compact dictionary
format, useful for serialisation or passing externally trained models to Bellatrex.

```python
from bellatrex import pack_trained_ensemble

clf_packed = pack_trained_ensemble(clf)  # clf must already be fitted
explainer = BellatrexExplain(clf_packed).fit(X_train, y_train)
```

## Support and Contributions

Bellatrex is an open-source project that was initially developed from research funding by [Flanders AI](https://www.flandersai.be/en). Since the end of that funding period, the project has been maintained through volunteer work, but there is always exciting work ahead: new features, performance improvements, tests for robustness... if you find Bellatrex useful or believe in its goals, there are several meaningful ways you can help support its ongoing development:

- 🐛 **Test and Report Issues:** if you encounter any bugs, inconsistencies, or simply find areas for improvement, open an [issue](https://github.com/KlestDedja/Bellatrex/issues) and share example code and error traces.
- ⭐ **Give a star Bellatrex:** it will make the project more visible to others and motivate ongoing voluntary development.
- 🔧 **Contribute code:** open a PR directly for small fixes, or open an issue first to discuss larger changes. See [ROADMAP.md](./ROADMAP.md) for planned features.

## References

Please cite the following paper if you are using Bellatrex:

> Dedja, K., Nakano, F.K., Pliakos, K. and Vens, C., 2023. BELLATREX: Building explanations through a locally accurate rule extractor. Ieee Access, 11, pp.41348-41367.

<pre>
@article{dedja2023bellatrex,
  title={BELLATREX: Building explanations through a locally accurate rule extractor},
  author={Dedja, Klest and Nakano, Felipe Kenji and Pliakos, Konstantinos and Vens, Celine},
  journal={Ieee Access},
  volume={11},
  pages={41348--41367},
  year={2023},
  publisher={IEEE}
}
</pre>
