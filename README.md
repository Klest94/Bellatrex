<a name="logo-anchor"></a>
<p align="center">
<img src="https://github.com/KlestDedja/Bellatrex/blob/main/app/bellatrex-logo.png?raw=true" alt="Bellatrex Logo" width="60%"/>
</p>

[![Python Versions](https://img.shields.io/pypi/pyversions/bellatrex)](https://pypi.org/project/bellatrex/)
[![Downloads](https://static.pepy.tech/badge/bellatrex)](https://pepy.tech/project/bellatrex)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/KlestDedja/Bellatrex/blob/main/LICENSE.txt)
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
- [Installation](#installation)
- [Interactive GUI mode](#interactive-gui-mode)
- [Quickstart](#quickstart)
- [Support and Contributions](#support-and-contributions)
- [References](#references)

## How Bellatrex works

When explaining a prediction for a specific test instance, Bellatrex:

1) pre-selects a subset of the rules used to make the prediction

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


## Installation

Bellatrex supports Python 3.10 and newer and is available on PyPI:

```bash
pip install bellatrex
```

Bellatrex remains compatible with, and release-tested on, Python 3.10 through 3.14.

If this step fails and you don't find a solution immediately, please [open an issue](https://github.com/KlestDedja/Bellatrex/issues). In the meantime, you can also try to [clone](https://github.com/KlestDedja/Bellatrex) the repository manually.


## Interactive GUI mode

The NiceGUI-based interactive frontend is installed by default and can be opened with
`plot_overview(plot_gui=True)`. It is still experimental: responsiveness can be slow, and
opening several sample explanations at once is not yet fully supported.

## Quickstart

A step-by-step walkthrough covering every supported task is available in
[`tutorial.ipynb`](https://github.com/KlestDedja/Bellatrex/blob/main/tutorial.ipynb). A
script version for local runs and automated checks is available as
[`tutorial.py`](https://github.com/KlestDedja/Bellatrex/blob/main/tutorial.py).

## Support and Contributions

Bellatrex is an open-source project that was initially developed from research funding by [Flanders AI](https://www.flandersai.be/en). Since the end of that funding period, the project has been maintained through volunteer work, but there is always exciting work ahead: new features, performance improvements, tests for robustness... if you find Bellatrex useful or believe in its goals, there are several meaningful ways you can help support its ongoing development:

- 🐛 **Test and Report Issues:** if you encounter any bugs, inconsistencies, or simply find areas for improvement, open an [issue](https://github.com/KlestDedja/Bellatrex/issues) and share example code and error traces.
- ⭐ **Give a star Bellatrex:** it will make the project more visible to others and motivate ongoing voluntary development.
- 💡 **Create your own feature**: have an idea for a useful addition? Open a PR for self-contained changes, or open an issue first to discuss possible designs, usefulness, and alignment with the project.
- 🔧 **Contribute code:** open a PR directly for small fixes, or open an issue first to discuss larger changes. See [CONTRIBUTING.md](./CONTRIBUTING.md) for the local quality checks and [ROADMAP.md](./ROADMAP.md) for planned features.

## References

If you are using Bellatrex, please cite the following paper:

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
