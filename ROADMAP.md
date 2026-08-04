# Roadmap

This file sketches the direction of the project. It is a guidepost, and priorities can
shift over time.


## Immediate goals (v0.4.x patches)

- Increase line coverage from roughly 67% to above 80%. The largest gains are still
  available in the NiceGUI backend and `utilities.py`.
- Stabilize the NiceGUI explorer, especially responsiveness and handling of multiple
  explanation windows.


## Recently completed

- Released v0.4.0 with a NiceGUI frontend and Python 3.13 and 3.14 support.
- Published `BellatrexExplain`, `pack_trained_ensemble`, and `predict_helper` as top-level
  imports.
- Updated and simplified `tutorial.ipynb` and `tutorial.py` for the 0.4.0 API.
- Added consistent local and Codecov coverage reporting and repaired release-facing links.
- Standardized quick CI and development tooling on Python 3.12 while retaining the
  Python 3.10–3.14 release compatibility matrix.
- Enforced Black, Ruff, and mypy in quick CI and on pull requests targeting `main`,
  and added a headless tutorial smoke test.
- Added local pre-commit hooks for Black, Ruff, mypy, and the non-GUI pytest suite.
- Added automated README link checks for documentation changes and as a required
  release gate.


## Mid-term ideas (future releases)

### New features

- Bellatrex already supports a model agnostic approach (building a surrogate random forest model). Test thoroughly, update related APIs, and documentation

- Enhance vector representation by including leaf predictions in the representation of the trees. The resulting representation could have $d+q$ dimensions, where $q$ is the output dimensionality and $d$ is the input dimensionality. Currently only feature splits are used to create the tree representation.
The weight of the extra $q$ dimensions should be controlled by a new parameter.

- Enhance Bellatrex explanations for multi-output tasks, enable users to:
    - select a subset of targets to run explanations for;
    - select a (single) target to run ``plot_visuals()``

- Impose best practices for code API stability over several version: raise `DeprecationWarning` when functions are being dropped


### Type safety

- Add type hints throughout the codebase, starting with the public API in
  `bellatrex_explain.py` and `utilities.py`, where annotation coverage remains limited.
- Use `from __future__ import annotations` to keep annotations compatible with
  older Python versions while writing modern syntax.

### Refactoring

- Break up functions that have grown too long. Primary targets:
    - `plot_preselected_trees()` in `utilities.py` (~268 lines): extract coloring,
      legend, and annotation logic into focused helpers.
    - `explain()` in `bellatrex_explain.py` (~186 lines): move grid search into its
      own internal method.
    - `plot_rules()` in `visualization.py` (~237 lines): separate colormap and
      subplot construction.
- Consider splitting `BellatrexExplain` responsibilities: a lean core class,
  a `RulesExporter` for file I/O, and visualization helpers. The current class
  handles fitting, tuning, plotting, and text export simultaneously.
- Replace the 12-parameter constructor with a config/dataclass object for
  infrequently-used options (`ys_oracle`, `force_refit`, `verbose`, …), keeping the
  common-case API simple.

### Documentation

- Add docstrings to all internal functions that currently have none (e.g.,
  `frmt_pretty_print`, `rule_to_file`, `_validate_p_grid`).
- Set up auto-generated API docs with Sphinx, Read the Docs, or Zensical.

### Testing

- Increase unit test coverage for `utilities.py` (currently the largest file with
  the fewest dedicated tests).
- Add edge-case tests: single-feature datasets, all-identical predictions, NaN
  inputs, feature mismatch between fit and explain.

### Drop dependency on pandas
- Replace internal `pandas.DataFrame` usage with:
  - NumPy arrays where appropriate.
  - A lightweight `Batch` object for structured tabular data.
- Ensure public API doesn’t break for users who rely on pandas-like inputs
- If pandas dependency cannot be avoided, add compatibility with `polars`


### Wider compatibility and support
- Add explainability features for more model types (Extra Trees, gradient-boosted trees, etc.).
- Provide a simple web demo (Streamlit or FastAPI).
- Improve documentation:
    - interactive notebooks
    - a full documentation website


## How to contribute

Open a PR directly, either for small fixes or for suggesting new features and roadmap items.

---
_Last updated: 2026-08-02_
