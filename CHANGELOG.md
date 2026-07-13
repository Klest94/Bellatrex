# Changelog

## [Unreleased]

### Documentation
- Aligned the roadmap, README, changelog, and script tutorial with the released 0.4.0 API
  and the simplified notebook workflow.

## [0.4.0] - 2026-06-08

This is a major release with a completely revised user interface: DearPyGUI was replaced
with NiceGUI for easier maintenance. Bellatrex now supports Python 3.10 through 3.14;
support for Python 3.9 has been dropped.

### Added
- `pack_trained_ensemble` and `predict_helper` are now importable directly from `bellatrex`
  (e.g. `from bellatrex import pack_trained_ensemble`).
- `BellatrexExplain` now has a `__repr__` method that shows key parameters.
- `BellatrexExplain.__init__` initialises `sample`, `tuned_method`, `sample_index`, and
  `surrogate_pred_str` to `None`, so calling visualization methods before `explain()` raises
  a clear `ValueError` rather than an `AttributeError`.

### Changed
- **API**: `explain()` now raises `TypeError` with a helpful message when `X` is not a pandas
  DataFrame, instead of failing silently with an `AttributeError`.
- `verbose=0` (the default) is now fully silent.  Fitting and status messages are only printed
  at `verbose >= 1`, consistent with the scikit-learn convention.
- The mutable default `p_grid={}` in `__init__` is replaced with `p_grid=None`; the default
  grid is defined as the class-level constant `_DEFAULT_P_GRID`.
- `ys_oracle` constructor parameter is now correctly stored (`self.ys_oracle = ys_oracle`
  instead of being silently dropped).
- `n_jobs > 1` no longer emits a spurious warning; thread-based parallelism just runs.
- `create_rules_txt` / `print_rules_txt`: path-resolution logic simplified.  `out_dir=None`
  resolves to `$BELLATREX_EXPLAIN_DIR` (if set) or `<cwd>/explanations-output`; any other
  relative path is resolved directly from the current working directory.
  The env var `BELLATREX_EXPLAIN_DIR` is now actually respected (it was previously commented
  out).
- Dead private method `_pick_runtime_dir` removed; a new `_resolve_output_dir` helper
  centralises the two-method path logic into one place.
- `tutorial.ipynb` and `tutorial.py` updated to use the new top-level imports.

## [0.3.1] - 2025-10-25

### Enhanced
- Test coverage has been extended, and `codecov` platform is being used for reporting.
  Its reports are synchronized with local pytest runs.
- Improved GitHub workflow actions, which now include `cross-platform`,
  `cross-version`, `coverage`, `CodeQL`, and automatic `release` to PyPi.


## [0.3.0] - 2025-07-24

### Enhanced
Revamped README.md, which now includes project badges (build status, version,
license, etc.), clearer project description, and usage instructions.

A CI pipeline is now running with the first workflows (for example install checks,
linting, and test execution), which are now functioning correctly. A first version
of dependency checks has been set in place.

An initial test coverage is introduced, with first set of pytest running successfully.

### Fixed

Many `DeprecationWarning` warnings have been resolved (mainly within numpy and
matplotlib), making the package more future-proof.


## [0.2.3] - 2024-11-15

### Fixed
- Fixed bug when running `set_up=multi-label` with `EnsembleWrapper`
- Updated dependency constraints with `dearpygui`; newest versions are not
  supported (`lighttheme = create_theme_imgui_light()` raises an error).

### Refactored
- Cleaned some code and dropped unused files and functions (still ongoing)
### Enhanced
- Further streamlined loading of pre-trained models with `EnsembleWrapper` class.
- Improved compatibility between `EnsembleWrapper` and GUI interface.



## [0.2.2] - 2024-08-12
### Fixed
- Updated README.md with absolute paths instead of relative ones.
- Updated Python requirements, show explicitly under ```setup.classifiers`` that the package is compatible with any `python >= 3.9`.
### Enhanced
- Improved plotting format in plot_visuals() and inline printing in general, with custom function `frmt_pretty_print`.
- Streamlined storing of trained (Random Forest) models and loading of pre-trained
  models (Random Forest).

## [0.2.1] - 2024-09-02
### Fixed
- Fixed bug while calling plot_visuals() with `preds_distr = None` and
  `conf_level != None`; sometimes the wrong axis was being modified.

## [0.2.0] - 2024-09-01
### Refactored
- Refactor the explain() method to accomodate for plot_overview() and
  plot_visuals(), separating tasks and making method chaining possible.
- Refactor code to improve performance (e.g. not storing entire test set under `self.X` if not necessary).
### Enhanced
- Improved plotting format in plot_visuals() and inline printing in general, with custom function `frmt_pretty_print`.

## [0.1.4] - 2024-07-28
### Added
- Added regression and multi-label classification datasets to bellatrex.datasets, added feature names to binary dataset.
- Added enhanced visualization output, compatible with single-output predictions.

### Refactored
- Refactored code in the BellatrexExplain class, and gui_plots_code script.
- Refactored code in the visualization script for compatibiility with BellatrexExplain.

## [0.1.3] - 2024-07-24
### Fixed
- Fixed version file (it was still problematic), made it a .txt file. Updated MANIFEST.in accordingly.

## [0.1.2] - 2024-07-24
### Tested
- Test PyPi released successfully.

## [0.1.1] - 2024-07-24
### Fixed
- Fixed a bug with the version file that prevented the package from being installed properly. Moved the version file and updated the related path imports.

## [0.1.0] - 2024-07-23
### Added
- Initial release, including a draft version of the Graphical User Interface.
