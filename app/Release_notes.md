# Changelog

## [0.2.3] - 2024-08-20
### Refactored
- Cleaned some code and dropped unused files and functions (still ongoing)
### Enhanced
- Further streamlined loading of pre-trained models.


## [0.2.2] - 2024-08-12
### Fixed
- Updated README.md with absolute paths instead of relative ones.
- Updated Python requirements, show explicitly under ```setup.classifiers`` that the package is compatible with any `python >= 3.9`.
### Enhanced
- Improved plotting format in plot_visuals() and inline printing in general, with custom function `frmt_pretty_print`.
- Streamlined storing of trained (Random Forest) models and loading of pre-trained models (Random Forest)

## [0.2.1] - 2024-09-02
### Fixed
- Fixed bug while calling plot_visuals() with `preds_distr = None` and `conf_level != None`, sometimes the wrong axis was being modified.

## [0.2.0] - 2024-09-01
### Refactored
- Refactor the explain() method to accomodate for plot_overview() and plot_visuals(). Separating tasks and making method chaining possible.
- Refactor code to improve performance (e.g. not storing entire test set under `self.X` if not necessary).
### Enhanced
- Improved plotting format in plot_visuals() and inline printing in general, with custom function `frmt_pretty_print`.

## [0.1.4] - 2024-07-28
### Added
- Added regression and multi-label classification datasets to bellatrex.datasets, added feature names to binary dataset.
- Added enhanced visualization output, compatible with single-output predictions.

### Refactored
- Refactored code in the BellatrexExplain class, and gui_plots_code script.
- Refactored code in the visualisation script for compatibiility with BellatrexExplain.

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