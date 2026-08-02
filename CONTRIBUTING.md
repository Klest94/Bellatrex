# Contributing to Bellatrex

## Development environment

Create a virtual environment (for example, named `.venv`) and activate it. For a lightweight setup
that can run the test suite:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e . pytest
```

For the complete maintainer environment, including coverage, notebook, formatting,
linting, and type-checking tools, install the development extra instead:

```bash
python -m pip install -e ".[dev]"
```

Installing the development extra is recommended but not required to open a pull request;
GitHub Actions runs the mandatory checks before changes can merge. In VS Code, select
the `.venv` interpreter. The repository recommends the Python, Pylance, Black, Ruff,
and mypy extensions. Pylance provides language navigation and completion; mypy is the
project's type-checking authority.

## Quality checks

Run the same checks locally that GitHub runs:

```bash
black --check app tests tutorial.py
ruff check app tests tutorial.py
mypy app/bellatrex
pytest -q -m "not gui"
```

To apply safe automatic formatting and lint fixes:

```bash
black app tests tutorial.py
ruff check --fix app tests tutorial.py
```

Black owns formatting, Ruff owns linting and import sorting, and mypy owns type
checking. Their shared configuration lives in `pyproject.toml`.

The development extra also installs pre-commit. Enable the repository hooks once per
clone, then run them against the whole tree whenever needed:

```bash
pre-commit install
pre-commit run --all-files
```

The hooks run the same Black, Ruff, mypy, and non-GUI pytest checks listed above. They
use the active development environment, which keeps their dependency versions aligned
with `pyproject.toml`.

## Dead-code policy

Ruff blocks unused imports and unused local variables. Broader tools such as Vulture
are not blocking checks because Python callbacks, decorators, public APIs, and
dynamically registered GUI routes can look unused to static analysis. Use Vulture,
coverage reports, and editor reference searches as manual review aids before a
release or large refactor, and verify each candidate before deleting it.
