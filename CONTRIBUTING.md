# Contributing to Bellatrex

## Development environment

Create a virtual environment named `.venv`, activate it, and install Bellatrex with
its development tools:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

In VS Code, select that `.venv` interpreter. The repository recommends the Python,
Pylance, Black, Ruff, and mypy extensions. Pylance provides language navigation and
completion; mypy is the project's type-checking authority.

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

## Dead-code policy

Ruff blocks unused imports and unused local variables. Broader tools such as Vulture
are not blocking checks because Python callbacks, decorators, public APIs, and
dynamically registered GUI routes can look unused to static analysis. Use Vulture,
coverage reports, and editor reference searches as manual review aids before a
release or large refactor, and verify each candidate before deleting it.
