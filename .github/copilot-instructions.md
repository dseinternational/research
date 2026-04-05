# Copilot Instructions

This is the **Down Syndrome Education International (DSE) Research** shared library repository. It hosts `dse-research-utils`, a Python utility package used across DSE research projects (each in its own repository).

## Repository Structure

```
research/
└── src/
    ├── python/                      # dse-research-utils shared package (v0.1.0)
    │   └── src/dse_research_utils/
    │       ├── environment/         # System info and execution context
    │       ├── math/                # Constants (EPSILON, etc.)
    │       ├── metadata/            # Package version introspection
    │       ├── ml/                  # ML utilities (placeholder)
    │       ├── plot/                # matplotlib/ArviZ plotting helpers
    │       └── statistics/          # Descriptive stats, intervals, PyMC models
    └── dotnet/                      # Future .NET utilities (TODO placeholder)
```

## Commands

### Spellcheck
```bash
npm ci
npm run spellcheck      # runs cspell over all *.md files
```

### Python package build
```bash
cd src/python
python -m build         # builds wheel via hatchling
```

### Python lint and format
```bash
cd src/python
ruff check .              # lint
ruff check . --fix        # lint with auto-fix
ruff format .             # format
```

### Python tests (pytest — no tests written yet)
```bash
pytest                                          # full suite
pytest path/to/test_file.py                     # single file
pytest path/to/test_file.py::test_function_name # single test
```

## Python Environment

Use **conda**, not pip or venv — the VS Code workspace is configured to use conda by default.

```bash
conda env create -f environment.yml   # create environment
conda activate dse-research           # activate it
```

The conda environment targets **Python 3.14**. The `dse-research-utils` package also requires `python >= 3.14`.

Key dependencies: PyMC ≥ 5.28, ArviZ ≥ 0.23, JAX, pandas ≥ 3.0, polars, numpy ≥ 2.4, scipy ≥ 1.17, statsmodels, pingouin, scikit-learn ≥ 1.8, PyTorch, Optuna, DuckDB ≥ 1.5, matplotlib ≥ 3.10, Jupytext.

## .NET

SDK version is pinned to **10.0.200** (`rollForward: latestMinor`) via `global.json`. Test runner is `Microsoft.Testing.Platform`. No projects exist yet.

NuGet has two sources (defined in `NuGet.config`):
- `nuget.org` — all packages
- `dseinternational` (`https://nuget.pkg.github.com/dseinternational/index.json`) — `DSE.*` packages only

Package source mapping is enforced with `<clear />`, so only these two sources are active.

## Python Conventions

**License header** — every source file starts with:
```python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
```

**Naming**: `snake_case` for all files, functions, and variables. Constants are `UPPER_CASE` and grouped by domain (e.g. `FIGSIZE_XS`, `COLOUR_BLUE`, `DPI_NOTEBOOK`).

**Type hints**: always used on function signatures.

**Docstrings**: NumPy-style with `Parameters`, `Returns`, and `---` separators. Dataclass fields use attribute docstrings (bare string literals after the field declaration).

**Type union syntax**: Use `X | Y` (not `Union[X, Y]`) — e.g. `list[float] | np.ndarray | pd.Series`.

**`print`**: Files import `from rich import print` to override the built-in.

**Imports**: Use fully-qualified absolute imports for internal modules — e.g. `from dse_research_utils.math.constants import EPSILON`. No relative imports. No `__init__.py` re-exports; all subpackage `__init__.py` files are empty.

**Dataclasses**: stdlib `@dataclass` from `dataclasses` — used throughout `statistics/models/`. Use `__post_init__` for validation.

**Ruff config** (in `src/python/pyproject.toml`): line-length 120, target Python 3.14, lint rules: `F`, `E`, `W`, `I`, `UP`, `B`, `SIM`, `RUF`.

**Plot functions**: Create a figure, render the plot, optionally save to `output_dir` as both `.png` (300 DPI) and `.svg`, then `return plt.gcf()`.

**Notebook/script setup**: Call `dse_research_utils.environment.setup.init_workbook()` at the top of Jupyter notebooks (prints environment info) or `init_script()` for scripts (silent setup). Both apply the default matplotlib style via `set_matplotlib_default_style()`.

**Bayesian sampling presets** (in `statistics/models/sampling.py`):
- `dev` — 2 chains × 1000 draws (fast iteration)
- `reporting` / `rep` — 6 chains × 6000 draws, higher `target_accept`


