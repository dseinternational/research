# Copilot Instructions

This is the **Down Syndrome Education International (DSE) Research** monorepo. It hosts open-access research datasets and a shared Python utility library (`dse-research-utils`) used across research projects.

## Repository Structure

```
research/
├── projects/                        # Self-contained research projects (data + analysis)
│   └── reading-language-intervention-trial/
│       ├── data/rli-trial-data.csv          # Trial dataset
│       └── data/rli-trial-data-variables.csv # Variable codebook
└── src/
    ├── python/                      # dse-research-utils package (v0.1.0)
    │   └── src/dse_research_utils/
    │       ├── environment/         # System info and execution context
    │       ├── math/                # Constants (EPSILON, etc.)
    │       ├── metadata/            # Package version introspection
    │       ├── ml/                  # ML utilities (placeholder)
    │       ├── plot/                # matplotlib/ArviZ plotting helpers
    │       └── statistics/          # Descriptive stats, intervals, PyMC models
    └── dotnet/                      # Future .NET utilities (TODO placeholder)
```

Each research project under `projects/` is self-contained with its own `README.md`, `LICENSE`, and `data/` folder.

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

**Docstrings**: NumPy-style with `Parameters`, `Returns`, and `---` separators.

**Type union syntax**: Use `X | Y` (not `Union[X, Y]`) — e.g. `list[float] | np.ndarray | pd.Series`.

**`print`**: Files import `from rich import print` to override the built-in.

**Dataclasses**: `@dataclass` from stdlib (`dataclasses`) and from `attr` are both used. `statistics/models/` uses `attr`; `statistics/models/data.py` uses stdlib — follow the convention of the file being edited.

**Notebook/script setup**: Call `dse_research_utils.environment.setup.init_workbook()` at the top of Jupyter notebooks (prints environment info) or `init_script()` for scripts (silent setup). Both apply the default matplotlib style via `set_matplotlib_default_style()`.

**Bayesian sampling presets** (in `statistics/models/sampling.py`):
- `dev` — 2 chains × 1000 draws (fast iteration)
- `reporting` / `rep` — 6 chains × 6000 draws, higher `target_accept`

## Data Conventions

Research datasets live at `projects/<project-name>/data/`. Every dataset CSV is accompanied by a `*-variables.csv` codebook describing each variable. Data may carry a separate license (e.g. CC BY 4.0) distinct from the code license.
