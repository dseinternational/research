# Copilot Instructions

This file provides guidance to GitHub Copilot when working with code in this repository.

> **Keep in sync:** `CLAUDE.md`, `AGENTS.md`, and `.github/copilot-instructions.md` must contain identical guidance (except for the first heading). When updating one, update all three.

## Overview

Shared library repository for [Down Syndrome Education International (DSE)](https://www.down-syndrome.org/) research projects. The primary artifact is `dse-research-utils`, a Python utility package (`src/python/`). A .NET utilities area (`src/dotnet/`) is a future placeholder.

## Repositories using these libraries

Current projects using these libraries include:

- [dseinternational/language-reading-predictors](https://github.com/dseinternational/language-reading-predictors)
- [dseinternational/vocabulary-growth](https://github.com/dseinternational/vocabulary-growth)
- [dspopulations/us-birth-certificates](https://github.com/dspopulations/us-birth-certificates)

To support developing across these repositories simultaneously, we typically check these out relative to this project as follows:

- ../language-reading-predictors
- ../vocabulary-growth
- ../../dspopulations/us-birth-certificates

## Commands

### Spellcheck
```bash
npm ci
npm run spellcheck      # runs cspell over all *.md files
```

### Python — environment
```bash
conda env create -f environment.yml   # create (Python 3.14, conda required — not pip/venv)
conda activate dse-research
```

### Python — build
```bash
cd src/python
python -m build         # builds wheel via hatchling
```

### Python — lint and format
```bash
cd src/python
ruff check .            # lint
ruff check . --fix      # lint with auto-fix
ruff format .           # format
```

### Python — tests
```bash
pytest                                           # full suite
pytest path/to/test_file.py                      # single file
pytest path/to/test_file.py::test_function_name  # single test
```

## Architecture

`src/python/src/dse_research_utils/` is structured by domain:

- **`environment/`** — system info, execution context; `init_workbook()` / `init_script()` for notebook/script setup
- **`math/`** — constants (`EPSILON`, etc.)
- **`metadata/`** — package version introspection
- **`ml/`** — ML utilities (placeholder)
- **`plot/`** — matplotlib/ArviZ plotting helpers; constants follow `FIGSIZE_XS`, `COLOUR_BLUE`, `DPI_NOTEBOOK` naming
- **`statistics/`** — descriptive stats, credible/confidence intervals, PyMC models and sampling presets

All `__init__.py` files are empty — no re-exports. Use fully-qualified absolute imports everywhere (e.g. `from dse_research_utils.math.constants import EPSILON`).

Bayesian sampling presets in `statistics/models/sampling.py`, selected via `get_sampling_configuration(config)`:
- `dev` / `development` — 2 chains × 500 draws, `target_accept=0.85` (fast iteration)
- `test` / `testing` — 4 chains × 2000 draws, `target_accept=0.90`
- `reporting` / `report` / `rep` — 6 chains × 6000 draws, `target_accept=0.95`

## Python Conventions

**License header** — every source file starts with:
```python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
```

- **Naming**: `snake_case` for files/functions/variables; `UPPER_CASE` for constants grouped by domain (e.g. `FIGSIZE_XS`, `COLOUR_BLUE`, `DPI_NOTEBOOK`)
- **Type hints**: always on function signatures
- **Type unions**: `X | Y` syntax (not `Union[X, Y]`)
- **Docstrings**: NumPy-style (`Parameters`, `Returns`, `---` separators); dataclass fields use attribute docstrings (bare string literals after the field declaration)
- **`print`**: import `from rich import print` to override built-in
- **Imports**: fully-qualified absolute imports only — no relative imports, no `__init__.py` re-exports
- **Dataclasses**: stdlib `@dataclass`; use `__post_init__` for validation
- **Plot functions**: create figure → render → optionally save to `output_dir` as `.png` (300 DPI) and `.svg` → `return plt.gcf()`
- **Notebooks**: call `init_workbook()` at top (prints environment info); scripts call `init_script()` (silent setup); both apply the default matplotlib style via `set_matplotlib_default_style()`

Ruff config (in `src/python/pyproject.toml`): line-length 120, target Python 3.14, rules: `F`, `E`, `W`, `I`, `UP`, `B`, `SIM`, `RUF`.

## .NET

SDK pinned to **10.0.200** (`rollForward: latestMinor`) via `global.json`. Test runner is `Microsoft.Testing.Platform`. NuGet sources: `nuget.org` (all packages) and `dseinternational` (`https://nuget.pkg.github.com/dseinternational/index.json`, `DSE.*` packages only) — package source mapping enforced with `<clear />`. No projects exist yet.
