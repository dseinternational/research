# Migrating a consuming repository from conda to uv

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

This repository has moved from the hybrid conda + pip environment model to a `uv` workflow. This guide covers what changes for the repositories that consume `dse-research-utils`, and the order to do it in. See [#86](https://github.com/dseinternational/research/issues/86) for the investigation behind the change.

## Why the hybrid model existed, and why it no longer does

The conda-forge layer existed to supply a C toolchain and BLAS for PyMC's PyTensor backend, which PyPI could not. [PyMC 8318](https://github.com/pymc-devs/pymc/pull/8318) made Numba the default compile backend, so no compiler or BLAS is needed, and pip became the recommended install route. Every package in the former compiled core now ships a CPython 3.14 wheel for linux-x86_64, macOS-arm64 and win-amd64.

Two consequences worth knowing before you start:

- **Windows contributors no longer need WSL.** `jaxlib` — the package that blocked native Windows on conda-forge — publishes `cp314-win_amd64` wheels on PyPI, so the whole stack installs natively.
- **Intel macOS is no longer supported.** This is upstream's decision, not ours: numba publishes no macOS x86_64 wheels at all, and `shap` pins `numba<0.63` there. Apple Silicon, Linux and Windows are unaffected.

## What replaces the canonical core

The compiled core is no longer a YAML block copied into every repo's `environment.yml` and policed by `dse-check-env`. It is now declared in `dse-research-utils`' own `pyproject.toml`, so a consuming repo inherits the floors transitively just by depending on the library. Drift is structurally impossible rather than merely detectable.

Everything the core used to pin is now either a base dependency or an extra:

| Former conda core / add-on               | Now comes from                                    |
| ---------------------------------------- | ------------------------------------------------- |
| python, numpy, scipy, pandas, pyarrow, numba, matplotlib, scikit-learn, statsmodels | base `dependencies` |
| pymc, pytensor, nutpie, arviz\*, preliz, xarray | base `dependencies` |
| jax, numpyro                             | `jax` extra                                       |
| lightgbm, xgboost, shap                  | `boosting` extra                                  |
| polars, duckdb, pyreadstat               | `columnar` extra                                  |
| h5py, h5netcdf, zarr                     | `storage` extra                                   |
| graphviz (bindings), networkx            | `graphs` extra — **plus** the system `dot` binary |
| seaborn                                  | `viz` extra                                       |
| jupyter, jupytext                        | `notebook` extra                                  |
| dcor                                     | `dependence` extra                                |
| optuna, optuna-integration               | `tuning` extra                                    |
| orjson, tabulate                         | `io` extra                                        |

Three changes to watch for:

1. **`nutpie` must not be dropped.** PyMC 6 declares it as `pymc[nutpie]`, not a base dependency, and auto-selects it as the default NUTS sampler when present. It is in `dse-research-utils`' base dependencies so every repo keeps the same sampler, but do not "tidy it away".
2. **`jax` and `numpyro` are now opt-in.** The old core installed them everywhere. If your repo samples via NumPyro or JAX, add the `jax` extra explicitly.
3. **Extras are grouped, not à la carte.** Taking `columnar` for `pyreadstat` also brings `polars` and `duckdb`. That is deliberate — the groups keep the Arrow-backed data layer moving in lockstep — but it means a repo may install a package it does not import.

## Per-repository steps

The system Graphviz binary is not a Python package. Repos using `dse_research_utils.plot.graphs` still need `brew install graphviz`, `apt install graphviz`, or `winget install Graphviz.Graphviz`.

### 1. Replace the environment files

Delete `environment.yml`, and — in `vocabulary-growth` — also `conda-lock.yml`, `requirements-pip.lock` and `requirements-lock-tool.txt`. Add `.python-version` containing `3.14`.

Move everything from the old `pip:` block into the repo's existing `pyproject.toml`, and move the conda dependencies into the right extra on `dse-research-utils`:

```toml
dependencies = [
    "dse-research-utils[boosting,columnar,dependence,io,notebook,tuning,viz]"
]

[tool.uv.sources]
dse-research-utils = { git = "https://github.com/dseinternational/research.git", tag = "vX.Y.Z", subdirectory = "src/python" }
```

Suggested extras per repo, based on what each currently declares:

- **language-reading-predictors** — `boosting`, `columnar` (for `pyreadstat`), `dependence`, `graphs`, `io`, `notebook`, `tuning`, `viz`, and `jax` if NumPyro sampling is used
- **vocabulary-growth** — `columnar` (for `duckdb`), `io`, `notebook`, `viz`, and `jax` if NumPyro sampling is used
- **us-birth-certificates** — `boosting`, `columnar`, `dependence`, `graphs`, `io`, `notebook`, `storage`, `tuning`, and `jax` if NumPyro sampling is used; `fastparquet` belongs in the repo's own dependencies, as no extra carries it

Repo-only pure-Python packages (`formulaic`, `mpmath`, `pingouin`, `plotly`, `pyxlsb`, …) and dev tooling (`pytest`, `ruff`, `mypy`, `hatch`, `build`, stubs) go into `[dependency-groups]` in the repo's `pyproject.toml`, not into the package dependencies. Dependency groups are never published in package metadata.

Copy the `[tool.uv] environments` block from this repo's root `pyproject.toml` so the lockfile resolves for the same supported platforms.

### 2. Lock and verify

```bash
uv sync
uv run pytest
```

Commit the resulting `uv.lock`.

### 3. Update CI

Replace the `mamba-org/setup-micromamba` step with `astral-sh/setup-uv`, drop the `dse-check-env` step and the `shell: bash -el {0}` default, and add `windows-latest` to the test matrix. See `.github/workflows/ci.yml` in this repo for the shape.

### 4. Update dependabot

Replace the `pip` ecosystem entry with `package-ecosystem: uv`. Keep the numpy `>=2.5.0` ignore rule: `pytensor` still pins `numba<=0.66.0`, and `numba` 0.66.0 still pins `numpy<2.5`, so numpy 2.5 remains unreachable. It lifts when a PyTensor release admits numba 0.67.

## Ordering

Changes to the shared core follow the usual sequence, and this one is no different: **merge here, tag a release, then bump and re-run each consuming repo**. Consuming repos pin the library by git tag, so nothing downstream moves until its tag is bumped.

Until a repo has migrated, it keeps working unchanged: `environment-core.yml` and the `dse-check-env` console script are retained and deprecated, not deleted, and a parity test in this repo prevents the retained core from drifting away from `pyproject.toml`. Once all three repos are on uv, delete `environment-core.yml`, `dse-check-env`, `tests/test_environment_core_parity.py` and this guide.
