# Python utilities

`dse-research-utils` — the shared library for [Down Syndrome Education International](https://www.down-syndrome.org/) research projects.

This package's `pyproject.toml` is the canonical source of the dependency floors shared across DSE research repositories. Consuming repositories depend on the library and inherit those floors transitively rather than restating package versions of their own.

## Install

```bash
uv add "dse-research-utils @ git+https://github.com/dseinternational/research.git@v0.11.2#subdirectory=src/python"
```

The base install carries the modelling stack (PyMC, PyTensor, nutpie, ArviZ, PreliZ), the numerics core, and the netCDF engine (h5netcdf, h5py) that `InferenceData.to_netcdf` requires. Optional layers are extras:

| Extra        | Adds                       | For                                                |
| ------------ | -------------------------- | -------------------------------------------------- |
| `viz`        | seaborn                    | `plot.grids` histogram grids                       |
| `graphs`     | graphviz, networkx         | `plot.graphs` — also needs the system `dot` binary |
| `notebook`   | jupyter, jupytext          | notebook workflows; `plot.io.display_image`        |
| `dependence` | dcor                       | `ml.feature_dependence.distance_corr_matrix`       |
| `tuning`     | optuna, optuna-integration | hyper-parameter search                             |
| `io`         | orjson, tabulate           | fast JSON and table rendering                      |
| `jax`        | jax, numpyro               | JAX/NumPyro sampler backends                       |
| `boosting`   | lightgbm, xgboost, shap    | gradient boosting and explanation                  |
| `columnar`   | duckdb, polars, pyreadstat | columnar and statistical data formats              |
| `storage`    | zarr                       | zarr as an alternative to the netCDF core          |
| `all`        | every extra above          | development environments                           |

Helpers that need an extra lazy-import it and raise a clear error when it is absent.

## Development

Work from the repository root, which is the uv workspace root:

```bash
uv sync                                 # create .venv
uv run pytest                           # tests
uv run ruff check src/python            # lint
uv build --package dse-research-utils   # wheel + sdist
```
