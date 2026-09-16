> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Upgrade to 0.15.2

<!-- cspell:ignore cachetools msgspec -->

Version 0.15.2 raises every directly declared Python dependency minimum to its latest compatible stable release checked on 16 September 2026. Requirements already at that release keep their existing minimum. The library API and Python 3.14 requirement are unchanged. Publish `v0.15.2` on the merged release commit after its checks pass; downstream upgrades require that tag to exist.

## Library requirements

These changes are part of the package metadata, so a downstream repository must satisfy them when it selects 0.15.2. An optional requirement applies when that extra is selected.

| Package            | Previous minimum | New minimum | Scope      |
| ------------------ | ---------------- | ----------- | ---------- |
| azure-identity     | None             | 1.25.3      | Base       |
| azure-storage-blob | None             | 12.30.1     | Base       |
| matplotlib         | 3.11.1           | 3.11.2      | Base       |
| numba              | 0.58             | 0.67.0      | Base       |
| numpy              | 2.4.6            | 2.5.3       | Base       |
| pyarrow            | 24.0.0           | 25.0.1      | Base       |
| pymc               | 6.3.1            | 6.3.2       | Base       |
| pytensor           | 3.3.1            | 3.3.2       | Base       |
| scikit-learn       | 1.9.0            | 1.9.1       | Base       |
| scipy              | 1.18.0           | 1.18.1      | Base       |
| dcor               | None             | 0.7         | dependence |
| jax                | 0.10.2           | 0.11.1      | jax        |
| polars             | 1.44.1           | 1.44.2      | columnar   |
| zarr               | 3.3.0            | 3.4.0       | storage    |

NumPy retains its `<2.6` limit and PyTensor retains `<3.4`. PyTensor 3.3.2 requires numba at most 0.67.0, and numba 0.67.0 requires NumPy below 2.6. PyMC still requires `cachetools<7`, so the available cachetools 7 release remains excluded.

[PyTensor 3.3.2](https://github.com/pymc-devs/pytensor/releases/tag/rel-3.3.2) fixes Numba compatibility with SciPy 1.18. [Zarr 3.4.0](https://github.com/zarr-developers/zarr-python/releases/tag/v3.4.0) adds `msgspec` as a metadata-validation dependency.

## Repository tooling

The repository's development minimums also move to build 1.6.1, pandas-stubs 3.0.5.260914, Ruff 0.16.7 and scipy-stubs 1.18.1.0. Its research group requires Plotly 7.1.0. These groups are not included in the library's published dependency metadata.

The release also contains the refreshed Python lock, cspell 10.3.2, Prettier 3.9.7 and setup-uv 10.1.0. Consumers manage their own npm dependencies and workflows.

## Upgrade a downstream repository

1. Change the research source tag from `v0.15.1` to `v0.15.2` in the downstream `pyproject.toml`. Preserve its existing extras and other dependency constraints.
2. Resolve the new library requirement and install the resulting environment.

```bash
uv lock --upgrade-package dse-research-utils
uv lock --check
uv sync --locked
```

The new minimums force changes wherever the old lock selected an older version. Other packages can remain at their previously locked versions. To refresh the entire downstream environment, use `uv lock --upgrade` and review all resulting changes. A minimum requirement permits later compatible releases; it does not make the research lock an exact environment specification for consumers.

In the three downstream local checkouts inspected on 16 September 2026, nutpie also brings in Zarr as a dependency. The new Zarr minimum applies directly only to consumers selecting the `storage` extra. A consumer without that extra can update Zarr through its own lock refresh without adding a duplicate dependency declaration.

3. Run the downstream repository's tests, including its model compilation, sampling, plotting and storage checks. Verify the installed library version is 0.15.2 and record the resolved tag commit and package versions.
4. Keep the recorded environments of existing fitted results. Test results for the shared library do not establish that rerunning an analysis with new dependencies will reproduce identical numerical results.
