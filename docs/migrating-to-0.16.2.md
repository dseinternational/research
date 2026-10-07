> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Upgrade to 0.16.2

Version 0.16.2 raises dependency minimums. The Python API and Python 3.14 requirement are unchanged.

## Library requirements

The following minimum versions changed between the published `v0.16.1` and `v0.16.2` tags. A downstream repository must satisfy the base requirements when it selects 0.16.2. The DuckDB requirement applies when it selects the `columnar` extra.

| Package            | Previous minimum | New minimum | Scope    |
| ------------------ | ---------------- | ----------- | -------- |
| azure-identity     | 1.25.3           | 1.26.0      | Base     |
| azure-storage-blob | 12.30.3          | 12.31.0     | Base     |
| pytensor           | 3.3.2            | 3.3.3       | Base     |
| xarray             | 2026.7.0         | 2026.9.0    | Base     |
| duckdb             | 1.5.5            | 1.5.6       | columnar |

Numba remains at 0.67.0 because [PyTensor 3.3.3's dependency metadata](https://github.com/pymc-devs/pytensor/blob/rel-3.3.3/pyproject.toml) requires `numba<=0.67.0`. Numba 0.68.0 cannot be installed with that release. NumPy retains its `<2.6` limit and PyTensor retains `<3.4`.

## Upgrade

Follow the [shared upgrade procedure](README.md#upgrade-a-consuming-project), with `v0.16.2` as the target tag. New minimums require changes wherever the old lock selected an older version. The consuming project resolves its own remaining dependencies.

Run the project's checks for model compilation, sampling, plotting and storage where applicable. Projects upgrading from 0.15.2 or earlier must also review the [0.16.0 font changes](migrating-to-0.16.md) and [0.16.1 symbol fallback](migrating-to-0.16.1.md).
