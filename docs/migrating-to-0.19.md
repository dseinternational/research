> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words dateutil fetchone jaxlib -->

# Upgrade to 0.19.0

Version 0.19.0 is in preparation. These notes cover changes since the published `v0.18.0` tag, including dependency updates already merged after that release. Python 3.14 remains the minimum. Public helper signatures are unchanged, but dependencies can change results, file compatibility and figure colours.

## Dependency minimums

These requirements increase from 0.18.0. Other package minimums are unchanged. The `all` extra inherits all these increases.

| Requirement | Applies to                 | 0.18.0 minimum | 0.19.0 minimum |
| ----------- | -------------------------- | -------------- | -------------- |
| numba       | Base                       | 0.67.0         | 0.68.0         |
| numpy       | Base                       | 2.5.3          | 2.5.4          |
| pyarrow     | Base                       | 25.0.1         | 26.0.0         |
| pytensor    | Base                       | 3.3.3          | 3.3.4          |
| orjson      | `io`                       | 3.12.0         | 3.13.0         |
| jax         | `jax`                      | 0.11.2         | 0.12.0.1       |
| shap        | `boosting`, `boosting-cpu` | 0.52.0         | 0.54.1         |
| duckdb      | `columnar`                 | 1.5.6          | 2.0.0          |
| polars      | `columnar`                 | 1.44.2         | 2.0.0          |
| pyreadstat  | `columnar`                 | 1.3.6          | 1.3.7          |
| zarr        | `storage`                  | 3.4.0          | 3.4.1          |

NumPy remains below 2.6 because Numba 0.68 requires that limit. PyTensor remains below 3.4; version 3.3.4 admits Numba 0.68. The package keeps these compatibility bounds.

## Changes to review in consuming projects

### Databases and data imports

DuckDB 2.0 creates database files that DuckDB 1.x cannot read. Check shared database files and extensions before upgrading. Its Python relation API removes methods such as `rel.fetchone()`; connection methods have different rules. Review the [DuckDB migration guide](https://duckdb.org/docs/2.0/operations_manual/migration) and [Python migration guide](https://duckdb.org/docs/2.0/clients/python/migration).

Polars 2.0 changes SQL evaluation and numeric types, and reads Parquet enum columns as strings. Check the [Polars release notes](https://github.com/pola-rs/polars/releases/tag/py-2.0.0) against the project's queries, schemas and data exports.

Pyreadstat 1.3.7 fixes parallel Polars reads of chunks that contain only missing values. It also reports the declared SPSS string width in `variable_storage_width`, which can differ from the internal segment size returned before. Check metadata comparisons and code that relies on those widths. See the [pyreadstat release notes](https://github.com/Roche/pyreadstat/releases/tag/v1.3.7).

### JAX sampling

The `jax` and `all` extras require JAX 0.12.0.1 or later. JAX 0.12 requires AVX2 and FMA instructions on x86-64 CPUs; Apple silicon uses ARM64 builds. It also changes some array and random-number APIs. Review the [JAX 0.12.0 changes](https://docs.jax.dev/en/latest/changelog.html#jax-0-12-0-october-19-2026) if the project calls JAX directly. The [0.12.0.1 patch](https://github.com/jax-ml/jax/releases/tag/jax-v0.12.0.1) fixes Windows CPU intrinsics; this repository locks both JAX and jaxlib to that patch.

### Figure colours

The design tokens advance from `tokens/2026.3.0` to `tokens/2026.3.4`. The default categorical order is now blue, orange, green, purple, teal and pink. In 0.18.0, green preceded orange. As a result, `CHART_COLOURS[1]` and `"C1"` become orange, while `CHART_COLOURS[2]` and `"C2"` become green. The default `categorical_palette` follows the same order. The deprecated `COLOUR_GREEN` and `COLOUR_ORANGE` names retain their named hues.

Regenerate representative figures and check series meanings, legends and labels. Committed figures change only when rendered again. The [0.18.0 notes](migrating-to-0.18.md) still apply to projects upgrading from an earlier release.

## Repository environment

The contributor lockfile updates direct and indirect dependencies. This update includes python-dateutil 3.0.0, which changes time-zone handling. In particular, times in daylight-saving clock-change gaps can receive different UTC offsets. Review date parsing and time-zone calculations if the consuming project's resolver also selects it. See the [dateutil release notes](https://github.com/dateutil/dateutil/releases/tag/3.0.0).

The contributor tools now use Node.js 26 and pnpm 12.14.0. The CI uv setup action advances to 10.3.1. These tools, the root development groups and this repository's lockfile do not become requirements of the published Python package. Consuming projects resolve their own environments from the package requirements and their own constraints.

## Upgrade

After `v0.19.0` is published, follow the [shared upgrade procedure](README.md#upgrade-a-consuming-project) with that target tag. Preserve the project's selected extras. Run its tests and compare data schemas, date handling, representative model diagnostics and rendered figures.

Record the resolved commit and installed package versions. Keep historical fit manifests and environments intact. Apply the project's fit-compatibility rules before reusing or publishing stored fits. The shared library's tests do not establish compatibility of a consuming project's databases, queries or stored fits.
