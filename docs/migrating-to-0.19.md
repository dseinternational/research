> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words dateutil dtype fetchone isoparse jaxlib LKJ timedelta -->

# Upgrade to 0.19.4

These notes cover changes from `v0.18.0` to `v0.19.4`. Python 3.14 remains the minimum. Public helper signatures are unchanged, but dependencies can change results, file compatibility and figure colours.

## Changes since 0.19.3

Version 0.19.4 includes PyMC 6.4.0, PyTensor 3.3.7 and pandas 3.1.1. It also raises the Azure Identity, Azure Blob Storage, h5netcdf, PyArrow and Zarr minimums shown below. The build requirement advances from Hatchling 1.32.4 to 1.33.0. The contributor lock files and tools include the updates merged after 0.19.3.

PyMC 6.4 fixes forward sampling for LKJ correlation and covariance distributions with more than two dimensions, and their log-density normalization. Projects that use these distributions should review the fixes before comparing, resuming or reusing stored fits. PyMC also changes `find_MAP` and adds boundary-condition options to its Gaussian-process approximation. See the [PyMC 6.4 release notes](https://github.com/pymc-devs/pymc/releases/tag/v6.4.0). Library tests do not establish compatibility of a consuming project's saved models.

Pandas 3.1.1 fixes missing datetime values becoming Unix-epoch timestamps during conversion to Arrow timestamp types. It also fixes Arrow-backed forward and backward fill sharing data with the source, and compatibility with dateutil time zones. Review affected data conversions against the [pandas 3.1.1 release notes](https://pandas.pydata.org/docs/whatsnew/v3.1.1.html).

The design-token pin advances from `tokens/2026.3.4` to `tokens/2026.3.7`. The research colour file is unchanged between these tags, so this update introduces no further colour changes after 0.19.3.

## Changes in 0.19.3

Version 0.19.3 raised the base pandas minimum from 3.0.6 to 3.1.0 and the `columnar` Polars minimum from 2.0.0 to 2.1.0. Projects upgrading from 0.19.2 or earlier should also review the pandas 3.1 and Polars 2.1 behaviour changes below.

## Dependency minimums

These requirements increase from 0.18.0. Other package minimums are unchanged. The `all` extra inherits all these increases.

| Requirement        | Applies to                 | 0.18.0 minimum | 0.19.4 minimum |
| ------------------ | -------------------------- | -------------- | -------------- |
| azure-identity     | Base                       | 1.26.0         | 1.27.0         |
| azure-storage-blob | Base                       | 12.31.0        | 12.31.1        |
| h5netcdf           | Base                       | 1.8.1          | 1.8.2          |
| numba              | Base                       | 0.67.0         | 0.68.0         |
| numpy              | Base                       | 2.5.3          | 2.5.4          |
| pandas             | Base                       | 3.0.6          | 3.1.1          |
| pyarrow            | Base                       | 25.0.1         | 26.0.1         |
| pymc               | Base                       | 6.3.2          | 6.4.0          |
| pytensor           | Base                       | 3.3.3          | 3.3.7          |
| orjson             | `io`                       | 3.12.0         | 3.13.0         |
| jax                | `jax`                      | 0.11.2         | 0.12.0.1       |
| shap               | `boosting`, `boosting-cpu` | 0.52.0         | 0.54.1         |
| duckdb             | `columnar`                 | 1.5.6          | 2.0.0          |
| polars             | `columnar`                 | 1.44.2         | 2.1.0          |
| pyreadstat         | `columnar`                 | 1.3.6          | 1.3.7          |
| zarr               | `storage`                  | 3.4.0          | 3.4.2          |

All direct package minimums match the latest stable releases checked on 5 November 2026, except Numba. Numba 0.68.1 is available, but [PyTensor 3.3.7 requires Numba 0.68.0 or earlier](https://github.com/pymc-devs/pytensor/blob/rel-3.3.7/pyproject.toml). Numba therefore stays at 0.68.0. Raising its minimum would make the requirements impossible to resolve. NumPy remains below 2.6 because Numba 0.68 requires that limit. PyTensor remains below 3.4, as required by PyMC 6.4.

## Changes to review in consuming projects

### Databases and data imports

Pandas 3.1 changes categorical inference in text and Excel readers: numeric and boolean values read with `dtype="category"` now produce numeric and boolean categories across engines. Read identifier columns as strings before converting to categories to preserve leading zeros. Numeric datetime and timedelta construction now defaults to microsecond resolution, or nanoseconds when the input precision requires it, rather than retaining the input unit. Review schemas, category labels and time calculations against the [pandas 3.1 release notes](https://pandas.pydata.org/docs/whatsnew/v3.1.0.html).

DuckDB 2.0 creates database files that DuckDB 1.x cannot read. Check shared database files and extensions before upgrading. Its Python relation API removes methods such as `rel.fetchone()`; connection methods have different rules. Review the [DuckDB migration guide](https://duckdb.org/docs/2.0/operations_manual/migration) and [Python migration guide](https://duckdb.org/docs/2.0/clients/python/migration).

Polars 2.0 changes SQL evaluation and numeric types, and reads Parquet enum columns as strings. Check the [Polars release notes](https://github.com/pola-rs/polars/releases/tag/py-2.0.0) against the project's queries, schemas and data exports.

Polars 2.1 also rejects mixed string and numeric arithmetic in eager operations. Cast operands explicitly where the project's intended conversion is clear. See the [Polars 2.1 release notes](https://github.com/pola-rs/polars/releases/tag/py-2.1.0).

Pyreadstat 1.3.7 fixes parallel Polars reads of chunks that contain only missing values. It also reports the declared SPSS string width in `variable_storage_width`, which can differ from the internal segment size returned before. Check metadata comparisons and code that relies on those widths. See the [pyreadstat release notes](https://github.com/Roche/pyreadstat/releases/tag/v1.3.7).

### JAX sampling

The `jax` and `all` extras require JAX 0.12.0.1 or later. JAX 0.12 requires AVX2 and FMA instructions on x86-64 CPUs; Apple silicon uses ARM64 builds. It also changes some array and random-number APIs. Review the [JAX 0.12.0 changes](https://docs.jax.dev/en/latest/changelog.html#jax-0-12-0-october-19-2026) if the project calls JAX directly. The [0.12.0.1 patch](https://github.com/jax-ml/jax/releases/tag/jax-v0.12.0.1) fixes Windows CPU intrinsics; this repository locks both JAX and jaxlib to that patch.

### Figure colours

The design tokens advance from `tokens/2026.3.0` to `tokens/2026.3.7`. The default categorical order is now blue, orange, green, purple, teal and pink. In 0.18.0, green preceded orange. As a result, `CHART_COLOURS[1]` and `"C1"` become orange, while `CHART_COLOURS[2]` and `"C2"` become green. The default `categorical_palette` follows the same order. The deprecated `COLOUR_GREEN` and `COLOUR_ORANGE` names retain their named hues. These colour changes were already present in 0.19.3.

Regenerate representative figures and check series meanings, legends and labels. Committed figures change only when rendered again. The [0.18.0 notes](migrating-to-0.18.md) still apply to projects upgrading from an earlier release.

## Repository environment

The contributor lockfile updates direct and indirect dependencies. Since 0.19.3, it advances python-dateutil from 3.0.0 to 4.0.0. Version 4.0 requires exact ASCII digit counts in the numeric fields accepted by `isoparse` and related parsers. Some malformed dates and offsets that it previously accepted now raise `ValueError`. It also fixes calculations for the 366th day of a leap year. See the [dateutil 4.0 release notes](https://github.com/dateutil/dateutil/releases/tag/4.0.0). Projects upgrading from 0.18.0 should also review the [dateutil 3.0 time-zone changes](https://github.com/dateutil/dateutil/releases/tag/3.0.0), including UTC offsets in daylight-saving clock-change gaps.

The contributor tools use Node.js 26, pnpm 12.16.1, Hatch 1.18.2, pytest 9.2.0, Ruff 0.17.3 and pandas-stubs 3.1.0.261029. CI uses setup-uv 10.3.3 and pnpm/action-setup 6.1.1. The .NET SDK remains at 10.0.403. The npm packages, CI actions, runtime pins and design-token tag were also checked for stable updates. These tools, the root development groups and this repository's lockfile do not become requirements of the published Python package. Consuming projects resolve their own environments from the package requirements and their own constraints.

## Upgrade

Follow the [shared upgrade procedure](README.md#upgrade-a-consuming-project) with target tag `v0.19.4`. Preserve the project's selected extras. Run its tests and compare data schemas, date handling, representative model diagnostics and rendered figures.

Record the resolved commit and installed package versions. Keep historical fit manifests and environments intact. Apply the project's fit-compatibility rules before reusing or publishing stored fits. The shared library's tests do not establish compatibility of a consuming project's databases, queries or stored fits.
