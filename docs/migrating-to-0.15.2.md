> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Upgrade to 0.15.2

<!-- cspell:ignore cachetools msgspec -->

Version 0.15.2 raises the dependency minimums listed below. The library API and Python 3.14 requirement are unchanged. These are the requirements at the published tag, rather than a list of current upstream releases.

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

## Upgrade

Follow the [shared upgrade procedure](README.md#upgrade-a-consuming-project), with `v0.15.2` as the target tag. New minimums require changes wherever the old lock selected an older version. Other packages can remain locked.

The Zarr minimum applies directly only when the `storage` extra is selected. If another dependency also installs Zarr, the consuming project's resolver determines its version from all applicable requirements.

Test model compilation, sampling, plotting and storage where the project uses them. Preserve recorded environments for existing fits. Apply the [0.15.0 notes](migrating-to-0.15.md) and earlier requirements if upgrading from before those versions.
