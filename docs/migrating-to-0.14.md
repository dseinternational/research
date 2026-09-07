> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words HSGP float64 float32 PSIS ELPD nonfinite RMSE dtype dtypes lockfiles -->

# Migrating to 0.14.0

Version 0.14.0 collects the shared utilities merged in [PR #101](https://github.com/dseinternational/research/pull/101). It adds no dependency requirements. Each downstream project can adopt the whole set through one dependency update and one migration PR, while keeping its scientific decisions and stored formats in local adapters.

## Release sequence

The version change prepares the release. It does not publish `v0.14.0`. Merge the release-preparation PR, check CI on the merged commit, then tag that commit as `v0.14.0`. The existing `v0.13.0` tag points to the code before PR #101 and must remain there.

The installation commands in this guide require the new tag. Before tagging, maintainers can build the wheel and source distribution from the release-preparation branch and test them locally. A package reporting version `0.14.0` from such a build is not evidence that the release tag exists.

```bash
uv sync --locked
uv run pytest
uv run ruff check src/python
uv run ruff format --check src/python
uv build --package dse-research-utils
```

Run the repository's Markdown checks too. Check that both built distributions report `0.14.0`. Git tags are the installation source used by the consuming repositories; merging this PR does not update their pins or lockfiles.

## Update a consumer once

After the tag exists, a project using a direct Git dependency can update it with:

```bash
uv add "dse-research-utils @ git+https://github.com/dseinternational/research.git@v0.14.0#subdirectory=src/python"
```

Retain the project's existing extras. A project that records its Git source in `tool.uv.sources` can change only the tag in that entry:

```toml
[tool.uv.sources]
dse-research-utils = { git = "https://github.com/dseinternational/research.git", tag = "v0.14.0", subdirectory = "src/python" }
```

Keep the existing dependency declaration and extras, then run `uv lock` and `uv sync --locked`. Commit the changed pin and lockfile together with the adapters and their checks. There is no need to adopt every helper immediately, but copied implementations continue to require maintenance until callers delegate to the shared functions.

Projects upgrading from before `0.13.0` must also apply the [0.13.0 migration requirements](migrating-to-0.13.md). That version changed HSGP boundaries, diagnostic decisions and upload results; moving directly to `0.14.0` does not remove those requirements.

## What the release adds

| Area                             | Shared operations                                                                                                                        | Migration reference                                                                                                                                                                                               |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Files and provenance             | Atomic file writing, directory promotion with a caller lock and retained backup, Git status, package versions and file hashes.           | [File and provenance guide](shared-file-provenance.md), [directory promotion](consolidation-migration.md#promote-a-completed-directory).                                                                          |
| Statistical arrays               | Labelled sample extraction, observation-identity checks, predictive summaries and explicit likelihood-factor aggregation.                | [Statistical array guide](shared-predictive-arrays.md).                                                                                                                                                           |
| Reports                          | HTML asset inspection, upload-inventory and HTTP checks, present/missing/invalid reads and nearest-row lookup.                           | [Asset guide](shared-assets-and-feature-groups.md), [report reads](consolidation-migration.md#read-file-state-before-applying-report-rules).                                                                      |
| Feature evaluation               | Grouping from existing distance matrices and separate held-out and pooled out-of-fold permutation evaluators.                            | [Feature groups](shared-assets-and-feature-groups.md#reuse-a-dissimilarity-matrix-for-feature-grouping), [permutation scoring](consolidation-migration.md#keep-held-out-and-pooled-permutation-scoring-explicit). |
| Intervals and Gaussian processes | Equal-tailed array reductions with explicit axes and nonfinite policies, and frozen HSGP geometry with covariance-injected construction. | [Intervals](consolidation-migration.md#choose-interval-axes-and-missing-value-rules-explicitly), [HSGP geometry](consolidation-migration.md#save-hsgp-geometry-and-inject-the-covariance).                        |

Existing helpers delegate only where their intended behaviour is preserved. The diagnostic writer retains its serialization and cache behaviour. `ReportData.value_at` now uses safe nearest-row selection, which corrects unsigned subtraction, integer overflow and false ties from rounded integer queries. The distance-correlation linkage helper reuses the shared tree builder and supports empty or singleton feature sets. The high-level HSGP builder delegates calibration while retaining its prior and graph construction; invalid or unrepresentable calibration inputs now fail explicitly. Older interval helpers, report loaders and permutation functions retain their existing contracts.

## Check the meaning of the results

The [combined migration guide](consolidation-migration.md#complete-each-downstream-upgrade) maps the work to each consumer. Compare the actual downstream output tables and model calculations, as well as helper tests. In particular:

- Preserve observation and likelihood units, dimension labels, row ordering and factor selection. Equal totals do not establish equal pointwise PSIS-LOO results.
- Review numerical precision. Likelihood sums, array intervals and permutation scoring use float64, so calculations previously made in float32 can change. Predictive summaries have their own documented dtype and boundary-rounding conventions.
- Retain each project's mean or median, interval coverage, missing-value policy, score direction, donor generation and evaluated population. Pooled RMSE differs from an average of fold scores.
- Keep serialization, manifest fields, category metadata, feature-group identifiers and output column dtypes stable where compatibility requires them.
- Validate saved HSGP geometry, priors, variable names and dimensions. Review implementation and design identities before resuming a fit or publishing existing results. Preserve historical manifests as records of their actual producers.
- Retain report schema, freshness and scientific visibility checks. A readable file or available HTTP resource does not establish that a result can be published. Directory promotion needs cooperating writers and can expose a gap to readers between its two renames.

The development work compared synthetic fixtures with current downstream functions, including complete tables, likelihood arrays, report read states, permutation deltas and HSGP graph values. It did not migrate downstream production code, refit models or publish reports. Complete those project-specific checks before accepting the upgraded outputs.
