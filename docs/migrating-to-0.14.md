> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words HSGP float64 float32 PSIS ELPD nonfinite RMSE dtype dtypes lockfiles -->

# Migrating to 0.14.0

Version 0.14.0 collects the shared utilities merged in [PR #101](https://github.com/dseinternational/research/pull/101). It adds no dependency requirements. Each downstream project can adopt the whole set through one dependency update and one migration PR, while keeping its scientific decisions and stored formats in local adapters.

## Upgrade

Follow the [shared upgrade procedure](README.md#upgrade-a-consuming-project), with `v0.14.0` as the target tag. Projects upgrading from before 0.13.0 must also apply the [0.13.0 migration requirements](migrating-to-0.13.md). The [combined helper guide](consolidation-migration.md) covers adapter design and checks.

## What the release adds

| Area                             | Shared operations                                                                                                                               | Migration reference                                                                                                                                                                                               |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Files and provenance             | Atomic file writing, directory promotion with a caller lock and retained backup, Git status, package versions and file hashes.                  | [File and provenance guide](shared-file-provenance.md), [directory promotion](consolidation-migration.md#promote-a-completed-directory).                                                                          |
| Statistical arrays               | Labelled sample extraction, observation-identity checks, predictive summaries and explicit likelihood-factor aggregation.                       | [Statistical array guide](shared-predictive-arrays.md).                                                                                                                                                           |
| Reports                          | HTML asset inspection, upload-inventory and HTTP checks, present/missing/invalid reads and nearest-row lookup.                                  | [Asset guide](shared-assets-and-feature-groups.md), [report reads](consolidation-migration.md#read-file-state-before-applying-report-rules).                                                                      |
| Feature evaluation               | Grouping from existing distance matrices and separate held-out and pooled out-of-fold permutation evaluators.                                   | [Feature groups](shared-assets-and-feature-groups.md#reuse-a-dissimilarity-matrix-for-feature-grouping), [permutation scoring](consolidation-migration.md#keep-held-out-and-pooled-permutation-scoring-explicit). |
| Intervals and Gaussian processes | Equal-tailed array reductions with explicit axes and non-finite-value policies, and frozen HSGP geometry with covariance-injected construction. | [Intervals](consolidation-migration.md#choose-interval-axes-and-missing-value-rules-explicitly), [HSGP geometry](consolidation-migration.md#save-hsgp-geometry-and-inject-the-covariance).                        |

Existing helpers delegate only where their intended behaviour is preserved. The diagnostic writer retains its serialisation and cache behaviour. `ReportData.value_at` now uses safe nearest-row selection, which corrects unsigned subtraction, integer overflow and false ties from rounded integer queries. The distance-correlation linkage helper reuses the shared tree builder and supports empty or singleton feature sets. The high-level HSGP builder delegates calibration while retaining its prior and graph construction; invalid or unrepresentable calibration inputs now fail explicitly. Older interval helpers, report loaders and permutation functions retain their existing contracts.

## Check the meaning of the results

The [combined helper guide](consolidation-migration.md#validate-adoption) describes adoption checks. Compare the actual downstream output tables and model calculations, as well as helper tests. In particular:

- Preserve observation and likelihood units, dimension labels, row ordering and factor selection. Equal totals do not establish equal pointwise PSIS-LOO results.
- Review numerical precision. Likelihood sums, array intervals and permutation scoring use float64, so calculations previously made in float32 can change. Predictive summaries have their own documented dtype and boundary-rounding conventions.
- Retain each project's mean or median, interval coverage, missing-value policy, score direction, donor generation and evaluated population. Pooled RMSE differs from an average of fold scores.
- Keep serialisation, manifest fields, category metadata, feature-group identifiers and output column dtypes stable where compatibility requires them.
- Validate saved HSGP geometry, priors, variable names and dimensions. Review implementation and design identities before resuming a fit or publishing existing results. Preserve historical manifests as records of their actual producers.
- Retain report schema, freshness and scientific visibility checks. A readable file or available HTTP resource does not establish that a result can be published. Directory promotion needs cooperating writers and can expose a gap to readers between its two renames.

Library tests compare synthetic tables, likelihood arrays, file-read states, permutation scores and HSGP calculations. Run the consuming project's own checks before accepting upgraded outputs.
