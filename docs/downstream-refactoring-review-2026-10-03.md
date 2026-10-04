> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words logit invlogit iloc pvalue trivariate coef nonfinite missingness BFMI HSGP -->

# Downstream Python refactoring review

Reviewed on 3 October 2026 against `dse-research-utils` 0.16.2. The best next steps are to finish the US births statistics migration, use the ordinary Beta-Binomial builder in vocabulary-growth, and replace the remaining US births figure writers. Several other changes can use existing APIs, but need adapters to retain reporting rules and input checks. Much of the main fitting infrastructure already uses the library. Moving whole model pipelines would bring study-specific decisions into the shared package and would offer less reuse than moving their numerical and file operations.

## Scope and evidence

The review used local source at these commits. All four checkouts were clean when the review began. Links below identify those source snapshots rather than a moving branch.

| Repository                  | Commit                                     |
| --------------------------- | ------------------------------------------ |
| research                    | `0b51bcc5372bfe0d57297b9806abedf4c7fa9375` |
| language-reading-predictors | `2203024a800c4ecbccd06c66a79ec915885c5772` |
| vocabulary-growth           | `ef0d2b3dc275cae073973964fbc98cd3d84d65b7` |
| us-birth-certificates       | `639b2167a256ed586c5313fafaaadeee9e5a4de7` |

I inventoried 684 Python files under `src/`, 215 under `scripts/`, and 50 notebook source files across the downstream projects. I compared function bodies and shared imports, searched the 442 Quarto source files for embedded helpers, and read the implementations, callers and relevant tests for the candidates below. This was a focused review of reuse boundaries, not a manual audit of every line or a review of the scientific specifications. One historical US births notebook, `notebooks/00011-predictors-11.py`, could not be parsed because of an unterminated string at line 33. It was excluded from the syntax-tree comparison and does not support a finding below.

Small runtime comparisons used the US births Python 3.14 environment, with this checkout's library source and the current downstream source on the import path. They used synthetic arrays and a small PyMC model. No research model was fitted, no existing fit was rewritten, and no downstream source was changed. The comparisons establish the stated examples, not compatibility for every production input.

## Refactors using existing library APIs

### 1. Finish the US births statistics migration

The [US births statistics module](https://github.com/dspopulations/us-birth-certificates/blob/639b2167a256ed586c5313fafaaadeee9e5a4de7/src/dspopulations_us_birth_certificates/stats_utils.py#L19) still implements transforms, descriptive summaries, distance correlation, Spearman correlation and mutual information. The [language and reading module](https://github.com/dseinternational/language-reading-predictors/blob/2203024a800c4ecbccd06c66a79ec915885c5772/src/language_reading_predictors/stats_utils.py#L4) already imports these functions from the library. The US births [model pipeline](https://github.com/dspopulations/us-birth-certificates/blob/639b2167a256ed586c5313fafaaadeee9e5a4de7/src/dspopulations_us_birth_certificates/models/base_pipeline.py#L62) still uses the local distance-correlation calculation, so this is more than unused notebook code.

Use `statistics.transforms` for `standardize`, `logit`, `invlogit` and `convert_to_categorical`. Use `statistics.descriptive.differential_entropy_standardized` and `ml.feature_dependence` for the corresponding calculations. Keep the current import paths as compatibility adapters while callers move to shared imports. Do not replace the distinct `standardize` and `standardise` functions interchangeably. They use different standard deviations and different rules for missing or constant inputs.

The distance-correlation comparison matched for finite, missing and constant inputs. Standardisation also matched in three small cases. The shared inverse logit avoided the overflow warning produced by the local expression at -1000. These are the lowest-risk parts of this migration. The array-coercion wrappers in [selection priors](https://github.com/dspopulations/us-birth-certificates/blob/639b2167a256ed586c5313fafaaadeee9e5a4de7/src/dspopulations_us_birth_certificates/selection/priors.py#L40) should retain their list-to-array conversion if they delegate to the numeric transforms. Symbolic model expressions should continue to use PyMC operations.

Treat the following parts as behaviour changes, not simple code movement.

- The local Spearman array path failed on a two-column input. With a missing value in a three-column input, it set the affected cross-correlations to zero. The shared helper returned a square matrix and used available pairs, as the local DataFrame path already does. This can change feature groups for array inputs.
- The local mutual-information helper rejected an array because it assumed `.iloc`. When the estimator was made to return zero scores, it produced four non-finite cells for a two-feature matrix. The shared helper returned finite off-diagonal distances of one and a zero diagonal. This controlled case establishes the zero-denominator branch; an arbitrary constant dataset need not make the estimator return exactly zero scores.
- The shared descriptive summary adds `n_non_na`, `anderson_stat` and `anderson_pvalue`. Existing fields matched in the synthetic comparison, but a direct import changes the table schema and runs an additional test. Retain the old field order and selection in an adapter unless the report change is intended. For workloads with very many records, measure the extra computation before adopting that additional test.

Validation should compare feature matrices and linkage outputs, not only summary numbers. Extend the existing `tests/test_shared_adapters.py` coverage with two-column arrays, missing pairs, zero mutual-information scores and the retained descriptive table schema.

### 2. Use the ordinary Beta-Binomial builder in vocabulary-growth

The [univariate engine](https://github.com/dseinternational/vocabulary-growth/blob/ef0d2b3dc275cae073973964fbc98cd3d84d65b7/src/vocab_growth/models/common.py#L964) and its [predictive stage](https://github.com/dseinternational/vocabulary-growth/blob/ef0d2b3dc275cae073973964fbc98cd3d84d65b7/src/vocab_growth/models/common.py#L1536) repeat clipping followed by `alpha = p * kappa`, `beta = (1 - p) * kappa` and `pm.BetaBinomial`. The same pattern occurs for the understood outcome and predictive nodes in the [bivariate engine](https://github.com/dseinternational/vocabulary-growth/blob/ef0d2b3dc275cae073973964fbc98cd3d84d65b7/src/vocab_growth/models/common_bivariate.py#L816), the trivariate engine and the univariate engine with child effects.

Replace those ordinary constructions with [statistics.models.likelihood.beta_binomial_from_p](https://github.com/dseinternational/research/blob/0b51bcc5372bfe0d57297b9806abedf4c7fa9375/src/python/src/dse_research_utils/statistics/models/likelihood.py#L15). Pass the same probabilities, concentration parameters, denominators, observations, node names and dimensions. Remove the surrounding clip only where the helper performs exactly that operation and the clipped value has no other use.

The small model comparison preserved the free-variable names and order and gave exactly equal log densities at two parameter points, including observations at zero and the maximum count. Full adoption still needs the project's `tests/test_graph_equivalence.py` checks for each affected engine. A library call can preserve the posterior calculation while changing vocabulary-growth's executable-code signature. Follow its saved-fit compatibility rules when deciding whether old samples can be reused.

Keep `nested_outcome_alpha_beta`, product-marginal concentration, separate fallback dispersion and Dirichlet-Multinomial constructions local. They encode different likelihoods. Sending their inputs through the ordinary builder without preserving those choices could change the model.

### 3. Replace the remaining US births figure writers

The [selection renderer](https://github.com/dspopulations/us-birth-certificates/blob/639b2167a256ed586c5313fafaaadeee9e5a4de7/src/dspopulations_us_birth_certificates/selection/render.py#L56), [core reporting module](https://github.com/dspopulations/us-birth-certificates/blob/639b2167a256ed586c5313fafaaadeee9e5a4de7/src/dspopulations_us_birth_certificates/selection/core_reporting.py#L62), [race surveillance audit](https://github.com/dspopulations/us-birth-certificates/blob/639b2167a256ed586c5313fafaaadeee9e5a4de7/scripts/audit_core_race_surveillance.py#L1478) and [sensitivity comparison](https://github.com/dspopulations/us-birth-certificates/blob/639b2167a256ed586c5313fafaaadeee9e5a4de7/scripts/compare_core_reduction_sensitivities.py#L1744) each write PNG and SVG files and close the figure. Other US births plotting code already delegates to the library.

Use `plot.io.save_styled_figure` behind these local wrappers. Pass `fig`, `bbox_inches="tight"`, `close=True` and `svg_max_bytes=None` explicitly. Preserve 150 DPI in core reporting and the current supplied or shared DPI elsewhere. Preserve the two-path return value in the audit scripts. The selection renderer deliberately writes a data CSV beside the figure and another under `tables/`; preserve both destinations. Use `save_plot_data` for the second copy.

The synthetic comparison produced identical PNG and CSV bytes at 150 DPI, wrote an SVG, and closed the supplied figure. It did not compare SVG bytes, which can include generated identifiers. The shared writer also catches SVG errors and retains the PNG, whereas these local writers raise. If the caller requires a successful SVG write, use `svg=False` and `close=False`, retain that write locally, then close the figure after success. Accepting the shared error policy is a separate behaviour change. Check filenames, directory layout, return values, figure lifetime and SVG failures in `tests/test_render_selection_diagnostics.py`, `tests/test_core_reduction_reporting_age.py` and the existing shared-adapter tests. This refactor does not require a new plotting abstraction.

### 4. Extend labelled sample extraction in vocabulary-growth

The three extraction functions at the start of [posterior_analysis.py](https://github.com/dseinternational/vocabulary-growth/blob/ef0d2b3dc275cae073973964fbc98cd3d84d65b7/src/vocab_growth/posterior_analysis.py#L20) repeat chain/draw stacking and transposition. Its calibration module already uses `statistics.samples.sample_matrix`.

Use `sample_matrix(array, sample_dims=("chain", "draw"), observation_dims=(dim,))` as the shared operation. Retain the integer conversion for predictive word counts. The current extractors also return a new NumPy array; preserve that copy if callers can mutate the result, because the shared matrix may share memory with its source. Where samples are paired with observed values, retain the matrix metadata long enough to call `observed_values` and check the labels.

The shared and local extractors returned the same values in the same order when the stored dimension order was permuted. The shared pairing check rejected observations with reordered labels. It also rejected an array without explicit coordinate indexes. That stricter contract needs a deliberate policy for old traces. Do not invent positional labels and then describe them as proof of observation identity.

Keep the outcome-mask expansion and child-effect simulation local. Test missing outcome rows, count data types, copy behaviour, coordinate order and old-trace handling in `tests/test_posterior_analysis.py`. Equal shapes alone are insufficient evidence that observations line up.

### 5. Route remaining interval reductions through explicit shared mechanics

The [language and reading posterior helpers](https://github.com/dseinternational/language-reading-predictors/blob/2203024a800c4ecbccd06c66a79ec915885c5772/src/language_reading_predictors/statistical_models/posteriors.py#L17) repeat inner and outer quantiles in `band50`, `beta_summary` and `coef_row`. US births [selection diagnostics](https://github.com/dspopulations/us-birth-certificates/blob/639b2167a256ed586c5313fafaaadeee9e5a4de7/src/dspopulations_us_birth_certificates/selection/diagnostics.py#L76) still reduce the first two stored axes through `_quantile` and calculate some interval pairs directly, despite having a project interval adapter.

For language and reading, use `statistics.array_intervals.equal_tail_interval` with the current probability, sample axes and `nonfinite="propagate"`. Preserve means, medians, tail probabilities and field names in the local summary. For US births, route interval pairs through its existing `intervals.equal_tail_interval`, which already calls the shared helper. Choose axes from the intended chain and draw dimensions rather than assuming their storage positions.

The finite and NaN examples matched raw quantiles with the explicit propagation policy. The shared array API converts values to float64 and returns NaN bounds for empty slices. Check those differences before replacing calls that use float32 inputs or currently raise for empty input. Do not substitute `eti_bands` automatically, because it drops all non-finite draws. Do not change US births means to medians or combine vocabulary-growth's equal-tailed and highest-density interval policies. An interval convention is a reporting choice; duplicate arithmetic is the part to remove.

The [vocabulary-growth interval-column helper](https://github.com/dseinternational/vocabulary-growth/blob/ef0d2b3dc275cae073973964fbc98cd3d84d65b7/src/vocab_growth/posterior_analysis.py#L141) could also use its existing `intervals.summarise` adapter for the median and two bands. That helper filters non-finite draws for the median, whereas the current code uses raw `np.median`. Adopt it only after choosing the desired missing-value rule and comparing full output tables.

### 6. Finish file provenance reuse

Most primary provenance paths already use `metadata.provenance`. Remaining file-hash loops include [language and reading preprocessing](https://github.com/dseinternational/language-reading-predictors/blob/2203024a800c4ecbccd06c66a79ec915885c5772/src/language_reading_predictors/statistical_models/preprocessing.py#L162), [ITT missingness](https://github.com/dseinternational/language-reading-predictors/blob/2203024a800c4ecbccd06c66a79ec915885c5772/src/language_reading_predictors/statistical_models/itt_missingness.py#L157), `scripts/horseshoe_prior_sensitivity.py`, `scripts/run_refit_sweep.py`, and the [US births surveillance audit](https://github.com/dspopulations/us-birth-certificates/blob/639b2167a256ed586c5313fafaaadeee9e5a4de7/scripts/audit_core_race_surveillance.py#L135).

Delegate these binary file reads to `metadata.provenance.sha256_file`. Preserve path resolution, missing-file errors, digest prefixes and the refit sweep's `None` result on `OSError`. The comparison matched a digest for a 2,304,000-byte file. Leave hashes of in-memory source bytes, prepared frames, model graphs and ordered sets of fingerprints local. They hash different things and need their current ordering and serialisation.

The US births audit's `_package_versions` can delegate metadata lookup to the shared `package_versions`, then explicitly raise when a required version is unavailable. Its `_git_provenance` needs more care. The shared snapshot provides commit, branch and dirty state, but does not return the audit's `worktree_status` lines. Keep that extra query if the report needs it, and preserve failure on unavailable facts. Replacing the whole provenance dictionary with a snapshot would lose part of the schema.

### 7. Finish report reader and lookup reuse

The main language and reading report already uses the library's readers. Its [release reader](https://github.com/dseinternational/language-reading-predictors/blob/2203024a800c4ecbccd06c66a79ec915885c5772/src/language_reading_predictors/statistical_models/release/base.py#L140) still parses JSON separately. US births reports repeat checks for missing files and JSON/CSV parsing, for example in [the boosting report](https://github.com/dspopulations/us-birth-certificates/blob/639b2167a256ed586c5313fafaaadeee9e5a4de7/docs/models/usbc10/index.qmd#L25). These can use `report.readers.read_json` and `read_csv`, while local adapters translate `present`, `missing` and `invalid` into the existing report or release behaviour. The shared CSV reader exposes only `index_col`; keep specialised readers that need other parser options.

Treat the stricter JSON parser as a behaviour change. It rejects bare non-finite constants and overflowing numeric literals. A present parsed object is still subject to schema, fit compatibility and publication checks. Preserve a loud failure or a withheld result where that is the current rule. Do not turn a corrupt required artefact into a pending-fit message merely to use `ReportData`.

Vocabulary-growth still has nearest-row calculations in [the comparison report](https://github.com/dseinternational/vocabulary-growth/blob/ef0d2b3dc275cae073973964fbc98cd3d84d65b7/docs/comparison/index.qmd#L59) and [the paper helper](https://github.com/dseinternational/vocabulary-growth/blob/ef0d2b3dc275cae073973964fbc98cd3d84d65b7/docs/paper/_paper_data.qmd#L83). Use `report.readers.nearest_row` inside their adapters, while retaining rejection of queries outside the grid range and the current all-NaN-Series or `None` return. The shared default selects an endpoint outside the grid, so a direct replacement would remove that safeguard. The local helpers drop NaN keys; the shared helper also skips infinite and non-numeric keys. Make this stricter grid check explicit and use the same finite keys for the range guard and the lookup. Keep VG15's explicitly checked age lookup unchanged.

Another small cleanup is to use `statistics.loo.as_dataset` for the compatibility conversions in vocabulary-growth's comparison and recovery modules. Preserve optional-group handling and the intentional resetting of chain/draw coordinates in recovery simulation. This cleanup has lower value than the candidates above.

## Shared API additions worth considering

### Expose per-chain sampling signals and reductions of existing diagnostic tables

The US births [fit validator](https://github.com/dspopulations/us-birth-certificates/blob/639b2167a256ed586c5313fafaaadeee9e5a4de7/src/dspopulations_us_birth_certificates/selection/fit_validation.py#L16) repeats extraction of divergences and energy diagnostics. Several vocabulary-growth experiments, including [the repeater-prior comparison](https://github.com/dseinternational/vocabulary-growth/blob/ef0d2b3dc275cae073973964fbc98cd3d84d65b7/scripts/experiments/vg12_repeaters_prior.py#L234), calculate the energy diagnostic by iterating the first stored axis. That assumes the array is stored as chain by draw. The library's implementation reads the named dimensions instead.

The existing public `sampling_quality` API provides minimum BFMI, where BFMI is a check on how chains move through energy levels, but it does not expose the per-chain values these reports retain. Add a public per-chain field or function before migrating those reports. Avoid importing the library's private `_bfmi_per_chain` as a new downstream dependency. Keep variable selection, deterministic-constant exemptions, tree-depth checks, probability barriers and national-margin checks local.

A second useful addition would reduce an already computed diagnostic DataFrame into the largest R-hat, smallest effective sample size and names with unavailable diagnostics. R-hat checks agreement across sampling chains. Effective sample size estimates how many independent draws would give similar sampling precision. The implementation already exists privately in `statistics.diagnostics`. A public version would let US births reuse `summary_table` without asking ArviZ to calculate the summary again. It could accept explicit diagnostic column names or require the adapter to rename them first.

Do not make these additions a universal publication gate. US births currently uses `R-hat < 1.01`; the shared writer uses `R-hat <= 1.01`. Vocabulary-growth retains named exceptions and separate hard and soft checks. Language and reading checks agreement between persisted measurements and pass flags. Share measurement extraction and reduction; keep those decisions explicit. These changes require a library release and downstream tests before adoption.

### Share the file permission option without changing the default

All three projects already use `storage.files.atomic_write`, but [language and reading](https://github.com/dseinternational/language-reading-predictors/blob/2203024a800c4ecbccd06c66a79ec915885c5772/src/language_reading_predictors/atomic_files.py#L41), [vocabulary-growth](https://github.com/dseinternational/vocabulary-growth/blob/ef0d2b3dc275cae073973964fbc98cd3d84d65b7/src/vocab_growth/fit_artifacts.py#L96) and [US births](https://github.com/dspopulations/us-birth-certificates/blob/639b2167a256ed586c5313fafaaadeee9e5a4de7/src/dspopulations_us_birth_certificates/file_io.py#L24) separately restore the permission bits of an ordinary newly created file. Two do so by temporarily changing the process-wide `umask`; language and reading probes a new file instead.

Consider an optional shared permission policy or public helper for that operation. Retain the owner-only default of `atomic_write`, and make the caller opt into its existing publication mode. Compare new-file permissions, replaced-file permissions, callback copies and failures on supported platforms. A mode-bit probe alone does not prove that access-control entries on the replaced file will match. This is a useful common boundary, but the current adapters are necessary until that contract exists.

US births can already use its own shared-library-backed `file_io.write_atomically` for the remaining direct writes of `validation.json`, report metadata and report tables. Keep serialisation, CSV index rules and error messages in the caller. Single-file replacement does not make a set of artefacts a transaction.

## Boundaries to retain

The following areas already use shared mechanics and should not be counted as new extraction opportunities.

- LRP and vocabulary-growth predictive calibration already use labelled sample matrices and `predictive_observation_checks`.
- LRP child-level likelihood aggregation and vocabulary-growth administration aggregation already use `aggregate_log_likelihood`. The held-out units and latent-effect integration remain project choices.
- Main HSGP paths already use shared geometry. Vocabulary-growth's unpinned experimental branch tests a different centring choice and should not be migrated blindly.
- Feature-group linkage and permutation scoring already have shared implementations. Keep LRP's child-schedule donor design and the US births scoring population, score direction and cluster identifiers in adapters.
- Output-root resolution, directory promotion and Azure uploads are already shared where the projects use them. Retain locks, fit-acceptance rules, backup retention and publication checks in the projects.

The vocabulary-growth age-standardisation helper is also not a direct replacement for `statistics.transforms.standardise`. The local helper rejects an input containing NaN through its ordinary mean and standard deviation; the shared helper omits NaN for those calculations. That difference matters more than removing a few arithmetic lines.

## Suggested implementation order

1. Replace the remaining US births figure writers and straightforward numeric transforms, then file-hash loops. These changes have small, explicit compatibility surfaces.
2. Finish the US births dependence and descriptive migration as a reviewed numerical and table change. Compare downstream matrices, feature groups and report fields.
3. Migrate vocabulary-growth's ordinary Beta-Binomial nodes one engine at a time. Use the existing graph-equivalence checks and saved-fit identity rules.
4. Migrate array extraction, interval reductions and report readers with explicit policies for labels, missing values, old traces and report support.
5. Add the public sampling and permission APIs only where the downstream adapters show a real remaining need. Keep publication decisions and scientific specifications in the projects.
