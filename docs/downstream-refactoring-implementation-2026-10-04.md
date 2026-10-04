> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words BFMI logit invlogit nonfinite missingness trivariate coef worktrees umask kappa -->

# Downstream refactoring implementation

Implemented on 4 October 2026 from the [approved review](downstream-refactoring-review-2026-10-03.md). The three downstream projects now use existing shared APIs for the selected calculations, sample extraction, report reads, provenance and file writes. This library also has new public diagnostic reductions and an optional file permission policy. Those additions need a release before downstream projects can replace their remaining diagnostic and permission adapters.

## Changes by repository

### US birth certificates

`stats_utils.py` now delegates numeric transforms, dependence measures, linkage and descriptive calculations to the library. A small adapter retains the existing 17 descriptive rows, their order and the grouped table layout. Prior transforms retain list-to-array conversion. The shared inverse logit avoids overflow for large negative inputs.

The dependence migration adopts the reviewed corrections. Spearman calculations accept two-column arrays and use available pairs when values are missing. Mutual-information distances accept arrays and stay finite when the estimator returns all zero scores. These changes can affect feature groups for inputs that previously failed or produced invalid distances.

Four figure writers use shared PNG saving. They retain local SVG writes because these reports require SVG failures to raise. Filenames, resolution, path return values and figure lifetime are retained. Selection rendering also keeps both CSV destinations. Validation JSON, report metadata and tables use the project's existing shared-library-backed atomic file adapter.

The surveillance audit delegates file hashes and package-version lookup. It still requires the files and package metadata to exist. Its full Git status record and hashes of prepared frames remain local. Selection interval reductions use the existing project adapter after placing named chain and draw dimensions first. Means and interval coverage remain project choices.

The new `report_readers.py` adapter serves five Quarto templates. An absent optional file uses the caller's existing default or pending message. An absent required file raises. Invalid JSON or CSV raises with the file path and parser reason. The JSON parser also rejects non-finite constants and overflowing numeric literals. Existing withheld-validation and required-table rules remain in the templates.

### Vocabulary growth

Twenty-two ordinary Beta-Binomial constructions across four model engines now use `statistics.models.likelihood.beta_binomial_from_p`. They pass the same names, dimensions, observations, denominators and concentration parameters. The specialised nested-outcome, product-marginal, fallback and Dirichlet-Multinomial constructions remain local.

Three posterior extractors use `statistics.samples.sample_matrix` when all required dimensions have explicit indexes. They retain observation-by-sample order, integer conversion for word counts and a fresh array that callers can mutate. Arrays from older traces without explicit indexes retain the previous positional extraction. This fallback does not prove observation identity. Explicit duplicate coordinate labels are rejected.

Comparison and recovery group conversions use `statistics.loo.as_dataset`. Recovery still resets the selected chain and draw coordinates where it did before. Comparison and paper row lookups use a shared project adapter over `report.readers.nearest_row`. It rejects requests outside the range of finite numeric grid values and retains the first row in a tie. The report retains its all-NaN row and the paper retains its `None` result when no supported row exists.

The interval-column helper still uses its existing raw median. Replacing it with the project's full summary adapter would change the treatment of missing draws. That change was conditional in the review and has not been adopted.

### Language and reading predictors

Five file-hash wrappers now use `metadata.provenance.sha256_file`. They retain resolved source paths, missing-file failures and the refit sweep's `None` result on a file error. Hashes of source bytes, prepared frames and ordered fingerprints remain local.

`band50`, `beta_summary` and `coef_row` use the shared equal-tail array reduction with explicit coverage and propagation of non-finite draws. The adapters retain means, medians, tail probabilities and field names. Empty samples still stop summary generation. Shared interval calculations use float64, including when the input uses float32.

The release JSON reader uses `report.readers.read_json` and retains its `missing` and `unreadable` outcomes. A parsed `null` remains distinct from a missing file. Non-finite JSON values now produce the `unreadable` outcome. Release gates and saved-fit checks remain local.

### Shared library

`statistics.diagnostics.bfmi_per_chain` exposes the existing energy calculation in named chain and draw order. `diagnostic_extrema` reduces an existing unrounded diagnostic table into maximum R-hat, minimum effective sample size and names with unavailable diagnostics. Neither chooses variables nor applies a pass/fail rule. `sampling_quality` uses the public operations. The former private energy function remains as a compatibility wrapper.

`storage.files.atomic_write` accepts optional permission bits or `mode="default"`. The default remains the callback's file permissions, including the initial owner-only mode when the callback does not change it. `default_file_mode` reads the permission bits of a newly created ordinary file with an empty exclusive probe. It removes the probe and does not change the process-wide `umask`. Permission failures preserve the old destination. These options do not preserve access-control entries or inherit an existing destination's permissions.

## Validation

The checks used the projects' existing Python 3.14 environments. Downstream full or selected suites loaded this checkout's library source. The final adapter checks also passed without that source override, against each original environment's installed library, version 0.16.2. Dependency declarations and lock files are unchanged.

| Repository                      | Completed checks                                                                                                                                                                                                                          |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Shared library                  | Full Python suite, 1,186 passed and 5 skipped; Ruff; Markdown format and spellcheck.                                                                                                                                                      |
| US birth certificates           | Full default suite, 381 passed, 1 skipped and 12 slow tests excluded; Ruff across source, tests and scripts; spellcheck across 54 documents.                                                                                              |
| Vocabulary growth               | Full fast suite, 2,644 passed, 11 skipped and 392 slow cases excluded; all 118 registered-model graph comparison tests; Ruff; configured type checks across 4 files; Markdown format and spellcheck.                                      |
| Language and reading predictors | Selected provenance, preprocessing and missingness suite, 142 passed; selected release and posterior suite, 182 passed; final posterior/reader adapter suite, 8 passed; Ruff; strict type checks across 536 files; Markdown format check. |

The registered-model tests compare variable order, dimensions, coordinates, factorisation and saved reference log densities for all 23 vocabulary-growth models. These checks support unchanged model calculations. They do not approve reuse of an old fit. Executable code signatures can change when a shared call replaces a local expression, so the existing fit-compatibility rules still apply.

The new tests cover two-column dependence arrays, missing pairs, zero mutual-information scores, descriptive table layout, required SVG failures, sample copies, dimension order, duplicate labels, unsupported report queries, missing and invalid files, unrounded diagnostic boundaries, missing diagnostics and file permission failures. Permission checks run in child processes with three explicit POSIX masks. Windows sharing tests and other unavailable platform or optional-dependency checks were skipped.

An initial vocabulary-growth run found three failures in a paper-helper test fixture that did not supply the newly imported dependency, plus one sandbox restriction on its local HTTP server. The fixture now supplies the actual project helper. The final full fast run passed with loopback access. Two initial US births tests could not link a local PyTensor C compilation because the system linker could not find `d64`. Tests then used `PYTENSOR_FLAGS=cxx=`. The scientific graph checks therefore ran with the interpreter backend rather than the unavailable C linker.

The descriptive adapter still runs the shared Anderson-Darling calculation before dropping its two rows. A synthetic timing check compared the full original and shared summaries for 10,000 normal observations. All retained values matched. Median times over three runs were 2.79 seconds and 2.84 seconds. The added Anderson-Darling calculation alone took a median 0.062 seconds for one million normal observations. These timings do not establish production performance or the cost of every data distribution.

All 79 Python cells in the seven edited Quarto files compiled as source. The new reader tests execute the boosting template's setup and parsing helpers. Complete reports were not rendered from production fit artefacts. No production fit was run or altered. The projects' full slow fitting suites have not been run.

## Review and release state

The downstream edits are committed on `dev/codex/shared-refactors` in each repository. These branches preserve the changes even if the temporary worktree directories are later removed. The original downstream checkouts and their existing local edits remain untouched. The worktrees and commits are available below. The library changes are on `dev/codex/downstream-shared-helpers` in this checkout.

| Repository                      | Worktree                                                        | Commit                                     |
| ------------------------------- | --------------------------------------------------------------- | ------------------------------------------ |
| Language and reading predictors | `/private/tmp/dse-shared-refactors/language-reading-predictors` | `57ff51dd14b09494abe829b11a98b6660bd3f12e` |
| Vocabulary growth               | `/private/tmp/dse-shared-refactors/vocabulary-growth`           | `dc17c0ed3b4c7043142e3eebdebcff125986e96b` |
| US birth certificates           | `/private/tmp/dse-shared-refactors/us-birth-certificates`       | `eee5efcd517fff55092d7c0c7188811c30fdc86d` |

The new library APIs are unreleased. Downstream projects retain their `v0.16.2` pins and only import APIs already present in that release. After a library release, their permission adapters and per-chain diagnostic calculations can adopt the new public functions. Their own thresholds, exemptions and publication decisions must remain explicit. No branches have been pushed and no pull requests have been opened.
