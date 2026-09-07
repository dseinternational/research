> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words HSGP BFMI MCMC reff PyTensor expit lengthscale nonzero nonempty unassessable exceedance nonfinite plotcollection invlogit -->

# Migrating to 0.13.0

Version 0.13.0 changes Gaussian-process approximations, convergence decisions and upload results. Its tag is published. The checklist below records the original upgrade requirements, not a live record of each consumer's completion. Projects moving from an older version directly to [0.14.0](migrating-to-0.14.md) must still address these changes. Updating a dependency pin does not refit models.

## Preserve the full HSGP design during refits

The boundary half-width is now `L = (max(X) - min(X)) / 2 * c`, consistent with PyMC's midpoint centring. Previously it used `max(abs(X)) * c`. Calibration also increases the basis size when a larger boundary floor requires it. This changes the prior approximation and can change the number of sampled weights. Do not score old draws using a newly constructed basis.

Both `build_hsgp_1d` and `build_tau_modifier` accept explicit `L` and `center` arguments. For a design without lengthscale calibration, compute and record these values once from the full standardised inputs, then pass the same values to every fit and evaluation.

```python
center = float((x_full.min() + x_full.max()) / 2)
L = float((x_full.max() - x_full.min()) / 2 * c)
design = {"m": m, "L": L, "center": center}

# Inside separate PyMC models, with the same priors and input standardisation.
full_curve = build_hsgp_1d("f", x_full, **design)
subset_curve = build_hsgp_1d("f", x_subset, **design)
```

`L` must contain the evaluated inputs strictly inside `center +/- L`. `center` requires explicit `L`. With explicit `L`, `c` is ignored and `ls_range` is rejected; preserve the calibrated basis size instead of recalibrating it on each subset. Freezing only `L` is insufficient because PyMC would otherwise recalculate the midpoint on each subset. A one-row subset is supported when the boundary is explicit.

The regression test uses asymmetric full inputs `[-2.4712, -1, 0, 0.5, 1.9]`. It evaluates both constructors with identical nonzero weights and verifies identical function values on the remaining rows after dropping the minimum, and on a single retained row. Separate tests compare approximate covariance matrices with the exact matrices and verify basis sizes of 70 and 28 for the reviewer's boundary-floor examples. These checks do not establish approximation accuracy for all priors or prediction ranges.

## Read diagnostic status explicitly

`diagnostics_summary.json` stores unavailable or non-finite numbers as JSON `null`, including nested amendments. It never writes bare `NaN` or `Infinity` tokens. The returned payload and optional table cache contain the same sanitised values.

The new `scan_completed` field is true when a nonempty R-hat/ESS summary was reduced successfully, even if some parameter diagnostics are unavailable. It is false when the scan raised or returned no parameters. Check `unassessable_parameters` and `checks.diagnostics_assessable` separately. A completed scan does not imply convergence. Readers must handle null extrema before converting them to floats. Older files lack `scan_completed`; preserve a conservative fallback for those files.

The gate requires R-hat and both effective-sample-size columns. Missing information can turn a previous pass into a review result. `sampling_quality` raises for an empty summary and continues to return unassessable parameter names alongside available extrema. BFMI lists and NumPy arrays are accepted; empty or non-finite arrays have no usable minimum.

## Update LOO summaries and relative-efficiency wrappers

`loo_summary_row` rejects non-finite thresholds, including a non-finite `loo.good_k`. A zero threshold remains valid. `pareto_k_above` retains its literal threshold-exceedance meaning. Two new columns make unavailable diagnostics visible.

- `pareto_k_nonfinite` counts non-finite values.
- `pareto_k_unusable` counts values that are either non-finite or above the threshold. Use this count for an unusable-observation share. Do not add the two other counts because positive infinity belongs to both.

The public `SampledParametersUnavailableError` identifies missing sampled-parameter metadata. `reff_or_default` returns `None` only when parameter names cannot be obtained. A missing named posterior variable raises `KeyError`; an empty selection raises `ValueError`. Other reader and calculation errors propagate. Consumer wrappers should delegate to the shared fallback and supply their metadata reader.

```python
return shared_loo.reff_or_default(
    trace,
    names=names,
    attr_reader=read_sampled_parameters_attr,
    label=label,
    warn=warn,
)
```

## Use raw upload paths for local comparisons

`BlobUploadResult.urls` and `prefix_url` are percent-encoded URLs. The new `relative_paths` field contains raw POSIX paths, aligned with `urls` and excluding skipped files. Compare local filenames with `relative_paths`; stripping a URL prefix leaves encoded names that differ from filenames containing spaces, plus signs or non-ASCII characters. Each uploaded URL is `prefix_url + quote(relative_path, safe="/")`.

`report_url` identifies only the uploaded root `index.html`. Use that field instead of selecting the first URL ending in `/index.html`, which may select a nested document. HTTP verification should use the returned URLs or encode raw relative paths once before making requests.

## Other changed behaviour

- `save_plotcollection(close=True)` closes only the collection's figures. It leaves unrelated figures open.
- Ordinary least squares rejects constant predictors and non-finite observations. HSGP rejects invalid input ranges, basis sizes and calibration bounds. ROPE half-widths must be finite and non-negative.
- Interval summaries use the same finite draws for medians and interval bounds.
- `invlogit` supports numeric scalars and NumPy arrays. Use `pm.math.sigmoid` for symbolic PyMC expressions.

## Consumer adoption checklist

These changes are pending in the consuming repositories. Complete and test them as part of adopting the new tag; this library PR does not change their pins or source.

- [ ] In language-reading-predictors, update `statistical_models/hsgp.py`, `factories/mechanism.py`, `fitted_payloads.py` and the LOO refit path to persist and forward `m`, `L` and `center`. Remove the `max(abs(X))` boundary calculation and subset-dependent `hsgp_c_for` reconstruction. Extend the consumer test to execute the shared constructor and compare full/subset functions rather than repeating a boundary formula. Refit affected models and rebuild their held-out evaluations with the same recorded design.
- [ ] In language-reading-predictors, update `statistical_models/convergence.py` to handle null extrema and the explicit scan status. Pass the structured upload result through `storage.py` so `scripts/upload.py` uses its root `report_url`.
- [ ] In vocabulary-growth, delegate `loo_reff.reff_or_default` through the shared `attr_reader` hook. Test unavailable metadata, empty selections, missing named variables and reader errors. Audit LOO report counts and use the explicit unusable count where appropriate.
- [ ] In vocabulary-growth, update `storage._verify_report_upload` to use `result.relative_paths`. Update `publication_checks.verify_published` to use encoded URLs and test names with spaces, plus signs and non-ASCII characters. Use `scan_completed` in `models.common.enforce_convergence_gate` so unavailable diagnostics receive the right explanation while still failing the gate.
- [ ] After the library release and consumer changes are reviewed, update each consumer pin and lockfile, run its relevant tests, and complete any required refits before accepting new reported results.

The cited language-reading-predictors low-BFMI test passes against this branch in that project's environment. A check using vocabulary-growth's actual `unpublished_assets` function confirms that the new raw paths match the uploaded files, while its current URL-stripping approach still fails for encoded names. Azure clients were simulated. These targeted checks do not replace the pending consumer migrations or their full test suites.
