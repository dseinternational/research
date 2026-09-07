> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words DataArray DataFrame MultiIndex PSIS ELPD dtype float32 float64 midpoint nan nonfinite missingness probs tolist nonempty nonoverlapping -->

# Shared sample arrays and predictive summaries

Version 0.14.0 adds explicit likelihood aggregation and per-observation predictive summaries. See the [upgrade notes](migrating-to-0.14.md) for installation and the release sequence. These APIs use the library's existing NumPy, pandas and xarray dependencies. Downstream projects retain their observation-unit definitions, grouping, reporting schemas, missingness rules and fit-acceptance policies.

## Preserve observation identity when reshaping samples

`statistics.samples.sample_matrix` turns an xarray `DataArray` into an observation-by-sample NumPy matrix. Every dimension must be declared as either a sample dimension or an observation dimension. Each dimension must have explicit, unique, non-missing coordinate labels. The last declared dimension in each group varies fastest. Coordinates remain in their existing order.

The returned `SampleMatrix` contains the values, sample and observation indexes, and the declared dimension names. Its `observed_values` method checks the observed array against the saved observation indexes before flattening it. It accepts a different dimension order but rejects reordered labels, different index types and implicit coordinates. It never aligns by sorting, drops rows or assumes that equal shapes prove alignment.

```python
import xarray as xr

from dse_research_utils.statistics.samples import sample_matrix

replicated = xr.DataArray(
    [[0, 0, 1, 1], [0, 0, 1, 1]],
    dims=("child", "draw"),
    coords={"child": ["child-1", "child-2"], "draw": [10, 20, 30, 40]},
)
observed = xr.DataArray(
    [0, 1], dims="child", coords={"child": ["child-1", "child-2"]}
)
matrix = sample_matrix(
    replicated, sample_dims=("draw",), observation_dims=("child",)
)
y = matrix.observed_values(observed)
```

For an ordinary posterior trace, use `sample_dims=("chain", "draw")`. An already stacked sample index is also supported. Use `observation_dims=()` for a scalar posterior variable, which produces one row. Multiple observation dimensions produce a `MultiIndex`; this preserves the distinction between each child's repeated visits or each panel cell. No event dimension is flattened implicitly.

Arrays with positional axes need deliberate coordinate assignment in the consumer before conversion. Meaningful identity must live in the declared dimension indexes; auxiliary coordinates are not used to identify observations. Values retain their dtype and any non-finite entries, and may share memory with the source. Copy `matrix.values` before modifying the values if the input must stay unchanged.

## Compute per-observation predictive checks

`statistics.predictive.predictive_observation_checks` takes aligned numeric observations and predictive draws. It returns predictive means and medians, central interval bounds, closed-interval inclusion flags and the empirical predictive mass inside each interval. The caller supplies the probabilities. It can also request midpoint probability integral transforms, abbreviated PIT, and their empirical predictive reference variances.

```python
from dse_research_utils.statistics.predictive import predictive_observation_checks

checks = predictive_observation_checks(
    y,
    matrix.values,
    interval_probs=(0.5, 0.9),
    pit_method="midpoint",
    observation_chunk_size=256,
)
assert checks.midpoint_pit.tolist() == [0.25, 0.75]
assert checks.expected_midpoint_pit_variance.tolist() == [0.0625, 0.0625]
```

Means, medians and PIT values have one entry per observation. Bounds, inclusion flags and predictive mass have shape `(observation, interval)` in the supplied probability order. `sample_axis` can be 0 or 1. Chunking limits temporary draw-sized arrays and leaves results unchanged. Draws are equally weighted; this is not a weighted-prediction API.

Midpoint PIT counts the probability below the observation plus half the probability tied with it. If the empirical predictive distribution puts mass `p_k` on each distinct value, its reference variance is `(1 - sum(p_k**3)) / 12`. For the balanced binary example above, this is 0.0625. The continuous uniform variance of 1/12 is not the correct reference for that example. The reference is estimated from finite predictive draws.

Bounds use NumPy's linear quantiles at `(1 - p) / 2` and `1 - (1 - p) / 2`. Inclusion counts both endpoints. Actual predictive mass can differ from nominal coverage. For draws `[0, 1]`, the 90% bounds are `[0.05, 0.95]`, with no predictive draws inside. Do not label the measured mass as necessarily at least nominal coverage.

Both inputs must contain finite real numeric values and nonempty observation and sample axes. Masked arrays require explicit filtering or filling. Boolean predictive draws require explicit conversion to numeric zero/one values. Overflow raises an error. The helper neither silently removes missing values nor chooses which observation population to report.

LRP's adapter should retain its existing non-finite observation mask and apply that same mask to the matrix, row labels and groups. It must also retain its existing empty-table or zero-observation coverage result when no rows remain. VG should retain its own empty-input policy and explicitly handle invalid observations before calling. These adapters control which rows enter a report.

Scalar quantile calculations preserve float32 bounds and medians. The predictive mean uses the input reduction dtype before storage as float64. A VG adapter should compute interval widths in the bounds' dtype and then store the per-observation widths as float64 before taking group means. It should retain the difference of group means for its mean-error field. These details matter for exact numerical and table compatibility.

There is one explicit arithmetic convention to review when migrating unusual interval widths. LRP currently uses `(1 + p) / 2` for the upper quantile in these checks. It is algebraically equivalent to the shared complement formula, but floating-point rounding can differ. With draws `[0, 1]`, `p=1e-6` and observed value `0.5000005000000001`, LRP excludes the observation while the shared formula includes it. The tested ordinary reporting widths agree. A consumer using other widths should compare boundary cases before migration.

Age bands, arm labels, outcome groups, zero-count summaries and report columns remain in consumer adapters. Group-level off-floor rates use a different observation unit and remain a separate calculation. Checks that reuse observations from a fit do not establish calibration for new children or validate the model as a whole.

## Aggregate likelihood factors into explicit units

`statistics.log_likelihood.LogLikelihoodFactor` carries one likelihood array, its row dimension, one output-unit label per row and any event dimensions to sum within each row. `aggregate_log_likelihood` combines these contributions into the explicitly requested unit order.

```python
from dse_research_utils.statistics.log_likelihood import (
    LogLikelihoodFactor,
    aggregate_log_likelihood,
)

log_likelihood = xr.DataArray(
    [[[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]],
    dims=("chain", "draw", "administration"),
    coords={"chain": [3], "draw": [10, 20], "administration": [100, 105, 200]},
)
by_child = aggregate_log_likelihood(
    [
        LogLikelihoodFactor(
            values=log_likelihood,
            row_dim="administration",
            row_unit_ids=["a", "a", "b"],
        )
    ],
    unit_ids=["b", "a"],
    unit_dim="child",
)
assert by_child.values.tolist() == [[[-3.0, -3.0], [-6.0, -9.0]]]
```

Mapping each row to its own administration instead would produce three pointwise scores. Mapping to the child labels above produces two scores. Their total per draw is the same, but their pointwise values answer different predictive questions. Summing terms does not integrate a child's fitted latent effects or turn conditional prediction into prediction for a new child.

Each factor must declare all dimensions. For a matrix-valued composition likelihood, declare its cell dimensions in `event_dims`; only those dimensions are summed within a row. Sample dimensions default to `("chain", "draw")` and can be supplied explicitly. Their labels, order, types, indexes and retained sample-only auxiliary coordinates must agree across factors. The helper does not reorder or broadcast mismatched samples.

A sequence of row labels maps by position. A `DataArray` row map must use the factor's row dimension and exactly match its explicit row index. In a mask-based consumer, apply the mask to the administration labels first, for example `administration_ids[mask]`; the mask itself is not a unit map. Missing, duplicate or undeclared output labels are rejected. Repeated row labels are valid. Every requested unit must receive at least one contribution. Output coordinates, including `MultiIndex` level names, must not replace retained sample coordinates.

An individual factor may have zero rows if other factors cover all requested units. Empty event or sample axes, an empty factor list and an empty output-unit list are rejected. Scalar factors need an explicit singleton row dimension. Passing the same array object twice is rejected, but the caller still needs to select nonoverlapping factors; copies and already-derived aggregates cannot be identified reliably from values alone.

Negative infinity is retained because an impossible observation can have log likelihood `-inf`. NaN, positive infinity and arithmetic overflow are rejected. No sum skips missing values. The result is a new `DataArray` named `log_likelihood`. It is not attached to a trace automatically. Keep derived aggregates separate from their source factors when another tool might otherwise count both.

All likelihood values are converted to float64 before event or row sums. This can change a consumer's existing float32 event reduction. For example, `[-1e8, -1, -1]` sums to `-100000000` in float32 and `-100000002` after conversion to float64. Downstream comparisons match exactly for the tested float64 likelihoods, including pointwise PSIS-LOO estimates and Pareto-k values. A consumer using lower-precision likelihoods must review the numerical change before migration.

## Validate consumer adoption

Focused tests check array identity, complete predictive outputs, dtype-sensitive calculations, empirical PIT references, explicit likelihood units and invalid sums. Synthetic adapter comparisons against the current downstream implementations also check whole DataFrames and their column types, including missing-row and empty-output policies.

Before adopting these helpers, compare pointwise arrays and unit coordinates, then downstream tables and rendered captions. Similar totals alone do not establish equal held-out units or report populations. Saved-fit identity and publication rules still apply when code moves into the shared library. The [file and provenance guide](shared-file-provenance.md) describes those separate compatibility requirements.
