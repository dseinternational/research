> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words HSGP ExpQuad PyMC NumPy DataFrame float64 float32 int32 ndarray nonfinite parametrization noncentered heldout OOF RMSE USBC LRP VG asdict keepdims quantile quantiles symlink symlinks nullable dtype dtypes ls pathlib prob allclose nanmean lengthscale dataclasses centered -->

# Adopt the shared helpers in one upgrade

PR #101 collects the library side of the downstream consolidation work into one release batch. The APIs are unreleased, and the release version has not been assigned. Each downstream project can update its dependency and lockfile once, then replace local implementations through small adapters in the same upgrade PR. This library PR does not migrate downstream production code or establish that saved fits are compatible with changed implementation identities.

The earlier guides cover [file writing and provenance](shared-file-provenance.md), [labelled samples, predictive summaries and likelihood aggregation](shared-predictive-arrays.md), and [report assets and feature groups](shared-assets-and-feature-groups.md). This guide covers the remaining additions and the checks needed for a combined migration.

## Promote a completed directory

`storage.directories.promote_directory` moves a completed staging tree to its destination and retains the previous destination at an explicit backup path. The caller supplies the lock and checks that the staged output is complete. Every writer of these paths, their parent directories and the staged tree must honour the same exclusion rule.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from dse_research_utils.storage.directories import promote_directory

with TemporaryDirectory() as directory:
    # Resolve this trusted temporary parent because macOS may alias it with a symlink.
    root = Path(directory).resolve()
    staged, destination, backup = (root / name for name in ("staged", "current", "previous"))
    staged.mkdir()
    destination.mkdir()
    (staged / "result.txt").write_text("new", encoding="utf-8")
    (destination / "result.txt").write_text("old", encoding="utf-8")
    result = promote_directory(staged, destination, backup=backup, lock=Lock())
    assert (result.destination / "result.txt").read_text(encoding="utf-8") == "new"
    assert result.backup == backup
    assert (backup / "result.txt").read_text(encoding="utf-8") == "old"
```

The example has one writer. A thread lock only coordinates threads that use that lock; projects with several writer processes must supply a process lock. A null context is appropriate only when the caller already has exclusive access. Parent directories must exist. The helper rejects occupied backup paths, nested or aliased paths, symlinks in any path component and known cross-filesystem moves before changing the destination. Resolving a trusted parent deliberately is different from resolving an untrusted final symlink to bypass these checks.

Replacement takes two renames, so readers can observe a gap between the old and new destination. Projects requiring uninterrupted reads need their own reader coordination or versioned-directory design. If promotion fails, the helper attempts to restore the original destination only when the paths still support safe recovery. It preserves the primary exception and attaches recovery details. It performs no recursive deletion. Backup retention, failed-stage preservation and cleanup remain project decisions. A cleanup failure after success must not be reported as a failed promotion.

## Read file state before applying report rules

`report.readers.read_csv` and `read_json` return a `FileRead` record with a `present`, `missing` or `invalid` status. Invalid reads include a reason and exception class. `ReportData.read_summary` and `ReportData.read_json` expose the same facts through an existing model-directory resolver. An indexed CSV can pass `index_col` explicitly.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from dse_research_utils.report.readers import nearest_row, read_csv, read_json

with TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "null.json").write_text("null", encoding="utf-8")
    (root / "broken.json").write_text("{", encoding="utf-8")
    (root / "empty.csv").write_text("age_months,estimate\n", encoding="utf-8")
    assert read_json(root / "missing.json").status == "missing"
    parsed = read_json(root / "null.json")
    assert parsed.status == "present" and parsed.value is None
    assert read_json(root / "broken.json").reason == "parse_error"
    table = read_csv(root / "empty.csv")
    assert table.status == "present" and table.value.empty

grid = pd.DataFrame({"age": [24, 30, 36], "estimate": [1.0, 2.0, 4.0]}, index=[8, 8, 9])
row = nearest_row(grid, key="age", at=33, max_distance=3)
assert row["estimate"] == 2.0  # First row wins the equal-distance tie.
assert nearest_row(grid, key="age", at=40, max_distance=3) is None
```

A parsed file still needs the project's schema, freshness and scientific visibility checks. JSON `null` is present, a header-only CSV is present and empty, and a zero-byte document is invalid. Pandas can accept a short CSV row by filling missing cells, so successful parsing does not establish a complete table. The new JSON reader rejects nonfinite numeric tokens and floating numbers that overflow. The existing `ReportData.load_summary` and `load_json` methods retain their return and exception behaviour, including the old JSON parser's permissive treatment of nonfinite numbers.

`nearest_row` skips unusable keys, retains the original row label and selects the first positional tie. With no distance bound it selects an endpoint outside the grid; that choice supplies neither interpolation nor evidence of support at the requested value. Exact distances for integer keys and integral queries avoid unsigned subtraction, signed overflow and false ties from rounded integer queries. `ReportData.value_at` delegates row selection to this helper.

LRP must keep its diagnostic-column allowlist, unknown-file rules and later trace-based withholding checks. VG must retain each report's accepted-fit checks. An invalid file needs a visible failure state chosen by the report; it should not automatically become a reassuring pending-fit placeholder. Parsing once also cannot freeze a visibility decision that may change later in the render.

## Choose interval axes and missing-value rules explicitly

`statistics.array_intervals.equal_tail_interval` returns lower and upper bounds. Callers supply coverage, sample axes and a nonfinite-value policy. Unreduced axes retain their order; `keepdims=True` retains reduced axes with length one. Reducing every axis returns two zero-dimensional arrays.

```python
import numpy as np

from dse_research_utils.statistics.array_intervals import equal_tail_interval

draws = np.array([[[0.0, 10.0], [2.0, 12.0]], [[4.0, 14.0], [6.0, np.nan]]])
lower, upper = equal_tail_interval(draws, prob=0.90, axis=(0, 1), nonfinite="omit_nan")
np.testing.assert_allclose(lower, [0.3, 10.2])
np.testing.assert_allclose(upper, [5.7, 13.8])
assert lower.shape == (2,)

# The project selects its own central estimate and matching missing-value rule.
means = np.nanmean(draws, axis=(0, 1))
np.testing.assert_array_equal(means, [3.0, 12.0])
```

`propagate` preserves NaNs, `omit_nan` removes only NaNs, `omit_nonfinite` removes NaNs and infinities, and `raise` rejects nonfinite samples. Policies retaining infinities follow NumPy's linear interpolation, which can return NaN or infinity even at full coverage. Empty or entirely omitted slices return NaN bounds without warnings. Finite arithmetic overflow raises. Masked arrays require an explicit conversion first.

The helper converts samples to float64. This is an explicit numerical contract and can change results previously calculated in float32. Coverage one is supported. USBC's adapter should keep its existing coverage restriction if compatibility requires it, preserve its default NaN propagation and retain its mean-based summary. The older shared interval helpers keep their coverage defaults, finite filtering and percentile arithmetic; they have not been redirected through this new API. A mean and a median are different summaries, even when their interval bounds match.

## Save HSGP geometry and inject the covariance

An HSGP approximates a smooth random function using finitely many basis functions. `statistics.models.hsgp_design.HSGPDesign` records their count `m`, domain half-width `L` and centre. `calibrate_hsgp_1d` applies PyMC's recommendation for an ExpQuad covariance to a declared domain and lengthscale range. `HSGPDesign.from_domain` instead takes an explicit basis count and domain expansion factor. Both return realised geometry that can be stored with the fit.

```python
import json
from dataclasses import asdict

import numpy as np
import pymc as pm

from dse_research_utils.statistics.models.hsgp_design import (
    HSGPDesign,
    calibrate_hsgp_1d,
    create_hsgp,
)

design = calibrate_hsgp_1d([-2.5, 1.9], ls_range=(0.2, 0.8), c_floor=None)
saved = json.dumps(asdict(design), allow_nan=False)
restored = HSGPDesign(**json.loads(saved))
gp = create_hsgp(restored, cov_func=pm.gp.cov.ExpQuad(1, ls=0.4))
rows = np.array([[-1.0], [0.0], [1.0]])
full_basis, _ = gp.prior_linearized(rows)
subset_basis, _ = gp.prior_linearized(rows[1:])
np.testing.assert_array_equal(full_basis.eval()[1:], subset_basis.eval())
```

`c_floor=None` preserves VG's unmodified calibration. An explicit floor expands the boundary and recalculates the basis count to retain frequency coverage. Calibration depends on the installed PyMC version. Save the realised scalars instead of recalibrating when query rows change. The calibration label identifies the recipe; it is not a model identity or a numerical-library version. Validate input units, scaling and approximation accuracy in the project. Unrepresentable geometry is rejected with a request to rescale inputs.

`create_hsgp` takes the caller's one-dimensional covariance and creates no random variables or model names. A public PyMC call fixes the saved centre before query rows enter the object. It supports centered and noncentered coefficients and an explicit first-basis removal option. Later numeric or symbolic query inputs must be finite, use the declared units and lie inside the saved domain. The caller remains responsible for those inputs and the covariance parameters.

VG can replace its low-level HSGP construction and private centre assignment while retaining the bounded lengthscale prior, variable names and order, dimensions, observed-row projection and anchoring. LRP can put these scalars into its existing mechanism design. The high-level `build_hsgp_1d` keeps its prior and graph construction; its calibration calculation now delegates to the shared helper. Neither project should append plotting rows and then recalibrate a saved fit.

## Keep held-out and pooled permutation scoring explicit

`ml.permutation.heldout_permutation_deltas` scores one fitted estimator on one held-out frame. `pooled_oof_permutation_deltas` uses each declared fold's fitted estimator, then scores the combined predictions once in original row order. Each accepts column-label blocks, explicit donor row positions, a prediction callback, a score callback and a required score direction. Positive deltas mean that permutation worsened the score.

```python
import numpy as np
import pandas as pd

from dse_research_utils.ml.permutation import (
    heldout_permutation_deltas,
    pooled_oof_permutation_deltas,
)


class FeaturePrediction:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["x"].to_numpy()


def predict(estimator: FeaturePrediction, frame: pd.DataFrame) -> np.ndarray:
    return estimator.predict(frame)


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}, index=[10, 10, 20, 30])
y = pd.Series(X["x"].to_numpy(), index=X.index)
blocks = {"signal": ["x"]}
donors = {"signal": np.array([[3, 2, 1, 0], [1, 2, 3, 0]])}
heldout = heldout_permutation_deltas(
    FeaturePrediction(), X, y, blocks,
    donor_indices=donors, predict=predict, score=rmse, score_direction="lower_is_better",
)
pooled = pooled_oof_permutation_deltas(
    [FeaturePrediction() for _ in range(4)], X, y, [[0], [1], [2], [3]], blocks,
    donor_indices=donors, predict=predict, score=rmse, score_direction="lower_is_better",
)
assert heldout.baseline_score == pooled.baseline_score == 0.0
np.testing.assert_array_equal(heldout.deltas["signal"], pooled.deltas["signal"])
assert np.all(pooled.deltas["signal"] > 0)
```

Each donor array has shape `(repeat, full_frame_row)`. Every block uses the same positive repeat count. A row can donate more than once, which permits LRP's unequal-subject wrapping rule. The library generates no random numbers. USBC retains its sequential block/repeat random-number consumption; LRP retains its subject maps, per-repeat seeds and shared map across blocks. Adapters convert positional feature blocks to labels, handle missing features explicitly and keep result-table names, group IDs and standard-deviation conventions.

The example has one row in each fold. Restricting donors to those folds would leave every feature unchanged and give zero importance. The pooled API therefore accepts donor positions from the full frame. It also permits partial evaluation coverage when declared by the fold positions. Rows must be unique within and across folds. Missing targets are allowed only outside scored rows; invalid scored targets or predictions raise rather than silently changing the evaluation population.

Frames retain their column dtypes and row indexes, including categorical columns and duplicate index labels. A target or prediction Series must have an exactly matching index; plain arrays are positional. Scoring uses float64 targets and predictions. Copies protect ordinary column assignments and scorer-array edits. Nested mutable cell objects and estimator state are not cloned, so callbacks must not mutate them. The returned record contains the baseline, raw deltas and evaluated positions; it does not select an estimator, train a model, average fold scores or choose a reporting summary.

USBC must keep its positive-class probability selection and average-precision scorer where those are currently used. Its explanation-sample selection determines the evaluated prevalence and population. LRP must retain pooled RMSE and its subject-level donor design. The older `ml.importance` functions remain unchanged and are not a substitute for pooled out-of-fold scoring.

## Complete each downstream upgrade

| Project                     | Adapters to include where local implementations remain                                                                                                                                                                                        | Checks beyond helper unit tests                                                                                                                                             |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| language-reading-predictors | File/provenance handling, child-level likelihood aggregation, predictive summaries, report reads, feature groups, pooled permutation and frozen HSGP geometry. Recheck adoption of the existing reliability and diagnostic-amendment helpers. | Child/fold identities, likelihood units, output table dtypes, diagnostic cache agreement, late report withholding, named model graph and saved design identity.             |
| vocabulary-growth           | File/directory handling, provenance, administration-level likelihood aggregation, predictive summaries, report assets/reads and covariance-injected HSGP construction.                                                                        | Serialisation and manifest identity, float precision, administration ordering, navigation assets, report acceptance rules, unchanged priors and projected/anchored curves.  |
| us-birth-certificates       | Existing numerical/figure helpers plus provenance, file writes, feature groups, held-out permutation and axis-aware intervals.                                                                                                                | Existing table schemas and group IDs, category metadata, mean-based summaries, missing-value policy, probability column, score direction and explanation-sample definition. |

First update the dependency and lockfile to the eventual release containing this PR. Then route local compatibility wrappers through the shared functions one area at a time. Compare numerical values, labelled arrays or named graphs, persisted file schemas and report visibility separately. Full project tests, notebook/report imports and representative artefact checks should run against the built package. A library helper test cannot verify a consumer adapter that has not been written.

The development checks compare synthetic fixtures with current downstream functions, including predictive tables, likelihood arrays, grouped output tables, report readers, array intervals, permutation results and HSGP graph calculations. They establish the tested contracts. They do not replace downstream migration tests, model fits, live report rendering or publication checks.

Record old and new implementation/design identities using each project's existing rules. A numerical match does not ensure that resume or publication checks accept the new code. Retain historical manifests as records of their actual producers. If the project requires a refit after an identity change, include that work explicitly in its upgrade. One library release can support this whole migration without forcing the projects to adopt identical scientific decisions.
