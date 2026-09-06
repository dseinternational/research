# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Permutation evaluation with caller-supplied donors and scoring contracts."""

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

type Predictor = Callable[[Any, pd.DataFrame], ArrayLike]
type Scorer = Callable[[np.ndarray, np.ndarray], float]
type ScoreDirection = Literal["higher_is_better", "lower_is_better"]


@dataclass(frozen=True)
class PermutationDeltas:
    """A baseline and raw importance repeats, without a reporting convention.

    Field reassignment is prevented; the dictionary and arrays remain mutable.
    All returned arrays are newly allocated.
    """

    baseline_score: float
    """Score from the original feature values on exactly ``evaluated_rows``."""

    deltas: dict[Hashable, np.ndarray]
    """Block key to float64 repeat deltas. Positive means the score worsened."""

    evaluated_rows: np.ndarray
    """Evaluated positions in the original frame, in ascending row order."""


def _numeric_vector(values: ArrayLike, *, index: pd.Index, label: str) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        raise ValueError(f"{label} must not be a masked array; express missing values explicitly")
    if isinstance(values, pd.Series):
        if not values.index.identical(index):
            raise ValueError(f"{label} Series index must exactly match the evaluated frame index")
        numeric = pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype)
        if not numeric or pd.api.types.is_complex_dtype(values.dtype):
            raise TypeError(f"{label} must contain real numeric values")
        try:
            with np.errstate(over="raise", invalid="raise"):
                array = values.to_numpy(dtype=float, na_value=np.nan)
        except FloatingPointError as exc:
            raise ValueError(f"{label} overflowed during float64 conversion") from exc
    else:
        array = np.asarray(values)
        if array.dtype.kind not in "biuf":
            raise TypeError(f"{label} must contain real numeric values")
    if array.ndim != 1 or len(array) != len(index):
        raise ValueError(f"{label} must be one-dimensional with one value per frame row")
    try:
        with np.errstate(over="raise", invalid="raise"):
            return np.array(array, dtype=np.float64, copy=True)
    except FloatingPointError as exc:
        raise ValueError(f"{label} overflowed during float64 conversion") from exc


def _positions(values: ArrayLike, *, n_rows: int, label: str, ndim: int) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        raise ValueError(f"{label} must not be a masked array")
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ValueError(f"{label} must have {ndim} dimensions")
    if array.dtype.kind not in "iu":
        raise TypeError(f"{label} must contain integer row positions")
    if np.any(array < 0) or np.any(array >= n_rows):
        raise ValueError(f"{label} contains an out-of-range row position")
    return np.array(array, dtype=np.intp, copy=True)


def _validate_frame_and_callbacks(
    frame: pd.DataFrame, predict: Predictor, score: Scorer, score_direction: ScoreDirection
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("X must be a pandas.DataFrame")
    if not len(frame):
        raise ValueError("X must contain at least one evaluation row")
    if not frame.columns.is_unique:
        raise ValueError("X must have unique column labels")
    if not callable(predict) or not callable(score):
        raise TypeError("predict and score must be callable")
    if not isinstance(score_direction, str) or score_direction not in ("higher_is_better", "lower_is_better"):
        raise ValueError("score_direction must be higher_is_better or lower_is_better")


def _blocks_and_donors(
    frame: pd.DataFrame,
    column_blocks: Mapping[Hashable, Sequence[Hashable]],
    donor_indices: Mapping[Hashable, ArrayLike],
) -> tuple[dict[Hashable, np.ndarray], dict[Hashable, np.ndarray]]:
    if not isinstance(column_blocks, Mapping) or not isinstance(donor_indices, Mapping):
        raise TypeError("column_blocks and donor_indices must be mappings")
    if column_blocks.keys() != donor_indices.keys():
        raise ValueError("donor_indices must have exactly the column_blocks keys")
    blocks = {}
    plans = {}
    repeat_count = None
    for key, labels in column_blocks.items():
        if isinstance(labels, str):
            raise TypeError("Each column block must be a sequence of column labels, not a string")
        columns = frame.columns.get_indexer(list(labels))
        if not len(columns):
            raise ValueError("Column blocks must not be empty; filter absent blocks explicitly")
        if np.any(columns < 0):
            raise ValueError("A column block refers to a column absent from X")
        if len(np.unique(columns)) != len(columns):
            raise ValueError("A column block must not repeat a column")
        plan = _positions(donor_indices[key], n_rows=len(frame), label="Donor plan", ndim=2)
        if plan.shape[1] != len(frame):
            raise ValueError("Each donor plan must contain one donor for every full-frame row")
        if not plan.shape[0]:
            raise ValueError("Each donor plan must contain at least one repeat")
        if repeat_count is not None and plan.shape[0] != repeat_count:
            raise ValueError("All donor plans must have the same repeat count")
        repeat_count = plan.shape[0]
        blocks[key] = columns
        plans[key] = plan
    return blocks, plans


def _evaluate(
    estimators: tuple[Any, ...],
    frame: pd.DataFrame,
    target: np.ndarray,
    folds: tuple[np.ndarray, ...],
    evaluated_rows: np.ndarray,
    predict: Predictor,
    score: Scorer,
) -> float:
    predictions = np.empty(len(frame), dtype=float)
    for estimator, rows in zip(estimators, folds, strict=True):
        # Each callback gets its own frame, including when indexes repeat.
        subset = frame.iloc[rows].copy(deep=True)
        prediction = _numeric_vector(predict(estimator, subset), index=frame.index[rows], label="Predictions")
        if not np.isfinite(prediction).all():
            raise ValueError("Predictions must be finite on every declared evaluation row")
        predictions[rows] = prediction
    # Copies keep routine callback mutations out of the next repetition.
    score_value = score(target[evaluated_rows].copy(), predictions[evaluated_rows].copy())
    if np.ma.isMaskedArray(score_value):
        raise ValueError("score must return an unmasked finite value")
    value = np.asarray(score_value)
    if value.ndim != 0 or value.dtype.kind not in "iuf":
        raise TypeError("score must return a single real numeric value")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("score must return a finite value")
    return result


def _permutation_deltas(
    estimators: tuple[Any, ...],
    frame: pd.DataFrame,
    target: np.ndarray,
    folds: tuple[np.ndarray, ...],
    evaluated_rows: np.ndarray,
    blocks: dict[Hashable, np.ndarray],
    donors: dict[Hashable, np.ndarray],
    predict: Predictor,
    score: Scorer,
    score_direction: ScoreDirection,
) -> PermutationDeltas:
    if not np.isfinite(target[evaluated_rows]).all():
        raise ValueError("Targets must be finite on every declared evaluation row")
    baseline = _evaluate(estimators, frame, target, folds, evaluated_rows, predict, score)
    deltas = {}
    for key, columns in blocks.items():
        values = np.empty(len(donors[key]), dtype=float)
        for repeat, donor_rows in enumerate(donors[key]):
            permuted = frame.copy(deep=True)
            for column in columns:
                # Taking the ExtensionArray preserves categorical, nullable and
                # Arrow-backed dtypes without aligning donor index labels.
                permuted.isetitem(int(column), frame.iloc[:, column].array.take(donor_rows))
            permuted_score = _evaluate(estimators, permuted, target, folds, evaluated_rows, predict, score)
            delta = baseline - permuted_score if score_direction == "higher_is_better" else permuted_score - baseline
            if not np.isfinite(delta):
                raise ValueError("The score difference overflowed")
            values[repeat] = delta
        deltas[key] = values
    return PermutationDeltas(baseline, deltas, evaluated_rows.copy())


def heldout_permutation_deltas(
    estimator: Any,
    X: pd.DataFrame,
    y: ArrayLike,
    column_blocks: Mapping[Hashable, Sequence[Hashable]],
    *,
    donor_indices: Mapping[Hashable, ArrayLike],
    predict: Predictor,
    score: Scorer,
    score_direction: ScoreDirection,
) -> PermutationDeltas:
    """Score joint column permutations with one estimator on one held-out frame.

    Parameters
    ----------
    estimator
        An already-fitted estimator, passed unchanged to ``predict``.
    X
        The nonempty held-out DataFrame. Column labels must be unique; duplicate
        row index labels are allowed. Every row is scored in its original order.
    y
        One finite real target per row. Series indexes must exactly match
        ``X.index``; other arrays are explicitly positional.
    column_blocks
        Block keys mapped to nonempty sequences of column labels. Overlapping
        blocks are allowed; duplicate or absent columns within a block are not.
        Filter missing features or empty blocks in the project adapter.
    donor_indices
        The same block keys mapped to integer arrays of shape
        ``(n_repeats, len(X))``. Each entry gives a positional donor for the
        corresponding row. All plans require the same positive repeat count.
        Repeated donor positions are allowed; no permutation is generated here.
    predict
        Callback ``predict(estimator, frame)`` returning one finite numeric
        prediction per row. Choose any probability column or response transform
        here; the library does not infer classification or regression behavior.
    score
        Callback ``score(target, prediction)`` returning one finite scalar.
        Both arguments are float64 arrays in evaluated-row order.
    score_direction : {"higher_is_better", "lower_is_better"}
        Direction defining improvement. Returned deltas are positive when
        permutation makes the score worse.

    Returns
    -------
    PermutationDeltas
        Baseline score, raw repeated deltas in block order, and all row positions.
        An empty block mapping with empty donor mapping returns the baseline and
        empty deltas. Table columns and standard-deviation conventions stay local.

    Notes
    -----
    Targets and predictions are converted to float64 before scoring. This can
    differ from a consumer's native float32 scorer. Input frames, targets and
    donor plans are not modified; routine in-place predictor/scorer mutations
    operate on copies. DataFrame column dtypes and row indexes are retained.
    Mutable objects stored inside object-dtype cells are not recursively copied,
    and estimator instances are not cloned. Callbacks must not mutate those
    cell contents or estimator state.

    Donor plans determine the randomization contract. For deterministic
    callbacks, repeating a plan reproduces the same deltas. The caller chooses
    row, subject or other donors and any balancing of the evaluation sample.
    """
    _validate_frame_and_callbacks(X, predict, score, score_direction)
    target = _numeric_vector(y, index=X.index, label="Targets")
    blocks, donors = _blocks_and_donors(X, column_blocks, donor_indices)
    rows = np.arange(len(X), dtype=np.intp)
    return _permutation_deltas((estimator,), X, target, (rows,), rows, blocks, donors, predict, score, score_direction)


def pooled_oof_permutation_deltas(
    estimators: Iterable[Any],
    X: pd.DataFrame,
    y: ArrayLike,
    test_indices: Iterable[ArrayLike],
    column_blocks: Mapping[Hashable, Sequence[Hashable]],
    *,
    donor_indices: Mapping[Hashable, ArrayLike],
    predict: Predictor,
    score: Scorer,
    score_direction: ScoreDirection,
) -> PermutationDeltas:
    """Score each repeat once over pooled, explicitly declared out-of-fold rows.

    Parameters
    ----------
    estimators
        One already-fitted estimator per held-out fold, in ``test_indices`` order.
    X
        Full feature DataFrame, including rows that may act only as donors.
        Column labels must be unique; duplicate row index labels are allowed.
    y
        Full target vector. Series indexes must exactly match ``X.index``.
        Targets must be finite on evaluated rows; unscored rows may be missing.
    test_indices
        Nonempty, one-dimensional integer row positions per estimator. Positions
        must be unique within and across folds and within the full frame. Partial
        coverage is allowed explicitly; no row is omitted because of its value
        or prediction. Every repeat evaluates exactly this same coverage.
    column_blocks, donor_indices, predict, score, score_direction
        As in :func:`heldout_permutation_deltas`. Donor plans cover all ``X``
        rows and may draw donors from outside the evaluated folds. To share one
        subject donor map across blocks, provide that same plan under each key.

    Returns
    -------
    PermutationDeltas
        The original pooled score, raw repeated changes in that pooled score,
        and evaluated positions sorted into original frame order. The score is
        computed once over all held-out predictions, not averaged across folds.

    Notes
    -----
    Float64 scoring, callback copies and metadata preservation follow
    :func:`heldout_permutation_deltas`. No estimators are trained and no donor
    rows are generated. In particular, permuting only within singleton-child
    folds cannot assess a child-constant predictor; the adapter must provide
    the intended donor plan across children.
    """
    _validate_frame_and_callbacks(X, predict, score, score_direction)
    target = _numeric_vector(y, index=X.index, label="Targets")
    fitted = tuple(estimators)
    folds = tuple(_positions(rows, n_rows=len(X), label="Fold rows", ndim=1) for rows in test_indices)
    if not fitted or len(fitted) != len(folds):
        raise ValueError("There must be exactly one estimator per nonempty fold")
    covered = np.zeros(len(X), dtype=bool)
    for rows in folds:
        if not len(rows):
            raise ValueError("Fold rows must not be empty")
        if len(np.unique(rows)) != len(rows) or covered[rows].any():
            raise ValueError("Evaluation rows must be unique within and across folds")
        covered[rows] = True
    evaluated_rows = np.flatnonzero(covered)
    blocks, donors = _blocks_and_donors(X, column_blocks, donor_indices)
    return _permutation_deltas(
        fitted, X, target, folds, evaluated_rows, blocks, donors, predict, score, score_direction
    )
