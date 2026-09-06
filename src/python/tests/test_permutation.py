# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Permutation evaluation contracts independent of donor-generation policy."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from dse_research_utils.ml.permutation import heldout_permutation_deltas, pooled_oof_permutation_deltas


def _predict(column, frame):
    return frame[column].to_numpy()


def _rmse(target, prediction):
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def _fixture():
    frame = pd.DataFrame({"signal": [0.0, 1.0, 2.0, 3.0], "noise": [9.0, 8.0, 7.0, 6.0]}, index=[40, 10, 30, 20])
    plan = np.array([[3, 2, 1, 0], [1, 2, 3, 0]], dtype=int)
    return frame, frame["signal"].copy(), plan


def _heldout(frame, target, plan, **kwargs):
    options = {
        "donor_indices": {"signal": plan},
        "predict": _predict,
        "score": _rmse,
        "score_direction": "lower_is_better",
    }
    options.update(kwargs)
    return heldout_permutation_deltas("signal", frame, target, {"signal": ["signal"]}, **options)


def test_heldout_returns_baseline_raw_deltas_and_all_row_positions_without_mutation():
    frame, target, plan = _fixture()
    original_frame, original_target, original_plan = frame.copy(), target.copy(), plan.copy()
    result = _heldout(frame, target, plan)
    assert result.baseline_score == 0
    np.testing.assert_array_equal(result.evaluated_rows, np.arange(4))
    np.testing.assert_array_equal(result.deltas["signal"], [np.sqrt(5), np.sqrt(3)])
    pd.testing.assert_frame_equal(frame, original_frame)
    pd.testing.assert_series_equal(target, original_target)
    np.testing.assert_array_equal(plan, original_plan)


def test_score_direction_makes_both_score_conventions_report_positive_importance():
    frame, target, plan = _fixture()
    lower = _heldout(frame, target, plan)
    higher = _heldout(frame, target, plan, score=lambda y, pred: -_rmse(y, pred), score_direction="higher_is_better")
    np.testing.assert_array_equal(lower.deltas["signal"], higher.deltas["signal"])


def test_predictor_can_explicitly_choose_positive_class_probabilities():
    frame = pd.DataFrame({"probability": [0.1, 0.2, 0.8, 0.9]})
    target = np.array([0, 0, 1, 1])
    result = heldout_permutation_deltas(
        None,
        frame,
        target,
        {"probability": ["probability"]},
        donor_indices={"probability": [[3, 2, 1, 0]]},
        predict=lambda _, x: np.column_stack((1 - x.probability, x.probability))[:, 1],
        score=average_precision_score,
        score_direction="higher_is_better",
    )
    assert result.baseline_score == 1
    assert result.deltas["probability"][0] == 1 - average_precision_score(target, [0.9, 0.8, 0.2, 0.1])


def test_joint_blocks_move_columns_together_and_reuse_the_original_frame_each_time():
    frame = pd.DataFrame({"signal": [0.0, 1.0, 2.0], "twice": [0.0, 2.0, 4.0]})
    calls = []

    def predict(_, x):
        np.testing.assert_array_equal(x.twice, 2 * x.signal)
        calls.append(x.signal.to_numpy().copy())
        return x.signal.to_numpy()

    plan = np.array([[2, 1, 0], [0, 1, 2]])
    result = heldout_permutation_deltas(
        None,
        frame,
        frame.signal,
        {"joint": ["signal", "twice"]},
        donor_indices={"joint": plan},
        predict=predict,
        score=_rmse,
        score_direction="lower_is_better",
    )
    assert result.deltas["joint"][0] > 0
    assert result.deltas["joint"][1] == 0
    np.testing.assert_array_equal(calls[-1], frame.signal)


def test_column_extension_dtypes_and_duplicate_row_indexes_are_preserved():
    frame = pd.DataFrame(
        {
            "signal": [0.0, 1.0, 2.0],
            "category": pd.Categorical(["a", "b", "a"], categories=["b", "a", "unused"], ordered=True),
            "integer": pd.array([1, None, 3], dtype="Int64"),
            "string": pd.array(["x", None, "y"], dtype="string"),
            "boolean": pd.array([True, None, False], dtype="boolean"),
        },
        index=[4, 4, 2],
    )
    original = frame.copy(deep=True)
    calls = []

    def predict(_, x):
        pd.testing.assert_index_equal(x.index, frame.index)
        pd.testing.assert_series_equal(x.dtypes, frame.dtypes)
        pd.testing.assert_index_equal(x.category.cat.categories, frame.category.cat.categories)
        assert x.category.cat.ordered
        calls.append(x.copy())
        return x.signal

    result = heldout_permutation_deltas(
        None,
        frame,
        frame.signal,
        {"all": list(frame.columns)},
        donor_indices={"all": [[2, 0, 1]]},
        predict=predict,
        score=_rmse,
        score_direction="lower_is_better",
    )
    assert result.deltas["all"][0] > 0
    for column in frame:
        expected = frame[column].iloc[[2, 0, 1]].copy()
        expected.index = frame.index
        pd.testing.assert_series_equal(calls[1][column], expected)
    pd.testing.assert_frame_equal(frame, original)


def test_multiindex_column_labels_are_supported_without_becoming_positions():
    frame = pd.DataFrame([[0.0, 9], [1.0, 8]], columns=pd.MultiIndex.from_tuples([("a", "signal"), ("b", "noise")]))
    result = heldout_permutation_deltas(
        ("a", "signal"),
        frame,
        [0, 1],
        {"signal": [("a", "signal")]},
        donor_indices={"signal": [[1, 0]]},
        predict=_predict,
        score=_rmse,
        score_direction="lower_is_better",
    )
    assert result.deltas["signal"][0] == 1


def test_callbacks_can_mutate_their_inputs_without_changing_later_repeats_or_originals():
    frame, target, plan = _fixture()
    before = frame.copy(deep=True)
    original_target = target.copy()

    def predict(_, x):
        values = x.signal.to_numpy().copy()
        x.loc[:, "signal"] = -999
        return values

    def score(y, predictions):
        result = _rmse(y, predictions)
        y[:] = -999
        predictions[:] = -999
        return result

    result = _heldout(frame, target, plan, predict=predict, score=score)
    np.testing.assert_array_equal(result.deltas["signal"], [np.sqrt(5), np.sqrt(3)])
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_series_equal(target, original_target)


def test_pooled_score_weights_rows_instead_of_averaging_unequal_fold_scores():
    frame = pd.DataFrame({"signal": [0.0, 0, 0, 0, 8]})
    result = pooled_oof_permutation_deltas(
        ["signal", "signal"],
        frame,
        np.zeros(5),
        [[0, 1, 2, 3], [4]],
        {},
        donor_indices={},
        predict=_predict,
        score=_rmse,
        score_direction="lower_is_better",
    )
    assert result.baseline_score == np.sqrt(64 / 5)
    assert result.baseline_score != (0 + 8) / 2
    assert result.deltas == {}


def test_partial_oof_coverage_uses_declared_rows_even_with_unscored_missing_targets():
    frame = pd.DataFrame({"signal": [0.0, 1.0, 8.0, 3.0, 9.0]}, index=[9, 9, 1, 8, 2])
    target = pd.Series(pd.array([None, 1, None, 3, None], dtype="Float64"), index=frame.index)
    seen = []

    def score(y, prediction):
        seen.append(y.copy())
        return _rmse(y, prediction)

    result = pooled_oof_permutation_deltas(
        ["signal", "signal"],
        frame,
        target,
        [[3], [1]],
        {"signal": ["signal"]},
        donor_indices={"signal": [[0, 2, 0, 4, 0]]},
        predict=_predict,
        score=score,
        score_direction="lower_is_better",
    )
    np.testing.assert_array_equal(result.evaluated_rows, [1, 3])
    assert result.baseline_score == 0
    assert result.deltas["signal"][0] == np.sqrt((7**2 + 6**2) / 2)
    for values in seen:
        np.testing.assert_array_equal(values, [1, 3])


def test_global_subject_donors_detect_signal_in_singleton_child_folds():
    frame = pd.DataFrame({"child_signal": [0.0, 1.0, 2.0, 3.0]})
    rows = [[0], [1], [2], [3]]
    common = dict(predict=_predict, score=_rmse, score_direction="lower_is_better")
    identity = pooled_oof_permutation_deltas(
        ["child_signal"] * 4,
        frame,
        frame.child_signal,
        rows,
        {"child": ["child_signal"]},
        donor_indices={"child": [[0, 1, 2, 3]]},
        **common,
    )
    global_donors = pooled_oof_permutation_deltas(
        ["child_signal"] * 4,
        frame,
        frame.child_signal,
        rows,
        {"child": ["child_signal"]},
        donor_indices={"child": [[3, 2, 1, 0]]},
        **common,
    )
    assert identity.deltas["child"][0] == 0
    assert global_donors.deltas["child"][0] == np.sqrt(5)


def test_unequal_subject_donors_may_repeat_rows_without_being_repaired():
    frame = pd.DataFrame({"signal": [1.0, 2.0, 3.0, 9.0]})
    result = pooled_oof_permutation_deltas(
        ["signal", "signal"],
        frame,
        frame.signal,
        [[0, 1, 2], [3]],
        {"signal": ["signal"]},
        donor_indices={"signal": [[3, 3, 3, 0]]},
        predict=_predict,
        score=_rmse,
        score_direction="lower_is_better",
    )
    assert result.deltas["signal"][0] == np.sqrt((8**2 + 7**2 + 6**2 + 8**2) / 4)


def test_reusing_explicit_plans_reproduces_deltas_independently_of_block_iteration_order():
    frame, target, plan = _fixture()
    common = dict(predict=_predict, score=_rmse, score_direction="lower_is_better")
    result = heldout_permutation_deltas(
        "signal",
        frame,
        target,
        {"signal": ["signal"], "noise": ["noise"]},
        donor_indices={"signal": plan, "noise": plan},
        **common,
    )
    reordered = heldout_permutation_deltas(
        "signal",
        frame,
        target,
        {"noise": ["noise"], "signal": ["signal"]},
        donor_indices={"noise": plan, "signal": plan},
        **common,
    )
    assert list(result.deltas) == ["signal", "noise"]
    for key in result.deltas:
        np.testing.assert_array_equal(result.deltas[key], reordered.deltas[key])


@pytest.mark.parametrize("folds", [[[0, 0], [1]], [[0, 1], [1, 2]]])
def test_duplicate_evaluation_rows_are_rejected(folds):
    frame, target, plan = _fixture()
    with pytest.raises(ValueError, match="unique within and across"):
        pooled_oof_permutation_deltas(
            ["signal", "signal"],
            frame,
            target,
            folds,
            {"signal": ["signal"]},
            donor_indices={"signal": plan},
            predict=_predict,
            score=_rmse,
            score_direction="lower_is_better",
        )


@pytest.mark.parametrize("rows", [[-1], [4], [[0]], [0.0], [True], np.ma.array([0], mask=[True])])
def test_invalid_fold_positions_are_rejected(rows):
    frame, target, plan = _fixture()
    with pytest.raises((ValueError, TypeError), match="Fold rows"):
        pooled_oof_permutation_deltas(
            ["signal"],
            frame,
            target,
            [rows],
            {"signal": ["signal"]},
            donor_indices={"signal": plan},
            predict=_predict,
            score=_rmse,
            score_direction="lower_is_better",
        )


@pytest.mark.parametrize(
    "estimators, folds", [([], []), (["signal"], []), (["signal"], [[0], [1]]), (["signal"], [np.array([], dtype=int)])]
)
def test_estimators_and_nonempty_folds_must_align(estimators, folds):
    frame, target, plan = _fixture()
    with pytest.raises(ValueError, match=r"fold|Fold"):
        pooled_oof_permutation_deltas(
            estimators,
            frame,
            target,
            folds,
            {"signal": ["signal"]},
            donor_indices={"signal": plan},
            predict=_predict,
            score=_rmse,
            score_direction="lower_is_better",
        )


@pytest.mark.parametrize(
    "plan",
    [
        [[0, 1, 2]],
        [0, 1, 2, 3],
        [[0, 1, 2, 4]],
        [[0, 1, 2, -1]],
        [[0.0, 1.0, 2.0, 3.0]],
        [[True, False, True, False]],
        np.empty((0, 4), dtype=int),
        np.ma.array([[0, 1, 2, 3]], mask=True),
    ],
)
def test_invalid_donor_plans_are_rejected_before_prediction(plan):
    frame, target, _ = _fixture()
    with pytest.raises((ValueError, TypeError), match=r"donor|Donor"):
        _heldout(frame, target, plan, predict=lambda *_: pytest.fail("Invalid plan reached prediction"))


@pytest.mark.parametrize(
    "blocks", [{"signal": []}, {"signal": ["absent"]}, {"signal": ["signal", "signal"]}, {"signal": "signal"}]
)
def test_missing_empty_duplicate_or_ambiguous_block_columns_are_rejected(blocks):
    frame, target, plan = _fixture()
    with pytest.raises((ValueError, TypeError), match=r"column|Column"):
        heldout_permutation_deltas(
            "signal",
            frame,
            target,
            blocks,
            donor_indices={"signal": plan},
            predict=_predict,
            score=_rmse,
            score_direction="lower_is_better",
        )


def test_donor_keys_and_repeat_counts_must_match_blocks():
    frame, target, plan = _fixture()
    with pytest.raises(ValueError, match="exactly"):
        _heldout(frame, target, plan, donor_indices={"other": plan})
    with pytest.raises(ValueError, match="repeat count"):
        heldout_permutation_deltas(
            "signal",
            frame,
            target,
            {"signal": ["signal"], "noise": ["noise"]},
            donor_indices={"signal": plan, "noise": plan[:1]},
            predict=_predict,
            score=_rmse,
            score_direction="lower_is_better",
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_predictions_are_never_used_to_hide_evaluation_rows(value):
    frame, target, plan = _fixture()
    calls = 0

    def predict(_, x):
        nonlocal calls
        calls += 1
        prediction = x.signal.to_numpy().copy()
        if calls > 2:
            prediction[0] = value
        return prediction

    with pytest.raises(ValueError, match="Predictions must be finite"):
        pooled_oof_permutation_deltas(
            [None, None],
            frame,
            target,
            [[0], [1, 2, 3]],
            {"signal": ["signal"]},
            donor_indices={"signal": plan},
            predict=predict,
            score=_rmse,
            score_direction="lower_is_better",
        )


@pytest.mark.parametrize("prediction", [[1, 2], [[0], [1], [2], [3]], [0, 1, 2, np.nan], [0, 1, 2, 3j]])
def test_prediction_contract_requires_one_finite_real_value_per_row(prediction):
    frame, target, plan = _fixture()
    with pytest.raises((ValueError, TypeError), match="Predictions"):
        _heldout(frame, target, plan, predict=lambda *_: prediction)


@pytest.mark.parametrize(
    "score_value",
    [
        np.nan,
        np.inf,
        -np.inf,
        [1],
        1j,
        "0.5",
        True,
        np.ma.masked,
        np.ma.array(123.0, mask=True),
        np.ma.array(123.0, mask=False),
    ],
)
def test_score_must_be_finite_real_scalar(score_value):
    frame, target, plan = _fixture()
    with pytest.raises((ValueError, TypeError), match="score must"):
        _heldout(frame, target, plan, score=lambda *_: score_value)


def test_score_difference_overflow_is_rejected():
    frame, target, plan = _fixture()
    scores = iter([-1e308, 1e308])
    with pytest.raises(ValueError, match="difference overflowed"):
        _heldout(frame, target, plan[:1], score=lambda *_: next(scores))


@pytest.mark.parametrize(
    "target", [[0, 1, 2], [[0], [1], [2], [3]], [0, 1, 2, np.nan], [0, 1, 2, np.inf], [0, 1, 2, 3j]]
)
def test_heldout_targets_must_be_finite_real_and_match_rows(target):
    frame, _, plan = _fixture()
    with pytest.raises((ValueError, TypeError), match="Targets"):
        _heldout(frame, target, plan)


def test_series_targets_and_predictions_require_exact_index_order():
    frame, target, plan = _fixture()
    with pytest.raises(ValueError, match="Targets Series index"):
        _heldout(frame, target.iloc[::-1], plan)
    with pytest.raises(ValueError, match="Predictions Series index"):
        _heldout(frame, target, plan, predict=lambda _, x: x.signal.iloc[::-1])


def test_complex_series_targets_are_rejected_without_discarding_the_imaginary_part():
    frame, _, plan = _fixture()
    with pytest.raises(TypeError, match="real numeric"):
        _heldout(frame, pd.Series([0, 1, 2, 3j], index=frame.index), plan)


def test_complex_series_predictions_are_rejected_without_discarding_the_imaginary_part():
    frame, target, plan = _fixture()
    with pytest.raises(TypeError, match="real numeric"):
        _heldout(frame, target, plan, predict=lambda _, x: pd.Series([0, 1, 2, 3j], index=x.index))


@pytest.mark.parametrize("dtype", ["bool", "boolean", "bool[pyarrow]"])
@pytest.mark.parametrize("kind", ["target", "prediction"])
def test_boolean_series_targets_and_predictions_have_consistent_numeric_handling(dtype, kind):
    frame = pd.DataFrame({"signal": [0.0, 0.0, 1.0, 1.0]}, index=[4, 2, 8, 9])
    target = frame.signal
    options = {}
    if kind == "target":
        target = pd.Series([False, False, True, True], dtype=dtype, index=frame.index)
    else:
        options["predict"] = lambda _, x: pd.Series(x.signal >= 0.5, dtype=dtype, index=x.index)
    result = _heldout(frame, target, [[3, 2, 1, 0]], **options)
    assert result.baseline_score == 0
    assert result.deltas["signal"][0] == 1


@pytest.mark.skipif(
    np.finfo(np.longdouble).max <= np.finfo(np.float64).max,
    reason="This platform does not provide a wider floating-point dtype than float64",
)
@pytest.mark.parametrize("kind", ["target", "prediction"])
def test_series_float64_conversion_overflow_is_rejected(kind):
    frame, target, plan = _fixture()
    too_large = np.longdouble(np.finfo(float).max) * np.longdouble(2)
    values = pd.Series(np.array([0, 1, 2, too_large], dtype=np.longdouble), index=frame.index)
    with pytest.raises(ValueError, match="overflowed during float64 conversion"):
        if kind == "target":
            _heldout(frame, values, plan)
        else:
            _heldout(frame, target, plan, predict=lambda *_: values)


def test_dataframe_columns_must_be_unique_and_evaluation_rows_nonempty():
    frame, target, plan = _fixture()
    frame.columns = ["signal", "signal"]
    with pytest.raises(ValueError, match="unique column"):
        _heldout(frame, target, plan)
    with pytest.raises(ValueError, match="evaluation row"):
        _heldout(frame.iloc[:0], [], np.empty((1, 0), dtype=int))


def test_scoring_contract_is_explicit_and_checked_even_without_blocks():
    frame, target, _ = _fixture()
    with pytest.raises(ValueError, match="score_direction"):
        heldout_permutation_deltas(
            "signal", frame, target, {}, donor_indices={}, predict=_predict, score=_rmse, score_direction="higher"
        )
