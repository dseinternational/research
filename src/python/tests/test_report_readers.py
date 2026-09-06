# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Read states preserve corruption, absence and valid empty/null distinctions."""

import json

import numpy as np
import pandas as pd
import pytest

from dse_research_utils.report.data import ReportData
from dse_research_utils.report.readers import nearest_row, read_csv, read_json


def test_csv_missing_empty_and_header_only_are_different(tmp_path):
    missing = read_csv(tmp_path / "absent.csv")
    assert missing.status == "missing" and missing.value is None
    assert missing.reason is missing.error_type is None
    empty = tmp_path / "empty.csv"
    empty.write_text(" \n\t", encoding="utf-8")
    invalid = read_csv(empty)
    assert (invalid.status, invalid.reason, invalid.error_type) == ("invalid", "empty_document", "EmptyDataError")
    header = tmp_path / "header.csv"
    header.write_text("age_months,median\n", encoding="utf-8")
    present = read_csv(header)
    assert present.status == "present"
    assert present.value.empty
    assert present.value.columns.tolist() == ["age_months", "median"]


def test_csv_parser_failure_is_not_a_schema_check(tmp_path):
    malformed = tmp_path / "bad.csv"
    malformed.write_text('name,value\n"unterminated,1', encoding="utf-8")
    result = read_csv(malformed)
    assert (result.status, result.reason, result.error_type) == ("invalid", "parse_error", "ParserError")
    short = tmp_path / "short.csv"
    short.write_text("name,value\nrow\n", encoding="utf-8")
    result = read_csv(short)
    assert result.status == "present"
    assert pd.isna(result.value.loc[0, "value"])


def test_csv_index_column_preserves_lrp_diagnostic_reader_contract(tmp_path):
    path = tmp_path / "diagnostics.csv"
    path.write_text("parameter,mean,r_hat,ess_bulk\nbeta,2.3,1.01,500\n", encoding="utf-8")
    result = read_csv(path, index_col=0)
    assert result.status == "present"
    pd.testing.assert_frame_equal(result.value, pd.read_csv(path, index_col=0), check_exact=True)
    assert result.value.index.name == "parameter"
    with pytest.raises(ValueError):
        read_csv(path, index_col="unknown")


@pytest.mark.parametrize("reader,suffix", [(read_csv, "csv"), (read_json, "json")])
def test_decode_and_directory_failures_are_invalid(reader, suffix, tmp_path):
    path = tmp_path / f"broken.{suffix}"
    path.write_bytes(b"\xff")
    result = reader(path)
    assert (result.status, result.reason, result.error_type) == ("invalid", "decode_error", "UnicodeDecodeError")
    result = reader(tmp_path)
    assert result.status == "invalid" and result.reason == "read_error"
    assert result.error_type in ("IsADirectoryError", "PermissionError")


@pytest.mark.parametrize("reader", [read_csv, read_json])
def test_permission_failure_is_not_missing(reader, monkeypatch, tmp_path):
    def denied(*args, **kwargs):
        raise PermissionError("Private path or diagnostic content")

    if reader is read_csv:
        monkeypatch.setattr(pd, "read_csv", denied)
    else:
        monkeypatch.setattr(type(tmp_path), "read_text", denied)
    result = reader(tmp_path / "data")
    assert (result.status, result.reason, result.error_type) == ("invalid", "read_error", "PermissionError")
    assert "Private" not in repr(result)


@pytest.mark.parametrize("contents,value", [("null", None), ("{}", {}), ("[]", []), ("false", False), ("3.5", 3.5)])
def test_json_null_scalars_and_empty_containers_are_present(tmp_path, contents, value):
    path = tmp_path / "value.json"
    path.write_text(contents, encoding="utf-8")
    result = read_json(path)
    assert result.status == "present" and result.value == value
    assert result.reason is result.error_type is None
    missing = read_json(tmp_path / "missing.json")
    assert missing.status == "missing" and missing.value is None


@pytest.mark.parametrize("contents,reason", [("", "empty_document"), (" \n\t", "empty_document"), ("{", "parse_error")])
def test_json_empty_and_malformed_are_invalid_not_null(tmp_path, contents, reason):
    path = tmp_path / "bad.json"
    path.write_text(contents, encoding="utf-8")
    result = read_json(path)
    assert (result.status, result.reason, result.error_type) == ("invalid", reason, "JSONDecodeError")


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_additive_json_reader_rejects_nonfinite_numbers_without_changing_legacy_loader(tmp_path, token):
    report = ReportData(lambda _id, _cfg: tmp_path)
    path = tmp_path / "value.json"
    path.write_text(f'{{"value": {token}}}', encoding="utf-8")
    fact = report.read_json("m", "value")
    assert (fact.status, fact.reason, fact.error_type) == ("invalid", "parse_error", "ValueError")
    assert not np.isfinite(report.load_json("m", "value")["value"])


def test_json_decoder_recursion_failure_is_an_invalid_parse(tmp_path, monkeypatch):
    path = tmp_path / "deep.json"
    path.write_text("[[[]]]", encoding="utf-8")

    def decoder_limit(contents, **kwargs):
        # Decoder nesting limits vary across Python builds. Simulate the
        # data-dependent failure without depending on a platform stack limit.
        assert contents == "[[[]]]"
        raise RecursionError("maximum nesting depth exceeded")

    monkeypatch.setattr(json, "loads", decoder_limit)
    result = read_json(path)
    assert (result.status, result.reason, result.error_type) == ("invalid", "parse_error", "RecursionError")


def test_json_memory_failure_is_not_reported_as_invalid_input(tmp_path, monkeypatch):
    path = tmp_path / "value.json"
    path.write_text("{}", encoding="utf-8")

    def exhausted(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(json, "loads", exhausted)
    with pytest.raises(MemoryError):
        read_json(path)


def test_report_methods_keep_resolver_defaults_and_legacy_exception_behavior(tmp_path):
    calls = []

    def resolve(model, config):
        calls.append((model, config))
        return tmp_path / f"{model}-{config}"

    report = ReportData(resolve, default_config="reporting")
    folder = tmp_path / "m-reporting"
    folder.mkdir()
    (folder / "table.csv").write_text("index,value\nrow,2\n", encoding="utf-8")
    (folder / "value.json").write_text("null", encoding="utf-8")
    assert report.read_summary("m", "table", index_col=0).value.index.tolist() == ["row"]
    assert report.read_json("m", "value").status == "present"
    assert report.load_json("m", "value") is None
    assert report.read_summary("m", "table", "dev").status == "missing"
    assert calls == [("m", "reporting"), ("m", "reporting"), ("m", "reporting"), ("m", "dev")]
    (folder / "table.csv").write_text("", encoding="utf-8")
    (folder / "value.json").write_text("{", encoding="utf-8")
    with pytest.raises(pd.errors.EmptyDataError):
        report.load_summary("m", "table")
    with pytest.raises(json.JSONDecodeError):
        report.load_json("m", "value")


def test_nearest_row_preserves_first_positional_tie_with_duplicate_index_labels():
    frame = pd.DataFrame(
        {"age": ["unknown", "24", "12", "24", "inf"], "value": [999, 40, 5, 400, 999]},
        index=["same"] * 5,
    )
    original = frame.copy(deep=True)
    row = nearest_row(frame, key="age", at=18)
    pd.testing.assert_series_equal(row, frame.iloc[1])
    pd.testing.assert_series_equal(nearest_row(frame, key="age", at=24), frame.iloc[1])
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("dtype", ["float32", "float64", "Int64", "Float64"])
def test_nearest_row_matches_existing_float_query_and_full_row(dtype):
    frame = pd.DataFrame({"age": pd.Series([12, 24, 36], dtype=dtype), "value": [5, 40, 120]})
    for at in (12.0, 18.0, 25.5, 80.0):
        expected = frame.iloc[(frame.age - at).abs().argmin()]
        pd.testing.assert_series_equal(nearest_row(frame, key="age", at=at), expected)


def test_nearest_row_bound_is_optional_inclusive_and_does_not_interpolate():
    frame = pd.DataFrame({"age": [12, 24, 36], "value": [5, 40, 120]})
    assert nearest_row(frame, key="age", at=50)["value"] == 120
    assert nearest_row(frame, key="age", at=50, max_distance=13) is None
    assert nearest_row(frame, key="age", at=50, max_distance=14)["age"] == 36
    assert nearest_row(frame, key="age", at=24, max_distance=0)["value"] == 40


def test_unsigned_keys_do_not_wrap_below_query():
    frame = pd.DataFrame({"key": np.array([0, 10], dtype=np.uint64), "label": ["near", "far"]})
    assert nearest_row(frame, key="key", at=2)["label"] == "near"


def test_signed_absolute_distance_does_not_overflow():
    frame = pd.DataFrame({"key": [np.iinfo(np.int64).min, np.iinfo(np.int64).max], "label": ["far", "near"]})
    assert nearest_row(frame, key="key", at=0)["label"] == "near"


@pytest.mark.parametrize("at", [2**63, float(2**63)])
def test_large_nullable_unsigned_keys_remain_exact(at):
    frame = pd.DataFrame(
        {"key": pd.Series([2**63 + 2, None, 2**63], dtype="UInt64"), "label": ["far", "missing", "near"]}
    )
    assert nearest_row(frame, key="key", at=at)["label"] == "near"


def test_float_distance_overflow_uses_nearest_represented_value():
    frame = pd.DataFrame({"key": [-1e308, -9e307], "label": ["far", "near"]})
    assert nearest_row(frame, key="key", at=1e308)["label"] == "near"
    assert nearest_row(frame, key="key", at=1e308, max_distance=np.finfo(float).max) is None


def test_integral_query_is_not_rounded_into_a_false_float_key_tie():
    # Both keys are representable floats. Rounding the exact integer query to
    # float changes it to the midpoint and would select the first, farther row.
    frame = pd.DataFrame({"key": np.array([2**54, 2**54 + 8], dtype=float), "label": ["far", "near"]})
    assert float(2**54 + 5) == 2**54 + 4
    assert nearest_row(frame, key="key", at=2**54 + 5)["label"] == "near"


@pytest.mark.parametrize("at", [np.nan, np.inf, -np.inf])
def test_nonfinite_queries_have_no_nearest_row(at):
    assert nearest_row(pd.DataFrame({"key": [1]}), key="key", at=at) is None


def test_absent_empty_or_unusable_keys_have_no_nearest_row():
    for frame in (None, pd.DataFrame(), pd.DataFrame({"other": [1]}), pd.DataFrame({"key": [None, "bad", np.inf]})):
        assert nearest_row(frame, key="key", at=1) is None


@pytest.mark.parametrize("bound", [-1, np.nan, np.inf])
def test_invalid_distance_bounds_raise_even_without_data(bound):
    with pytest.raises(ValueError, match="max_distance"):
        nearest_row(None, key="key", at=1, max_distance=bound)


@pytest.mark.parametrize("bound", [True, np.bool_(True), "1"])
def test_distance_bounds_require_a_real_number(bound):
    with pytest.raises(TypeError, match="max_distance"):
        nearest_row(None, key="key", at=1, max_distance=bound)


def test_ambiguous_columns_and_non_real_queries_are_not_silently_selected():
    with pytest.raises(ValueError, match="one column"):
        nearest_row(pd.DataFrame([[1, 2]], columns=["key", "key"]), key="key", at=1)
    with pytest.raises(TypeError, match="at must"):
        nearest_row(pd.DataFrame({"key": [1]}), key="key", at="1")
    with pytest.raises(TypeError, match="real numbers"):
        nearest_row(pd.DataFrame({"key": [1 + 2j]}), key="key", at=1)


def test_report_value_at_uses_exact_integer_distance(tmp_path):
    report = ReportData(lambda _id, _cfg: tmp_path)
    pd.DataFrame({"age_months": [np.iinfo(np.int64).min, np.iinfo(np.int64).max], "value": [1, 2]}).to_csv(
        tmp_path / "summary.csv", index=False
    )
    assert report.value_at("m", "summary", "value", at=0) == 2
