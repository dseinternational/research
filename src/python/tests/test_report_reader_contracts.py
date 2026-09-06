# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Executable report adapters keep read facts apart from local visibility rules."""

import pandas as pd

from dse_research_utils.report.readers import nearest_row, read_csv, read_json


def _local_report_view(result, *, released):
    """An illustrative report-owned policy, deliberately outside the library."""
    if result.status == "missing":
        return "missing", "The output file is missing."
    if result.status == "invalid":
        return "invalid", "The output file could not be read."
    table = result.value
    if not released:
        if result.path.name == "diagnostics.csv":
            table = table.loc[:, [name for name in table if name in {"r_hat", "ess_bulk"}]]
        elif result.path.name != "ppc_summary.csv":
            return "withheld", "Results are withheld by the report's publication decision."
    return "present", table.to_string(index=False) if not table.empty else "The table contains no rows."


def test_report_fixture_distinguishes_missing_invalid_and_withheld_outputs(tmp_path):
    (tmp_path / "effect.csv").write_text("estimate\n2.5\n", encoding="utf-8")
    (tmp_path / "broken.csv").write_text('estimate\n"broken', encoding="utf-8")
    (tmp_path / "empty.csv").write_text("estimate\n", encoding="utf-8")
    cases = [
        ("missing.csv", True, "missing"),
        ("broken.csv", True, "invalid"),
        ("effect.csv", False, "withheld"),
        ("effect.csv", True, "present"),
        ("empty.csv", True, "present"),
    ]
    views = [_local_report_view(read_csv(tmp_path / name), released=released) for name, released, _expected in cases]
    assert [state for state, _text in views] == [expected for _name, _released, expected in cases]
    assert len({text for _state, text in views[:3]}) == 3
    assert all("pending fit" not in text.lower() for _state, text in views)
    assert views[-1][1] == "The table contains no rows."


def test_late_withholding_changes_the_local_view_without_changing_read_facts(tmp_path):
    path = tmp_path / "effect.csv"
    path.write_text("estimate\n2.5\n", encoding="utf-8")
    fact = read_csv(path)
    assert _local_report_view(fact, released=True)[0] == "present"
    assert _local_report_view(fact, released=False)[0] == "withheld"
    assert fact.status == "present"
    assert fact.value.iloc[0, 0] == 2.5


def test_failed_gate_keeps_only_report_selected_diagnostics_visible(tmp_path):
    (tmp_path / "diagnostics.csv").write_text("parameter,mean,r_hat,ess_bulk\nbeta,9.0,1.02,500\n", encoding="utf-8")
    (tmp_path / "ppc_summary.csv").write_text("coverage\n0.8\n", encoding="utf-8")
    (tmp_path / "unknown_future_result.csv").write_text("estimate\n7.0\n", encoding="utf-8")
    diagnostic = read_csv(tmp_path / "diagnostics.csv", index_col=0)
    state, displayed = _local_report_view(diagnostic, released=False)
    assert state == "present" and "r_hat" in displayed and "ess_bulk" in displayed
    assert "mean" not in displayed and "9.0" not in displayed
    assert diagnostic.value.columns.tolist() == ["mean", "r_hat", "ess_bulk"]
    assert _local_report_view(read_csv(tmp_path / "ppc_summary.csv"), released=False)[0] == "present"
    unknown = read_csv(tmp_path / "unknown_future_result.csv")
    assert unknown.status == "present"
    assert _local_report_view(unknown, released=False)[0] == "withheld"


def test_parsed_json_does_not_choose_a_gate_schema_or_diagnostic_verdict(tmp_path):
    path = tmp_path / "diagnostics_summary.json"
    path.write_text('{"passed": false}', encoding="utf-8")
    fact = read_json(path)
    assert fact.status == "present" and fact.value == {"passed": False}
    path.write_text("null", encoding="utf-8")
    fact = read_json(path)
    assert fact.status == "present" and fact.value is None
    # The report, not read_json, requires an object before evaluating a gate.
    local_schema_valid = isinstance(fact.value, dict)
    assert not local_schema_valid


def test_comparison_row_adapter_preserves_vg_renaming_and_full_row(tmp_path):
    path = tmp_path / "comparison.csv"
    path.write_text("age_months,median,ci_lo,ci_hi\n24,4,2,6\n12,1,0,2\n36,8,5,11\n", encoding="utf-8")
    table = read_csv(path).value
    renamed = table.rename(columns={"median": "d_median", "ci_lo": "d_ci_lo", "ci_hi": "d_ci_hi"})
    row = nearest_row(renamed, key="age_months", at=18)
    expected = pd.Series({"age_months": 24, "d_median": 4, "d_ci_lo": 2, "d_ci_hi": 6}, name=0)
    pd.testing.assert_series_equal(row, expected)
