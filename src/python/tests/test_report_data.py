# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the shared report data-access helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dse_research_utils.report.data import ReportData, num, show_or_pending


def _make_report(tmp_path: Path, *, default_config: str | None = None) -> ReportData:
    """A ReportData over an LRP-style ``<root>/<id>-<config>`` layout."""

    def resolve(model_id: str, config: str | None) -> Path:
        return tmp_path / f"{model_id}-{config}"

    return ReportData(resolve, default_config=default_config)


def test_num_formats_or_em_dash():
    assert num(3.6) == "4"
    assert num(0.25, "{:.2f}") == "0.25"
    assert num(None) == "—"


def test_show_or_pending_passes_through_present_value():
    df = pd.DataFrame({"a": [1]})
    assert show_or_pending(df, "the table") is df


def test_show_or_pending_none_renders_placeholder():
    md = pytest.importorskip("IPython.display")
    out = show_or_pending(None, "the ITT table")
    assert isinstance(out, md.Markdown)
    assert "Pending fit" in out.data
    assert "the ITT table" in out.data


def test_model_dir_uses_default_config(tmp_path):
    report = _make_report(tmp_path, default_config="reporting")
    assert report.model_dir("vg01") == tmp_path / "vg01-reporting"
    assert report.model_dir("vg01", "dev") == tmp_path / "vg01-dev"


def test_load_summary_reads_csv_or_none(tmp_path):
    report = _make_report(tmp_path, default_config="reporting")
    d = tmp_path / "vg01-reporting"
    d.mkdir()
    pd.DataFrame({"age_months": [12, 24], "Ey_median": [5, 40]}).to_csv(d / "summary.csv", index=False)

    got = report.load_summary("vg01", "summary")
    assert got is not None
    assert list(got["Ey_median"]) == [5, 40]
    # Missing file degrades to None rather than raising.
    assert report.load_summary("vg01", "absent") is None
    assert report.load_summary("vg99", "summary") is None


def test_load_json_reads_or_none(tmp_path):
    report = _make_report(tmp_path, default_config="reporting")
    d = tmp_path / "vg01-reporting"
    d.mkdir()
    (d / "diagnostics_summary.json").write_text(json.dumps({"passed": True}))
    assert report.load_json("vg01", "diagnostics_summary") == {"passed": True}
    assert report.load_json("vg01", "absent") is None


def test_fig_returns_path_string(tmp_path):
    report = _make_report(tmp_path, default_config="reporting")
    assert report.fig("vg01", "trace.png") == str(tmp_path / "vg01-reporting" / "trace.png")


def test_summary_value_first_row_or_none(tmp_path):
    report = _make_report(tmp_path, default_config="reporting")
    d = tmp_path / "vg01-reporting"
    d.mkdir()
    pd.DataFrame({"P_psi_gt_1": [0.97, 0.5]}).to_csv(d / "posterior_summary_psi.csv", index=False)
    assert report.summary_value("vg01", "posterior_summary_psi", "P_psi_gt_1") == pytest.approx(0.97)
    assert report.summary_value("vg01", "posterior_summary_psi", "absent_col") is None
    assert report.summary_value("vg01", "absent_summary", "P_psi_gt_1") is None


def test_value_at_nearest_key(tmp_path):
    report = _make_report(tmp_path, default_config="reporting")
    d = tmp_path / "vg01-reporting"
    d.mkdir()
    pd.DataFrame({"age_months": [12, 24, 36], "Ey_median": [5, 40, 120]}).to_csv(
        d / "summary.csv", index=False
    )
    # 25 is nearest to 24.
    assert report.value_at("vg01", "summary", "Ey_median", at=25) == 40
    # Missing key column degrades to None.
    assert report.value_at("vg01", "summary", "Ey_median", at=25, key="not_a_col") is None


def test_registry_style_resolver_ignores_config(tmp_path):
    # A VG-style layout where the directory is fixed per model (config ignored).
    dirs = {"VG15": tmp_path / "VG15-age-sign-td"}
    report = ReportData(lambda mid, cfg: dirs[mid])
    dirs["VG15"].mkdir()
    (dirs["VG15"] / "s.csv").write_text("x\n1\n")
    assert report.load_summary("VG15", "s") is not None
