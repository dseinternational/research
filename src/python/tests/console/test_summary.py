# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

from dse_research_utils.console.summary import (
    artefact_summary,
    pipeline_summary,
    run_header_panel,
    run_summary_panel,
)


def test_run_header_panel_includes_title_and_details(captured_console):
    panel = run_header_panel("Fit XYZ", {"target": "score", "predictors": 12})
    captured_console.print(panel)
    output = captured_console.export_text()
    assert "Fit XYZ" in output
    assert "target" in output
    assert "predictors" in output
    assert "12" in output


def test_run_header_panel_without_details(captured_console):
    panel = run_header_panel("Just a title", {})
    captured_console.print(panel)
    output = captured_console.export_text()
    assert "Just a title" in output


def test_run_summary_panel_with_artefacts_dir(captured_console, tmp_path: Path):
    panel = run_summary_panel("Complete", artefacts_dir=tmp_path)
    captured_console.print(panel)
    output = captured_console.export_text()
    assert "Complete" in output
    assert str(tmp_path.resolve()) in output


def test_run_summary_panel_accepts_string_path(captured_console, tmp_path: Path):
    panel = run_summary_panel(artefacts_dir=str(tmp_path))
    captured_console.print(panel)
    output = captured_console.export_text()
    assert str(tmp_path.resolve()) in output


def test_run_summary_panel_extra_fields(captured_console):
    panel = run_summary_panel("Done", extra={"artifacts": 7, "duration": "2m 15s"})
    captured_console.print(panel)
    output = captured_console.export_text()
    assert "artifacts" in output
    assert "duration" in output


def test_pipeline_summary_renders_stages_and_total(captured_console):
    timings = {"prepare": 1.0, "fit": 3.0, "evaluate": 1.0}
    table = pipeline_summary(timings)
    captured_console.print(table)
    output = captured_console.export_text()
    assert "prepare" in output
    assert "fit" in output
    assert "evaluate" in output
    assert "Total" in output


def test_pipeline_summary_empty_timings(captured_console):
    table = pipeline_summary({})
    captured_console.print(table)
    output = captured_console.export_text()
    assert "Stage" in output


def test_artefact_summary_counts_files_by_extension(captured_console, tmp_path: Path):
    (tmp_path / "a.png").write_text("x")
    (tmp_path / "b.png").write_text("x")
    (tmp_path / "c.csv").write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.png").write_text("x")

    table = artefact_summary(tmp_path)
    captured_console.print(table)
    output = captured_console.export_text()
    assert ".png" in output
    assert ".csv" in output
    assert "3" in output
    assert "1" in output


def test_artefact_summary_missing_dir(captured_console, tmp_path: Path):
    table = artefact_summary(tmp_path / "does_not_exist")
    captured_console.print(table)
    output = captured_console.export_text()
    assert "Artefacts" in output
