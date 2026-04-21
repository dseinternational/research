# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import time

from dse_research_utils.console.sections import (
    banner,
    section_header,
    subsection,
    timed_section,
)


def test_banner_includes_title(captured_console):
    banner("Test Run")
    output = captured_console.export_text()
    assert "Test Run" in output


def test_banner_includes_subtitle(captured_console):
    banner("Run", subtitle="extra info")
    output = captured_console.export_text()
    assert "Run" in output
    assert "extra info" in output


def test_section_header_includes_title(captured_console):
    section_header("Stage One")
    output = captured_console.export_text()
    assert "Stage One" in output


def test_subsection_includes_title(captured_console):
    subsection("Sub A")
    output = captured_console.export_text()
    assert "Sub A" in output


def test_timed_section_records_timing(captured_console):
    timings: dict[str, float] = {}
    with timed_section("work", timings=timings):
        time.sleep(0.01)
    assert "work" in timings
    assert timings["work"] >= 0.01


def test_timed_section_uses_custom_key(captured_console):
    timings: dict[str, float] = {}
    with timed_section("Stage Title", timings=timings, key="stage_id"):
        pass
    assert "stage_id" in timings
    assert "Stage Title" not in timings


def test_timed_section_prints_completion(captured_console):
    with timed_section("finish"):
        pass
    output = captured_console.export_text()
    assert "finish" in output
    assert "✓" in output
