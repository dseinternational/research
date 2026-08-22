# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import io
import sys

import pytest
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dse_research_utils.console.console import (
    get_console,
    print_panel,
    print_rule,
    print_table,
    set_console,
)


def _text_stream(encoding: str) -> io.TextIOWrapper:
    """A writable text stream with a real encoding, standing in for ``sys.stdout``."""
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding, newline="")


@pytest.fixture
def fresh_console():
    """Clear the singleton so the next ``get_console`` call rebuilds it, and again after."""
    set_console(None)
    try:
        yield
    finally:
        set_console(None)


def test_get_console_returns_singleton():
    a = get_console()
    b = get_console()
    assert a is b


def test_set_console_replaces_singleton(captured_console):
    assert get_console() is captured_console


def test_set_console_none_resets(captured_console):
    first = get_console()
    set_console(None)
    second = get_console()
    assert second is not first


def test_print_table_routes_through_shared_console(captured_console):
    table = Table(title="T")
    table.add_column("col")
    table.add_row("row")
    print_table(table)
    output = captured_console.export_text()
    assert "T" in output
    assert "col" in output
    assert "row" in output


def test_print_panel_routes_through_shared_console(captured_console):
    panel = Panel("hello")
    print_panel(panel)
    output = captured_console.export_text()
    assert "hello" in output


def test_print_rule_includes_label(captured_console):
    print_rule("divider")
    output = captured_console.export_text()
    assert "divider" in output


def test_get_console_relaxes_legacy_stdout(monkeypatch, fresh_console):
    """A cp1252 stdout — the Windows default — is switched to errors='replace'."""
    stream = _text_stream("cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    get_console()
    assert stream.errors == "replace"


def test_get_console_leaves_utf8_stdout_strict(monkeypatch, fresh_console):
    stream = _text_stream("utf-8")
    monkeypatch.setattr(sys, "stdout", stream)
    get_console()
    assert stream.errors == "strict"


def test_get_console_tolerates_stdout_without_reconfigure(monkeypatch, fresh_console):
    """StringIO has no encoding and no reconfigure; creating the console must still work."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert get_console() is not None


def test_console_prints_unencodable_text_to_legacy_stdout(monkeypatch, fresh_console):
    """Regression (#91): rich raised rather than degrading a character cp1252 lacks."""
    stream = _text_stream("cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    get_console().print(Text("check \u2713 dash \u2014 ellipsis \u2026"))
    stream.flush()
    written = stream.buffer.getvalue().decode("cp1252")
    # cp1252 carries the em-dash and ellipsis; only the checkmark degrades.
    assert "check ? dash \u2014 ellipsis \u2026" in written
