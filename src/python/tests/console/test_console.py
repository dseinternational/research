# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from rich.panel import Panel
from rich.table import Table

from dse_research_utils.console.console import (
    get_console,
    print_panel,
    print_rule,
    print_table,
    set_console,
)


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
