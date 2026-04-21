# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pytest

from dse_research_utils.console.tables import (
    dataframe_table,
    key_value_table,
    metrics_table,
    params_table,
    print_key_value_table,
)


def test_key_value_table_from_mapping(captured_console):
    table = key_value_table({"alpha": 1, "beta": 2.5}, title="Config")
    captured_console.print(table)
    output = captured_console.export_text()
    assert "Config" in output
    assert "alpha" in output
    assert "beta" in output
    assert "2.5" in output


def test_key_value_table_from_iterable_of_tuples(captured_console):
    pairs = [("first", "a"), ("second", "b")]
    table = key_value_table(pairs)
    captured_console.print(table)
    output = captured_console.export_text()
    assert "first" in output
    assert "second" in output


def test_print_key_value_table_emits_via_shared_console(captured_console):
    print_key_value_table({"x": 1})
    output = captured_console.export_text()
    assert "x" in output


def test_metrics_table_with_explicit_columns(captured_console):
    rows = [
        {"metric": "MAE", "mean": 0.5, "std": 0.05},
        {"metric": "RMSE", "mean": 0.7, "std": 0.07},
    ]
    table = metrics_table(rows, title="CV", columns=["metric", "mean", "std"], precision=3)
    captured_console.print(table)
    output = captured_console.export_text()
    assert "MAE" in output
    assert "RMSE" in output
    assert "CV" in output


def test_metrics_table_empty_rows_returns_table():
    table = metrics_table([], title="Empty")
    assert table.row_count == 0


def test_params_table_renders_keys_and_values(captured_console):
    table = params_table({"lr": 0.01, "depth": 5})
    captured_console.print(table)
    output = captured_console.export_text()
    assert "lr" in output
    assert "depth" in output


def test_dataframe_table_renders_cells(captured_console):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    table = dataframe_table(df, title="Frame", show_index=False)
    captured_console.print(table)
    output = captured_console.export_text()
    assert "Frame" in output
    assert "a" in output
    assert "b" in output


def test_dataframe_table_head_truncation_adds_ellipsis():
    df = pd.DataFrame({"value": range(100)})
    table = dataframe_table(df, max_rows=5, truncation="head", show_index=False)
    # 5 data rows + 1 ellipsis filler row
    assert table.row_count == 6
    assert "95 more rows" in str(table.caption)


def test_dataframe_table_head_tail_truncation_adds_caption():
    df = pd.DataFrame({"value": range(100)})
    table = dataframe_table(df, max_rows=10, truncation="head_tail", show_index=False)
    assert table.row_count == 10
    assert "showing 10 of 100 rows" in str(table.caption)


def test_dataframe_table_rank_column(captured_console):
    df = pd.DataFrame({"feature": ["x", "y", "z"], "importance": [0.5, 0.3, 0.1]})
    table = dataframe_table(df, rank_column="#", show_index=False)
    captured_console.print(table)
    output = captured_console.export_text()
    assert "#" in output
    # Rank column means 1-indexed row numbers should appear
    assert "1" in output
    assert "3" in output


def test_dataframe_table_invalid_truncation_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="truncation must be"):
        dataframe_table(df, truncation="middle")


def test_dataframe_table_empty_frame_returns_empty_table():
    df = pd.DataFrame()
    table = dataframe_table(df, title="Empty")
    assert table.row_count == 0
