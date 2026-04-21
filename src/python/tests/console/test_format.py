# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from pathlib import Path

import pandas as pd

from dse_research_utils.console.format import EM_DASH, format_duration, format_value


class TestFormatValue:
    def test_none_renders_em_dash(self):
        assert format_value(None) == EM_DASH

    def test_nan_renders_em_dash(self):
        assert format_value(float("nan")) == EM_DASH

    def test_pandas_na_renders_em_dash(self):
        assert format_value(pd.NA) == EM_DASH

    def test_pandas_nat_renders_em_dash(self):
        assert format_value(pd.NaT) == EM_DASH

    def test_bool_true_renders_yes(self):
        assert format_value(True) == "yes"

    def test_bool_false_renders_no(self):
        assert format_value(False) == "no"

    def test_small_float_uses_scientific(self):
        result = format_value(1.5e-5, precision=3)
        assert "e" in result

    def test_large_float_uses_scientific(self):
        result = format_value(1.5e6, precision=3)
        assert "e" in result

    def test_moderate_float_uses_g_format(self):
        assert format_value(1.2345, precision=3) == "1.23"

    def test_int_uses_thousands_separator(self):
        assert format_value(1_234_567) == "1,234,567"

    def test_zero_float_is_plain(self):
        assert format_value(0.0, precision=3) == "0"

    def test_path_renders_as_string(self):
        p = Path("/tmp/foo")
        assert format_value(p) == str(p)

    def test_list_joins_with_comma(self):
        assert format_value([1, 2, 3]) == "1, 2, 3"

    def test_nested_list_formats_recursively(self):
        assert format_value([1.5e-5, 42], precision=2) == "1.50e-05, 42"

    def test_string_passes_through(self):
        assert format_value("hello") == "hello"


class TestFormatDuration:
    def test_milliseconds(self):
        assert format_duration(0.25) == "250 ms"

    def test_seconds(self):
        assert format_duration(12.345) == "12.35 s"

    def test_minutes(self):
        result = format_duration(125)
        assert result.startswith("2m")

    def test_hours(self):
        result = format_duration(3661)
        assert result.startswith("1h")
        assert "01m" in result

    def test_nan_safe(self):
        assert not math.isnan(0.5)
