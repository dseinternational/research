# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pytest

from dse_research_utils.environment.info import (
    get_environment_info,
    get_execution_context,
    report_environment_info,
)


def test_get_execution_context_in_tests_is_script() -> None:
    # pytest runs in a normal Python process without IPython.
    assert get_execution_context() == "script"


def test_get_environment_info_returns_dict_by_default() -> None:
    info = get_environment_info()
    assert isinstance(info, dict)
    for key in (
        "platform",
        "platform_version",
        "architecture",
        "processor",
        "cpu_count_logical",
        "cpu_count_physical",
        "memory_total_gb",
        "python",
        "execution_context",
    ):
        assert key in info


def test_get_environment_info_as_dataframe() -> None:
    df = get_environment_info(as_dataframe=True)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["name", "value"]
    assert len(df) > 0


def test_report_environment_info_prints(capsys: pytest.CaptureFixture[str]) -> None:
    report_environment_info()
    captured = capsys.readouterr()
    assert "Environment" in captured.out
