# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pytest

from dse_research_utils.metadata.packages import (
    get_package_version,
    get_package_versions,
    report_package_versions,
)


def test_get_package_version_known_package() -> None:
    version = get_package_version("numpy")
    assert version != "Not found"
    assert isinstance(version, str)


def test_get_package_version_missing_package() -> None:
    assert get_package_version("this-package-does-not-exist-xyz") == "Not found"


def test_get_package_versions_returns_dict_by_default() -> None:
    result = get_package_versions(["numpy", "this-does-not-exist-xyz"])
    assert isinstance(result, dict)
    assert set(result.keys()) == {"numpy", "this-does-not-exist-xyz"}
    assert result["this-does-not-exist-xyz"] == "Not found"


def test_get_package_versions_as_dataframe() -> None:
    result = get_package_versions(["numpy"], as_dataframe=True)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["version"]
    assert "numpy" in result.index


def test_report_package_versions_prints(capsys: pytest.CaptureFixture[str]) -> None:
    report_package_versions(["numpy"])
    captured = capsys.readouterr()
    assert "Package Versions" in captured.out
    assert "numpy" in captured.out
