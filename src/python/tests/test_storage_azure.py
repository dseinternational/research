# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pytest

from dse_research_utils.storage.azure import (
    _iter_upload_candidates,
    parse_blob_container_url,
)


def test_parse_blob_container_url() -> None:
    target = parse_blob_container_url("https://acct.blob.core.windows.net/reports")

    assert target.account_url == "https://acct.blob.core.windows.net"
    assert target.container_name == "reports"
    assert target.base_url == "https://acct.blob.core.windows.net/reports"


@pytest.mark.parametrize(
    "url",
    [
        "https://acct.blob.core.windows.net/reports?sig=secret",
        "https://acct.blob.core.windows.net/reports#fragment",
        "https://acct.blob.core.windows.net/reports/extra",
        "not-a-url",
    ],
)
def test_parse_blob_container_url_rejects_non_plain_container_urls(url: str) -> None:
    with pytest.raises(RuntimeError):
        parse_blob_container_url(url)


def test_iter_upload_candidates_returns_sorted_posix_paths(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")

    candidates = _iter_upload_candidates(tmp_path)

    assert candidates == [
        (tmp_path / "a.txt", "a.txt"),
        (nested / "b.txt", "nested/b.txt"),
    ]
