# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote, unquote, urlsplit

import pytest

from dse_research_utils.storage.azure import (
    _iter_upload_candidates,
    parse_blob_container_url,
    upload_directory_to_blob_storage,
)


@pytest.fixture
def fake_blob_uploads(monkeypatch):
    uploads = []

    def upload_blob(name, data, **kwargs):
        uploads.append((name, data.read()))

    client = SimpleNamespace(get_container_client=lambda name: SimpleNamespace(upload_blob=upload_blob))
    monkeypatch.setattr("azure.storage.blob.BlobServiceClient", lambda *args, **kwargs: client)
    return uploads


def test_uploaded_urls_encode_blob_names(tmp_path, fake_blob_uploads):
    (tmp_path / "figure #1 50%.svg").write_text("svg", encoding="utf-8")
    result = upload_directory_to_blob_storage(
        tmp_path,
        "model #1",
        project="test project",
        run_id="run",
        container_url="https://acct.blob.core.windows.net/reports",
        credential=object(),
    )
    parsed = urlsplit(result.urls[0])
    assert parsed.fragment == ""
    assert " " not in result.urls[0]
    assert unquote(parsed.path) == "/reports/" + fake_blob_uploads[0][0]
    assert "%23" in result.prefix_url


def test_upload_preserves_raw_relative_paths(tmp_path, fake_blob_uploads):
    names = ["psi (dev).png", "a+b.csv", "café #1.csv", "trace.nc"]
    for name in names:
        (tmp_path / name).write_text(name, encoding="utf-8")
    result = upload_directory_to_blob_storage(
        tmp_path,
        "model #1",
        project="test project",
        run_id="run",
        skip=lambda path: path == "a+b.csv",
        container_url="https://acct.blob.core.windows.net/reports",
        credential=object(),
    )
    assert result.relative_paths == ["café #1.csv", "psi (dev).png"]
    assert result.urls == [result.prefix_url + quote(path, safe="/") for path in result.relative_paths]
    assert result.uploaded_files == 2
    assert result.skipped_files == 2


def test_report_url_points_to_root_index(tmp_path, fake_blob_uploads):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "index.html").write_text("nested", encoding="utf-8")
    (tmp_path / "index.html").write_text("report", encoding="utf-8")
    result = upload_directory_to_blob_storage(
        tmp_path,
        "model",
        project="test",
        run_id="run",
        container_url="https://acct.blob.core.windows.net/reports",
        credential=object(),
    )
    assert result.report_url == result.prefix_url + "index.html"


def test_rejected_sas_is_not_copied_into_error_message():
    with pytest.raises(RuntimeError) as exc:
        parse_blob_container_url("https://acct.blob.core.windows.net/reports?sig=secret")
    assert "secret" not in str(exc.value)


def test_parse_blob_container_url() -> None:
    target = parse_blob_container_url("https://acct.blob.core.windows.net/reports")

    assert target.account_url == "https://acct.blob.core.windows.net"
    assert target.container_name == "reports"
    assert target.base_url == "https://acct.blob.core.windows.net/reports"


@pytest.mark.parametrize(
    "url",
    [
        "http://acct.blob.core.windows.net/reports",
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
