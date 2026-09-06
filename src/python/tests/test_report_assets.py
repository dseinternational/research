# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publication checks retain browser filenames and expose incomplete bundles."""

from contextlib import nullcontext
from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from dse_research_utils.report.assets import (
    AssetFailure,
    check_uploaded_assets,
    inspect_local_assets,
    verify_published_assets,
)


def _page(tmp_path, html, files=()):
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")
    page = tmp_path / "index.html"
    page.write_text(html, encoding="utf-8")
    return page


def test_scan_required_resources_preserves_missing_external_and_navigation(tmp_path):
    page = _page(
        tmp_path,
        """<LINK rel='stylesheet' href='assets/style.css?version=2'>
        <script src="assets/app.js#main"></script>
        <img src='missing.png'><img src='https://example.org/external.png'>
        <img src='data:image/png;base64,AAAA'><a href='download.csv'>download</a>
        <a href='absent.html'>next page</a><a href='#section'>section</a>""",
        ["assets/style.css", "assets/app.js", "download.csv"],
    )
    result = inspect_local_assets(page)
    assert result.required_paths == ("assets/app.js", "assets/style.css", "index.html", "missing.png")
    assert [ref.status for ref in result.references] == [
        "present",
        "present",
        "missing",
        "external",
        "external",
        "present",
        "missing",
        "external",
    ]
    assert [ref.relative_path for ref in result.references if ref.kind == "navigation"] == [
        "download.csv",
        "absent.html",
        None,
    ]
    assert check_uploaded_assets(result, ["index.html", "assets\\style.css"]) == (
        AssetFailure("missing.png", "missing"),
        AssetFailure("assets/app.js", "not_uploaded"),
    )
    all_links = inspect_local_assets(page, include_navigation=True)
    assert "download.csv" in all_links.required_paths
    assert AssetFailure("absent.html", "missing") in check_uploaded_assets(all_links, all_links.required_paths)


def test_raw_upload_paths_and_encoded_html_match_current_uploader_contract(tmp_path):
    # Consumer fixture adapted from VG: literal percent filenames must be URL
    # encoded in HTML, while BlobUploadResult.relative_paths stays raw.
    page = _page(
        tmp_path,
        """<a href='figures/psi%20%28dev%29.png'>figure</a>
        <a href='tables/a+b.csv'>table</a><a href='figures/caf%C3%A9.png'>figure</a>
        <a href='tables/50%2520.csv'>table</a><a href='assets/index.html'>appendix</a>""",
        ["figures/psi (dev).png", "tables/a+b.csv", "figures/café.png", "tables/50%20.csv", "assets/index.html"],
    )
    inspection = inspect_local_assets(page, include_navigation=True)
    assert check_uploaded_assets(inspection, inspection.required_paths) == ()
    assert check_uploaded_assets(inspection, [p for p in inspection.required_paths if p != "tables/a+b.csv"]) == (
        AssetFailure("tables/a+b.csv", "not_uploaded"),
    )
    assert check_uploaded_assets(inspection, [p for p in inspection.required_paths if p != "index.html"]) == (
        AssetFailure("index.html", "not_uploaded"),
    )
    requested = []

    def fetch(url, timeout):
        requested.append((url, timeout))
        return 200

    assert (
        verify_published_assets("https://example.org/model%20%28dev%29/", inspection.required_paths, fetch_status=fetch)
        == ()
    )
    assert requested == [
        ("https://example.org/model%20%28dev%29/" + name, 30.0)
        for name in (
            "assets/index.html",
            "figures/caf%C3%A9.png",
            "figures/psi%20%28dev%29.png",
            "index.html",
            "tables/50%2520.csv",
            "tables/a%2Bb.csv",
        )
    ]


def test_all_supported_direct_resource_attributes_are_required(tmp_path):
    html = """<video src='film.mp4' poster='poster.png'></video><audio src='audio.mp3'>
    <source src='fallback.mp4'><track src='captions.vtt'><iframe src='frame.html'></iframe>
    <embed src='embedded.pdf'><object data='object.pdf'><input type='image' src='button.png'>
    <svg><image href='figure.svg'/><use xlink:href='symbols.svg#shape'/></svg>
    <link rel='preload' href='font.woff'><link rel='icon' href='favicon.ico'>"""
    result = inspect_local_assets(_page(tmp_path, html))
    assert len(result.references) == 13
    assert all(ref.required and ref.status == "missing" for ref in result.references)
    assert "symbols.svg" in result.required_paths


def test_missing_and_unsupported_cannot_be_hidden_by_claimed_uploads(tmp_path):
    page = _page(tmp_path, "<img src='missing.png'><img srcset='small.png 1x, large.png 2x'>")
    inspection = inspect_local_assets(page)
    failures = check_uploaded_assets(inspection, ["index.html", "missing.png", "small.png", "large.png"])
    assert failures == (
        AssetFailure("missing.png", "missing"),
        AssetFailure("small.png 1x, large.png 2x", "unsupported_srcset"),
    )


def test_directory_base_cannot_hide_a_changed_query_only_url(tmp_path):
    page = _page(tmp_path, "<base href='./'><img src='?version=1'>")
    inspection = inspect_local_assets(page)
    assert check_uploaded_assets(inspection, inspection.required_paths) == (AssetFailure("./", "unsupported_base"),)


@pytest.mark.parametrize(
    "target", ["dir//image.png", "dir%2Fimage.png", "image.png/", "image.png/.", "image.png/child/.."]
)
def test_distinct_browser_paths_cannot_be_normalized_to_an_existing_uploaded_file(tmp_path, target):
    page = _page(tmp_path, f"<img src='{target}'>", ["dir/image.png", "image.png"])
    inspection = inspect_local_assets(page)
    assert inspection.references[0].status == "invalid"
    assert check_uploaded_assets(inspection, ["index.html", "dir/image.png", "image.png"])


def test_css_and_script_text_are_not_claimed_as_scanned_dependencies(tmp_path):
    page = _page(
        tmp_path,
        """<!-- <img src='comment.png'> -->
    <style>body { background: url(css-only.png); }</style>
    <script>const html = "<img src='dynamic.png'>";</script>""",
    )
    assert inspect_local_assets(page).references == ()


def test_missing_entry_and_invalid_utf8_raise_instead_of_empty_success(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_local_assets(tmp_path / "absent.html")
    page = tmp_path / "index.html"
    page.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        inspect_local_assets(page)


@pytest.mark.parametrize("path", ["../outside.png", "/root.png", "//host/file", "C:\\file.png", "", "a\x00b"])
def test_invalid_verification_paths_raise_before_any_request(path):
    requested = []
    with pytest.raises(ValueError):
        verify_published_assets(
            "https://example.org/report", ["valid.png", path], fetch_status=lambda *a: requested.append(a)
        )
    assert requested == []


@pytest.mark.parametrize(
    "base",
    [
        "file:///tmp/report",
        "https://user:password@example.org/report",
        "https://example.org/?token=x",
        "https://example.org/#page",
        "https://example.org/a b",
        "https://example.org\\report",
    ],
)
def test_invalid_base_url_rejected(base):
    with pytest.raises(ValueError):
        verify_published_assets(base, [])


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_timeout_rejected(timeout):
    with pytest.raises(ValueError):
        verify_published_assets("https://example.org", [], timeout=timeout)


def test_http_status_transport_errors_deduplication_and_no_error_message_leak():
    requested = []
    body = BytesIO(b"response")

    def fetch(url, timeout):
        requested.append(url)
        if url.endswith("status.png"):
            return 503
        if url.endswith("error.png"):
            raise HTTPError(url, 404, "private message", {}, body)
        if url.endswith("timeout.png"):
            raise TimeoutError("private message")
        if url.endswith("network.png"):
            raise URLError("private message")
        return 200

    failures = verify_published_assets(
        "https://example.org",
        ["ok.png", "status.png", "error.png", "timeout.png", "network.png", "ok.png"],
        fetch_status=fetch,
    )
    assert failures == (
        AssetFailure("status.png", "http_status", 503),
        AssetFailure("error.png", "http_status", 404),
        AssetFailure("timeout.png", "TimeoutError"),
        AssetFailure("network.png", "URLError"),
    )
    assert len(requested) == 5
    assert body.closed
    assert "private message" not in repr(failures)


def test_default_get_closes_response_and_disables_redirects(monkeypatch):
    class Response:
        status = 200
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

    response = Response()
    requests = []

    def build_opener(handler):
        assert handler.redirect_request(None, None, 302, "Found", {}, "https://example.org/login") is None

        def request(url, *, timeout):
            requests.append((url, timeout))
            return response

        return SimpleNamespace(open=request)

    monkeypatch.setattr("urllib.request.build_opener", build_opener)
    assert verify_published_assets("https://example.org", ["figure.png"], timeout=4) == ()
    assert requests == [("https://example.org/figure.png", 4)]
    assert response.closed


def test_non_200_default_response_is_reported(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *args: SimpleNamespace(open=lambda *a, **k: nullcontext(SimpleNamespace(status=302))),
    )
    assert verify_published_assets("https://example.org", ["index.html"]) == (
        AssetFailure("index.html", "http_status", 302),
    )
