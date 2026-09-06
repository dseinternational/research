# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publication identity at HTML, filesystem and HTTP URL boundaries."""

from pathlib import Path

import pytest

from dse_research_utils.report.assets import check_uploaded_assets, inspect_local_assets, verify_published_assets


@pytest.mark.parametrize(
    ("target", "filename"),
    [
        ("space%20name.png", "space name.png"),
        ("50%2520.csv", "50%20.csv"),
        ("plus+name.png", "plus+name.png"),
        ("caf%C3%A9.png", "café.png"),
        ("figure%23question%3F.png", "figure#question?.png"),
        ("amp&amp;name.png", "amp&name.png"),
    ],
)
def test_html_and_url_decoding_preserve_raw_publication_identity(tmp_path, target, filename):
    (tmp_path / filename).write_bytes(b"resource")
    page = tmp_path / "index.html"
    page.write_text(f"<img src='{target}?version=1#preview'>", encoding="utf-8")
    inspection = inspect_local_assets(page)
    reference = inspection.references[0]
    assert reference.status == "present"
    assert reference.relative_path == filename
    assert check_uploaded_assets(inspection, ["index.html", filename]) == ()


def test_literal_percent_filename_does_not_hide_missing_decoded_target(tmp_path):
    (tmp_path / "50%20.csv").write_text("unrelated file", encoding="utf-8")
    page = tmp_path / "index.html"
    page.write_text('<a href="50%20.csv">Download</a>', encoding="utf-8")
    inspection = inspect_local_assets(page, include_navigation=True)
    assert inspection.references[0].status == "missing"
    assert inspection.references[0].relative_path == "50 .csv"
    failures = check_uploaded_assets(inspection, ["index.html", "50%20.csv"])
    assert [(failure.path, failure.reason) for failure in failures] == [("50 .csv", "missing")]


@pytest.mark.parametrize("whitespace", ["\u00a0", "\u2003"])
def test_unicode_whitespace_is_part_of_the_requested_filename(tmp_path, whitespace):
    (tmp_path / "plot.png").write_bytes(b"different resource")
    page = tmp_path / "index.html"
    page.write_text(f'<img src="{whitespace}plot.png">', encoding="utf-8")
    inspection = inspect_local_assets(page)
    assert inspection.references[0].relative_path == f"{whitespace}plot.png"
    assert inspection.references[0].status == "missing"
    (tmp_path / f"{whitespace}plot.png").write_bytes(b"requested resource")
    assert inspect_local_assets(page).references[0].status == "present"


def test_internal_symlink_retains_the_browser_requested_alias(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "real.png").write_bytes(b"image")
    (tmp_path / "alias.png").symlink_to("images/real.png")
    page = tmp_path / "index.html"
    page.write_text('<img src="alias.png">', encoding="utf-8")
    inspection = inspect_local_assets(page)
    assert inspection.references[0].relative_path == "alias.png"
    assert inspection.references[0].status == "present"
    failures = check_uploaded_assets(inspection, ["index.html", "images/real.png"])
    assert [(failure.path, failure.reason) for failure in failures] == [("alias.png", "not_uploaded")]


def test_entry_page_under_internal_directory_symlink_retains_upload_path(tmp_path):
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "index.html").write_text('<img src="image.png">', encoding="utf-8")
    (tmp_path / "pages" / "image.png").write_bytes(b"image")
    (tmp_path / "alias").symlink_to("pages", target_is_directory=True)
    inspection = inspect_local_assets(tmp_path / "alias" / "index.html", root=tmp_path)
    assert inspection.pages == ("alias/index.html",)
    assert inspection.required_paths == ("alias/image.png", "alias/index.html")
    failures = check_uploaded_assets(inspection, ["pages/index.html", "pages/image.png"])
    assert {failure.path for failure in failures} == {"alias/index.html", "alias/image.png"}


@pytest.mark.parametrize("use_root_alias", [False, True])
def test_symlinked_upload_root_preserves_internal_alias_suffix(tmp_path, use_root_alias):
    root = tmp_path / "site"
    (root / "pages").mkdir(parents=True)
    (root / "pages" / "index.html").write_text('<img src="image.png">', encoding="utf-8")
    (root / "pages" / "image.png").write_bytes(b"image")
    (root / "alias").symlink_to("pages", target_is_directory=True)
    root_alias = tmp_path / "linked-root"
    root_alias.symlink_to(root, target_is_directory=True)
    entry = (root_alias if use_root_alias else root) / "alias" / "index.html"
    inspection = inspect_local_assets(entry, root=root_alias)
    assert inspection.root == root.resolve()
    assert inspection.pages == ("alias/index.html",)
    assert inspection.required_paths == ("alias/image.png", "alias/index.html")


def test_dot_segments_are_normalized_before_following_filesystem_symlinks(tmp_path):
    (tmp_path / "deep" / "nested").mkdir(parents=True)
    (tmp_path / "alias").symlink_to("deep/nested", target_is_directory=True)
    (tmp_path / "image.png").write_bytes(b"browser target")
    page = tmp_path / "index.html"
    page.write_text('<img src="alias/../image.png">', encoding="utf-8")
    inspection = inspect_local_assets(page)
    assert inspection.references[0].relative_path == "image.png"
    assert inspection.references[0].status == "present"


def test_parent_paths_inside_root_are_distinct_from_encoded_and_symlink_escapes(tmp_path):
    root = tmp_path / "site"
    (root / "docs" / "nested").mkdir(parents=True)
    (root / "image.png").write_bytes(b"image")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside image")
    (root / "outside-link.png").symlink_to(outside)
    page = root / "docs" / "nested" / "index.html"
    page.write_text(
        '<img src="../../image.png">'
        '<img src="%2e%2e/%2e%2e/%2e%2e/outside.png">'
        '<img src="../../outside-link.png">'
        '<img src="..%5c..%5coutside.png">'
        '<img src="/image.png">',
        encoding="utf-8",
    )
    inspection = inspect_local_assets(page, root=root)
    assert [(reference.relative_path, reference.status) for reference in inspection.references] == [
        ("image.png", "present"),
        (None, "invalid"),
        (None, "invalid"),
        (None, "invalid"),
        (None, "invalid"),
    ]


def test_duplicate_source_attribute_uses_the_first_browser_value(tmp_path):
    (tmp_path / "present.png").write_bytes(b"image")
    page = tmp_path / "index.html"
    page.write_text(
        '<img src="missing.png" src="present.png">'
        '<img src="present.png" src="missing.png">'
        '<!-- <img src="comment.png"> -->'
        '<script>const example = "<img src=script-text.png>";</script>',
        encoding="utf-8",
    )
    inspection = inspect_local_assets(page)
    assert [(reference.relative_path, reference.status) for reference in inspection.references] == [
        ("missing.png", "missing"),
        ("present.png", "present"),
    ]


@pytest.mark.parametrize("include_navigation", [False, True])
@pytest.mark.parametrize("follow_pages", [False, True])
def test_navigation_requiredness_and_page_crawling_are_independent(tmp_path, include_navigation, follow_pages):
    page = tmp_path / "index.html"
    page.write_text('<a href="child.htm">Page</a><a href="missing.csv">Download</a>', encoding="utf-8")
    (tmp_path / "child.htm").write_text(
        '<a href="index.html#top">Back</a><a href="?refresh=1">Refresh</a><img src="missing.png">', encoding="utf-8"
    )
    inspection = inspect_local_assets(page, include_navigation=include_navigation, follow_pages=follow_pages)
    assert inspection.pages == (("index.html", "child.htm") if follow_pages else ("index.html",))
    assert inspection.references[0].required == (include_navigation or follow_pages)
    assert inspection.references[1].required == include_navigation
    failures = check_uploaded_assets(inspection, ["index.html", "child.htm"])
    expected_missing = ({"missing.csv"} if include_navigation else set()) | ({"missing.png"} if follow_pages else set())
    assert {failure.path for failure in failures} == expected_missing


@pytest.mark.parametrize("target", ["../outside.html", "/outside.html"])
def test_crawling_rejects_invalid_local_html_links_without_reading_them(tmp_path, monkeypatch, target):
    root = tmp_path / "site"
    root.mkdir()
    (tmp_path / "outside.html").write_text("Must not be inspected", encoding="utf-8")
    page = root / "index.html"
    page.write_text(f'<a href="{target}">Outside</a>', encoding="utf-8")
    read_text = Path.read_text
    calls = []

    def guarded_read(path, *args, **kwargs):
        calls.append(path)
        assert path == page, "The crawler must not read paths beyond its root"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    inspection = inspect_local_assets(page, follow_pages=True)
    assert calls == [page]
    assert inspection.references[0].status == "invalid"
    assert inspection.references[0].required
    assert len(check_uploaded_assets(inspection, ["index.html"])) == 1


def test_recursive_directory_alias_fails_visibly_without_repeated_reads(tmp_path, monkeypatch):
    (tmp_path / "again").symlink_to(".", target_is_directory=True)
    page = tmp_path / "index.html"
    page.write_text('<a href="again/index.html">Loop</a>', encoding="utf-8")
    read_text = Path.read_text
    calls = []

    def guarded_read(path, *args, **kwargs):
        calls.append(path)
        assert path == page, "The crawler must stop a page alias before reading it again"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    inspection = inspect_local_assets(page, follow_pages=True)
    assert inspection.pages == ("index.html",)
    assert calls == [page]
    reference = inspection.references[0]
    assert reference.status == "unsupported"
    assert reference.reason == "cyclic_page_alias"
    assert reference.relative_path == "again/index.html"
    assert reference.required
    failures = check_uploaded_assets(inspection, inspection.required_paths)
    assert [(failure.path, failure.reason) for failure in failures] == [("again/index.html", "cyclic_page_alias")]


def test_distinct_finite_page_aliases_are_checked_in_each_browser_url_context(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "report.html").write_text('<img src="image.png">', encoding="utf-8")
    for name in ("first", "second"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "report.html").symlink_to("../shared/report.html")
    (tmp_path / "first" / "image.png").write_bytes(b"image")
    page = tmp_path / "index.html"
    page.write_text('<a href="first/report.html">First</a><a href="second/report.html">Second</a>', encoding="utf-8")
    inspection = inspect_local_assets(page, follow_pages=True)
    assert inspection.pages == ("index.html", "first/report.html", "second/report.html")
    failures = check_uploaded_assets(inspection, inspection.required_paths)
    assert [(failure.path, failure.reason) for failure in failures] == [("second/image.png", "missing")]


@pytest.mark.parametrize("links", [("first.html", "alias.html"), ("alias.html", "first.html")])
def test_already_scheduled_page_alias_does_not_fail_by_link_order(tmp_path, links):
    (tmp_path / "first.html").write_text('<a href="alias.html">Alias</a>', encoding="utf-8")
    (tmp_path / "alias.html").symlink_to("first.html")
    page = tmp_path / "index.html"
    page.write_text("".join(f'<a href="{link}">{link}</a>' for link in links), encoding="utf-8")
    inspection = inspect_local_assets(page, follow_pages=True)
    assert set(inspection.pages) == {"index.html", "first.html", "alias.html"}
    assert check_uploaded_assets(inspection, inspection.required_paths) == ()


def test_unsupported_reference_forms_cannot_pass_a_complete_upload_inventory(tmp_path):
    page = tmp_path / "index.html"
    page.write_text(
        '<img src="picture.png" srcset="picture.png 1x, retina.png 2x"><base href="other/">', encoding="utf-8"
    )
    (tmp_path / "picture.png").write_bytes(b"image")
    (tmp_path / "retina.png").write_bytes(b"image")
    inspection = inspect_local_assets(page)
    failures = check_uploaded_assets(inspection, ["index.html", "picture.png", "retina.png"])
    assert {failure.reason for failure in failures} == {"unsupported_srcset", "unsupported_base"}


def test_directory_does_not_count_as_a_present_resource(tmp_path):
    (tmp_path / "picture.png").mkdir()
    page = tmp_path / "index.html"
    page.write_text('<img src="picture.png">', encoding="utf-8")
    inspection = inspect_local_assets(page)
    assert inspection.references[0].status == "missing"
    assert check_uploaded_assets(inspection, inspection.required_paths)[0].reason == "missing"


def test_remote_verification_preserves_encoded_base_and_encodes_raw_filenames_once():
    calls = []

    def fetch(url, timeout):
        calls.append((url, timeout))
        return 200

    failures = verify_published_assets(
        "https://example.test/reports%20here",
        ["50%20.csv", "plus+name.png", "café #?.png", "50%20.csv"],
        timeout=7.0,
        fetch_status=fetch,
    )
    assert failures == ()
    assert calls == [
        ("https://example.test/reports%20here/50%2520.csv", 7.0),
        ("https://example.test/reports%20here/plus%2Bname.png", 7.0),
        ("https://example.test/reports%20here/caf%C3%A9%20%23%3F.png", 7.0),
    ]


@pytest.mark.parametrize("base_url", ["https://:443/reports", "https://example.test:bad/reports"])
def test_invalid_network_authority_is_rejected_before_transport(base_url):
    calls = []
    with pytest.raises(ValueError):
        verify_published_assets(base_url, ["index.html"], fetch_status=lambda url, timeout: calls.append(url) or 200)
    assert calls == []


@pytest.mark.parametrize("bad_path", ["../outside.png", "/root.png", "//other.test/image.png", "C:\\outside.png"])
def test_all_raw_paths_are_validated_before_any_remote_request(bad_path):
    calls = []
    with pytest.raises(ValueError):
        verify_published_assets(
            "https://example.test/report",
            ["valid.html", bad_path],
            fetch_status=lambda url, timeout: calls.append(url) or 200,
        )
    assert calls == []
