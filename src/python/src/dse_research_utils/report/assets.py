# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Inspect direct HTML resources and check explicit publication inventories.

These checks do not render pages, traverse CSS or JavaScript dependencies, or
decide whether scientific results may be published. No upload is performed.
"""

from __future__ import annotations

import math
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal
from urllib.parse import quote, unquote, urlsplit


@dataclass(frozen=True)
class AssetReference:
    """One URL-bearing HTML attribute, including references absent on disk."""

    page: str
    """Raw path of the referring page, relative to the inspection root."""
    tag: str
    """HTML element name."""
    attribute: str
    """Attribute containing the reference."""
    target: str
    """Attribute value after HTML entity decoding, before URL decoding."""
    kind: Literal["resource", "navigation"]
    """A rendered resource or an ordinary link."""
    required: bool
    """Whether local availability and upload membership will be checked."""
    status: Literal["present", "missing", "external", "invalid", "unsupported"]
    """Local file state, nonlocal reference, or an unresolved reference."""
    relative_path: str | None = None
    """Decoded raw filename relative to the inspection root, when resolvable."""
    reason: str | None = None
    """Machine-readable reason for an invalid or unsupported reference."""


@dataclass(frozen=True)
class AssetInspection:
    """Snapshot of inspected pages and their direct HTML references."""

    root: Path
    """Absolute local directory corresponding to the upload prefix."""
    pages: tuple[str, ...]
    """Raw root-relative paths of the pages actually parsed."""
    references: tuple[AssetReference, ...]
    """References in page and HTML encounter order, including navigation."""

    @property
    def required_paths(self) -> tuple[str, ...]:
        """Return sorted raw paths of pages and required local references.

        Missing files remain in this list. Invalid and unsupported references
        may have no resolved path; inspect failures before using it for copying.
        """
        return tuple(
            sorted(
                set(self.pages)
                | {ref.relative_path for ref in self.references if ref.required and ref.relative_path is not None}
            )
        )


@dataclass(frozen=True)
class AssetFailure:
    """A local, upload-inventory or HTTP verification failure."""

    path: str
    """Raw relative filename, or original target if it could not be resolved."""
    reason: str
    """Failure code, for example ``missing``, ``not_uploaded`` or ``http_status``."""
    status_code: int | None = None
    """HTTP response code, if one was received."""


class _References(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[tuple[str, str, str, Literal["resource", "navigation"], str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes: dict[str, str] = {}
        for name, value in attrs:
            # Browsers use the first duplicate attribute, not the last.
            attributes.setdefault(name, value or "")
        if tag == "base" and attributes.get("href", ""):
            self.references.append((tag, "href", attributes["href"], "resource", "unsupported_base"))
            return
        resource_attributes = {
            "img": ("src",),
            "script": ("src",),
            "iframe": ("src",),
            "embed": ("src",),
            "audio": ("src",),
            "video": ("src", "poster"),
            "source": ("src",),
            "track": ("src",),
            "object": ("data",),
            "image": ("href", "xlink:href"),
            "use": ("href", "xlink:href"),
        }
        selected = resource_attributes.get(tag, ())
        if tag == "input" and attributes.get("type", "").lower() == "image":
            selected = ("src",)
        if tag == "link":
            rel = set(attributes.get("rel", "").lower().split())
            if rel & {"stylesheet", "icon", "preload", "modulepreload", "apple-touch-icon", "manifest"}:
                selected = ("href",)
        for attribute in selected:
            if attribute in attributes:
                self.references.append((tag, attribute, attributes[attribute], "resource", None))
        if tag in ("a", "area") and "href" in attributes:
            self.references.append((tag, "href", attributes["href"], "navigation", None))
        # Do not silently certify a responsive image without inspecting every
        # candidate. A future parser can replace this explicit unsupported state.
        if tag in ("img", "source", "link"):
            for attribute in ("srcset", "imagesrcset"):
                if attributes.get(attribute):
                    self.references.append((tag, attribute, attributes[attribute], "resource", "unsupported_srcset"))


def _local_target(target: str, page: Path, root: Path) -> tuple[str, str | None, str | None]:
    target = target.strip(" \t\n\r\f")
    if any(ord(char) < 32 or ord(char) == 127 for char in target) or "\\" in target:
        return "invalid", None, "invalid_url"
    try:
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            return "external", None, None
        if not target or target.startswith("#"):
            return "external", None, None
        if re.search(r"%(?![0-9a-fA-F]{2})", parsed.path):
            return "invalid", None, "invalid_url_encoding"
        if re.search(r"%(?:2f|5c)", parsed.path, flags=re.IGNORECASE):
            return "invalid", None, "encoded_path_separator"
        decoded = unquote(parsed.path, encoding="utf-8", errors="strict")
        if decoded.startswith("/"):
            return "invalid", None, "root_relative_url"
        if "//" in decoded:
            return "invalid", None, "empty_path_segment"
        if decoded and decoded.split("/")[-1] in ("", ".", ".."):
            return "invalid", None, "directory_url"
        if "\\" in decoded or any(ord(char) < 32 or ord(char) == 127 for char in decoded):
            return "invalid", None, "invalid_url"
        candidate = Path(os.path.normpath(page.parent / decoded)) if decoded else page
        if not candidate.is_relative_to(root):
            return "invalid", None, "outside_root"
        relative = candidate.relative_to(root).as_posix()
        # Check lexical traversal and symlink traversal separately. Keep the
        # lexical filename because that is the uploader's inventory identity.
        if ".." in PurePosixPath(relative).parts or not candidate.resolve().is_relative_to(root):
            return "invalid", None, "outside_root"
        return ("present" if candidate.is_file() else "missing"), relative, None
    except UnicodeError, ValueError, OSError:
        return "invalid", None, "invalid_local_path"


def inspect_local_assets(
    html_path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] | None = None,
    include_navigation: bool = False,
    follow_pages: bool = False,
) -> AssetInspection:
    """Parse local HTML references without dropping missing files.

    Parameters
    ----------
    html_path
        Entry page, read as UTF-8. Read and decoding errors propagate.
    root
        Local upload root; defaults to the entry page's directory. Paths may
        traverse to a parent only while remaining inside this root. Absolute
        URL paths and symlinks escaping the root are invalid.
    include_navigation
        Also require local ``a`` and ``area`` links, including downloads.
        External URLs, embedded data and same-page fragments are recorded but
        never fetched or checked against the upload inventory.
    follow_pages
        Parse referenced local ``.html`` and ``.htm`` pages recursively. Such
        links become required even when ``include_navigation`` is false.
        Otherwise only the entry page's direct HTML references are inspected.
        A new URL alias of an ancestor page fails as unsupported, to stop
        symlink cycles without claiming to have checked the alias's resources.

    Returns
    -------
    AssetInspection
        Pages, resource and navigation references, and local file states.

    Notes
    -----
    HTML entities are decoded by the parser; URL paths are decoded once, with
    query strings and fragments excluded from filenames. Literal plus signs
    remain plus signs. A literal percent sign in a filename needs ``%25`` in
    HTML. Present directories are not files. Missing navigation is visible
    even when it is not required.

    Only direct resource attributes are inspected. CSS URLs, JavaScript imports,
    dynamically generated references and document content types are not checked.
    Nonempty ``base href`` and responsive ``srcset`` attributes produce explicit
    unsupported failures, rather than an incomplete success. This is a snapshot;
    callers must keep files stable through copying and upload.
    """
    entry = Path(os.path.abspath(html_path))
    lexical_root = Path(os.path.abspath(root)) if root is not None else entry.parent
    local_root = lexical_root.resolve()
    # Translate a symlinked upload root without resolving aliases *within* it.
    # The requested page's lexical suffix is also its published URL identity.
    if entry.is_relative_to(lexical_root):
        entry = local_root / entry.relative_to(lexical_root)
    if not entry.resolve().is_relative_to(local_root) or not entry.is_relative_to(local_root):
        raise ValueError("The entry page must be inside root")
    queue: list[tuple[Path, tuple[Path, ...]]] = [(entry, ())]
    scheduled = {entry.relative_to(local_root).as_posix()}
    references: list[AssetReference] = []
    pages: list[str] = []
    for page, ancestors in queue:
        relative_page = page.relative_to(local_root).as_posix()
        parser = _References()
        parser.feed(page.read_text(encoding="utf-8"))
        parser.close()
        pages.append(relative_page)
        ancestry = (*ancestors, page.resolve())
        for tag, attribute, target, kind, unsupported in parser.references:
            status, relative, reason = _local_target(target, page, local_root)
            if kind == "resource" and not target.strip(" \t\n\r\f"):
                status, relative, reason = "invalid", None, "empty_url"
            is_page = relative is not None and PurePosixPath(relative).suffix.lower() in (".html", ".htm")
            if follow_pages and status == "invalid":
                try:
                    target_path = unquote(urlsplit(target.strip(" \t\n\r\f")).path)
                    is_page = PurePosixPath(target_path).suffix.lower() in (".html", ".htm")
                except ValueError:
                    pass
            required = kind == "resource" or include_navigation or (follow_pages and is_page)
            if unsupported:
                status, relative, reason = "unsupported", None, unsupported
            if follow_pages and is_page and status == "present" and relative not in scheduled:
                next_page = local_root / relative
                if next_page.resolve() in ancestry:
                    status, reason = "unsupported", "cyclic_page_alias"
                else:
                    scheduled.add(relative)
                    queue.append((next_page, ancestry))
            references.append(
                AssetReference(relative_page, tag, attribute, target, kind, required, status, relative, reason)
            )
    return AssetInspection(local_root, tuple(pages), tuple(references))


def _raw_relative_path(path: str) -> str:
    if not isinstance(path, str) or not path or any(ord(char) < 32 or ord(char) == 127 for char in path):
        raise ValueError("Expected a nonempty raw relative filename")
    normalized = path.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if (
        normalized.startswith("/")
        or PureWindowsPath(path).drive
        or ".." in parts
        or not parts
        or "//" in normalized
        or normalized.split("/")[-1] in ("", ".", "..")
    ):
        raise ValueError("Expected a raw filename inside the upload root")
    return PurePosixPath(normalized).as_posix()


def check_uploaded_assets(inspection: AssetInspection, published_paths: Iterable[str]) -> tuple[AssetFailure, ...]:
    """Check required local references against raw uploaded filenames.

    Pass the uploader's ``relative_paths``, not its URLs. Windows separators are
    accepted; paths are not URL-decoded. Missing and unsupported local resources
    fail even if an upload inventory claims to contain them. The entry page and
    every recursively inspected page are required. This performs no network I/O.
    """
    if isinstance(published_paths, str):
        raise TypeError("published_paths must be an iterable of filenames, not a string")
    sent = {_raw_relative_path(path) for path in published_paths}
    failures: list[AssetFailure] = []
    for ref in inspection.references:
        if ref.required and ref.status in ("missing", "invalid", "unsupported"):
            failures.append(AssetFailure(ref.relative_path or ref.target, ref.reason or ref.status))
    failed_paths = {failure.path for failure in failures}
    failures.extend(
        AssetFailure(path, "not_uploaded")
        for path in inspection.required_paths
        if path not in sent and path not in failed_paths
    )
    return tuple(dict.fromkeys(failures))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: urllib.request.Request, fp: object, code: int, msg: str, headers: object, newurl: str
    ) -> None:
        return None


def _fetch_status(url: str, timeout: float) -> int:
    with urllib.request.build_opener(_NoRedirect()).open(url, timeout=timeout) as response:
        return response.status


def verify_published_assets(
    base_url: str,
    relative_paths: Iterable[str],
    *,
    timeout: float = 30.0,
    fetch_status: Callable[[str, float], int] | None = None,
) -> tuple[AssetFailure, ...]:
    """GET explicitly listed published paths and return structured failures.

    Parameters
    ----------
    base_url
        HTTP(S) directory URL corresponding to the inspection/upload root.
        Must have no credentials, query or fragment. A trailing slash is optional.
    relative_paths
        Raw filenames, for example ``inspection.required_paths``. Encode each
        path exactly once here; never pass already encoded URLs. Duplicates are
        checked once in first-encounter order. Local/upload checks are separate.
    timeout
        Positive finite request timeout in seconds.
    fetch_status
        Optional transport receiving ``(url, timeout)`` and returning an integer
        HTTP status. It must not follow redirects. The default uses urllib GET
        without redirects and closes the response without downloading its body.

    Returns
    -------
    tuple of AssetFailure
        Non-200 responses and transport exceptions. Exception messages and URLs
        are not copied into failures. Invalid arguments raise before any request.

    Notes
    -----
    A 200 response checks reachability, not content, MIME type or scientific
    publication eligibility. Redirects fail, including redirects to login pages.
    This function never uploads files or requests discovered external references.
    """
    parsed = urlsplit(base_url)
    # Accessing port performs urllib's numeric and range validation. A nonempty
    # netloc alone still accepts a missing host or an unusable port.
    _ = parsed.port
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(ord(char) <= 32 or ord(char) == 127 for char in base_url)
        or "\\" in base_url
    ):
        raise ValueError("base_url must be a plain HTTP(S) directory URL without credentials, query or fragment")
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if isinstance(relative_paths, str):
        raise TypeError("relative_paths must be an iterable of filenames, not a string")
    paths = tuple(dict.fromkeys(_raw_relative_path(path) for path in relative_paths))
    fetch = fetch_status if fetch_status is not None else _fetch_status
    failures: list[AssetFailure] = []
    for path in paths:
        url = base_url.rstrip("/") + "/" + quote(path, safe="/")
        try:
            status = fetch(url, timeout)
            if isinstance(status, bool) or not isinstance(status, int):
                raise TypeError("fetch_status must return an integer HTTP status")
            if status != 200:
                failures.append(AssetFailure(path, "http_status", status))
        except urllib.error.HTTPError as exc:
            failures.append(AssetFailure(path, "http_status", exc.code))
            exc.close()
        except Exception as exc:
            failures.append(AssetFailure(path, type(exc).__name__))
    return tuple(failures)
