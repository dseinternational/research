# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Azure Blob Storage helpers for publishing research model artefacts."""

from __future__ import annotations

import mimetypes
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_CONTAINER_URL_ENV_VAR = "DSERESEARCH_BLOB_CONTAINER_URL"


@dataclass(frozen=True)
class BlobContainerTarget:
    """Resolved Azure Blob container target."""

    account_url: str
    """Azure Blob account URL, for example ``https://acct.blob.core.windows.net``."""

    container_name: str
    """Container name parsed from the configured container URL."""

    base_url: str
    """Public base URL for blobs in the container."""


@dataclass(frozen=True)
class BlobUploadResult:
    """Summary of a directory upload."""

    urls: list[str] = field(default_factory=list)
    """Uploaded blob URLs."""

    prefix_url: str = ""
    """Common URL prefix for this upload batch."""

    report_url: str | None = None
    """Uploaded ``index.html`` URL, if present."""

    uploaded_files: int = 0
    """Number of files uploaded."""

    skipped_files: int = 0
    """Number of files skipped by trace or caller-supplied filters."""

    bytes_uploaded: int = 0
    """Total bytes uploaded."""

    elapsed_seconds: float = 0.0
    """Upload wall time in seconds."""


def container_url_from_environment(env_var: str = DEFAULT_CONTAINER_URL_ENV_VAR) -> str:
    """Read an Azure Blob container URL from an environment variable."""
    url = os.environ.get(env_var)
    if not url:
        raise RuntimeError(
            f"{env_var} environment variable is not set. Set it to a plain Azure "
            "Blob container URL such as "
            "'https://<account>.blob.core.windows.net/<container>'."
        )
    return url


def parse_blob_container_url(
    container_url: str,
    *,
    env_var: str = DEFAULT_CONTAINER_URL_ENV_VAR,
) -> BlobContainerTarget:
    """Parse and validate a plain Azure Blob container URL.

    SAS tokens and extra path segments are rejected deliberately. DSE research
    uploads authenticate with ``DefaultAzureCredential`` rather than embedding
    credentials in the URL.
    """
    parsed = urlparse(container_url.rstrip("/"))
    if parsed.query or parsed.fragment:
        raise RuntimeError(
            f"{env_var} must be a plain container URL with no query string or fragment: {container_url!r}."
        )

    path_parts = [part for part in parsed.path.split("/") if part]
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or len(path_parts) != 1:
        raise RuntimeError(
            f"{env_var} must be 'https://<account>.blob.core.windows.net/<container>' "
            f"with a single container path segment: {container_url!r}."
        )

    account_url = f"{parsed.scheme}://{parsed.netloc}"
    container_name = path_parts[0]
    return BlobContainerTarget(
        account_url=account_url,
        container_name=container_name,
        base_url=f"{account_url}/{container_name}",
    )


def upload_directory_to_blob_storage(
    output_dir: str | os.PathLike[str],
    model_label: str,
    *,
    project: str,
    container_url: str | None = None,
    env_var: str = DEFAULT_CONTAINER_URL_ENV_VAR,
    include_traces: bool = False,
    skip: Callable[[str], bool] | None = None,
    run_id: str | None = None,
    credential: Any | None = None,
) -> BlobUploadResult:
    """Upload an artefact directory to Azure Blob Storage.

    Parameters
    ----------
    output_dir
        Local directory to upload.
    model_label
        Final path segment under the upload prefix.
    project
        Project segment used in ``projects/<project>/output/...``.
    container_url
        Plain Azure Blob container URL. Defaults to ``env_var``.
    env_var
        Environment variable used when ``container_url`` is omitted.
    include_traces
        If True, include NetCDF trace files. They are skipped by default because
        reporting traces can be very large.
    skip
        Optional predicate receiving each POSIX-style relative file path. Return
        True to skip that file.
    run_id
        Upload batch identifier. Generated with ``uuid.uuid7()`` when omitted.
    credential
        Azure credential object. Defaults to ``DefaultAzureCredential``.

    Returns
    -------
    BlobUploadResult
        Structured upload summary, including all uploaded URLs.
    """
    # Lazy import keeps the package importable for users who do not publish to Azure.
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient, ContentSettings

    root = Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {root}")

    target = parse_blob_container_url(container_url or container_url_from_environment(env_var), env_var=env_var)
    if run_id is None:
        run_id = str(uuid.uuid7())

    blob_prefix = f"projects/{project}/output/{run_id}/{model_label}"
    credential = credential or DefaultAzureCredential()
    container_client = BlobServiceClient(target.account_url, credential=credential).get_container_client(
        target.container_name
    )

    urls: list[str] = []
    uploaded = 0
    skipped = 0
    bytes_sent = 0
    started = time.perf_counter()

    for local_path, relative_path in _iter_upload_candidates(root):
        if not include_traces and local_path.suffix == ".nc":
            skipped += 1
            continue
        if skip is not None and skip(relative_path):
            skipped += 1
            continue

        blob_name = f"{blob_prefix}/{relative_path}"
        content_type, _encoding = mimetypes.guess_type(local_path.name)
        content_type = content_type or "application/octet-stream"

        bytes_sent += local_path.stat().st_size
        with local_path.open("rb") as file_handle:
            container_client.upload_blob(
                blob_name,
                file_handle,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )
        urls.append(f"{target.base_url}/{blob_name}")
        uploaded += 1

    report_url = next((url for url in urls if url.endswith("/index.html")), None)
    return BlobUploadResult(
        urls=urls,
        prefix_url=f"{target.base_url}/{blob_prefix}/",
        report_url=report_url,
        uploaded_files=uploaded,
        skipped_files=skipped,
        bytes_uploaded=bytes_sent,
        elapsed_seconds=time.perf_counter() - started,
    )


def _iter_upload_candidates(root: Path) -> list[tuple[Path, str]]:
    """Return upload candidates as ``(path, posix_relative_path)`` pairs."""
    return [(path, path.relative_to(root).as_posix()) for path in sorted(root.rglob("*")) if path.is_file()]
