# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""File-reading facts and nearest-row access without report visibility policy."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FileRead[T]:
    """A file-read result, independent of schema and publication eligibility.

    Inspect ``status`` before ``value``. JSON ``null`` is a present value of
    ``None``; a parsed header-only CSV is a present empty DataFrame. A present
    value does not establish schema validity, freshness or permission to show it.
    The frozen record does not make the parsed value immutable.
    """

    path: Path
    """Requested local path, without resolving symlinks or changing relativity."""

    status: Literal["present", "missing", "invalid"]
    """Successful parsing, an absent path, or a read/parse failure."""

    value: T | None = None
    """Parsed value when present; otherwise ``None``."""

    error_type: str | None = None
    """Exception class for invalid input, without its message or file contents."""

    reason: Literal["empty_document", "parse_error", "decode_error", "read_error"] | None = None
    """Invalid-input category; successful and missing reads have no reason."""


def read_csv(path: str | os.PathLike[str], *, index_col: int | str | None = None) -> FileRead[pd.DataFrame]:
    """Read a UTF-8 CSV and distinguish missing files from unreadable input.

    Parameters
    ----------
    path
        Local file path. File-not-found errors are recorded as missing; other
        operating-system errors are invalid reads.
    index_col
        Optional index column passed to pandas, by position or name. This
        preserves the report callers that read indexed diagnostic tables.

    Returns
    -------
    FileRead[pandas.DataFrame]
        Present parsed data, missing path, or invalid data with a failure code.
        A zero-byte/whitespace-only file has no columns and is invalid with
        ``empty_document``; a header-only table is present and empty.

    Notes
    -----
    Invalid means a parser, decoder or file-read error, not a failed schema or
    diagnostic check. Pandas can accept short rows by filling cells with NaN;
    such a table is present. Caller option errors, such as an invalid index
    column, propagate. No bad-row skipping or publication policy is applied.
    """
    source = Path(path)
    try:
        value = pd.read_csv(source, index_col=index_col, encoding="utf-8")
    except FileNotFoundError:
        return FileRead(source, "missing")
    except pd.errors.EmptyDataError as exc:
        return FileRead(source, "invalid", error_type=type(exc).__name__, reason="empty_document")
    except pd.errors.ParserError as exc:
        return FileRead(source, "invalid", error_type=type(exc).__name__, reason="parse_error")
    except UnicodeError as exc:
        return FileRead(source, "invalid", error_type=type(exc).__name__, reason="decode_error")
    except OSError as exc:
        return FileRead(source, "invalid", error_type=type(exc).__name__, reason="read_error")
    return FileRead(source, "present", value)


def _reject_json_constant(token: str) -> None:
    raise ValueError("JSON does not permit nonfinite numeric constants")


def _finite_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("JSON number cannot be represented as a finite float")
    return value


def read_json(path: str | os.PathLike[str]) -> FileRead[Any]:
    """Read a UTF-8 JSON value without choosing a report's expected schema.

    Parameters
    ----------
    path
        Local file path.

    Returns
    -------
    FileRead
        Present parsed JSON, missing file, or invalid read/parse facts. JSON
        null, scalars and empty containers are present. An empty/whitespace-only
        document is invalid with ``empty_document`` rather than present null.

    Notes
    -----
    Bare NaN and Infinity tokens and numbers that overflow Python's float
    representation are rejected. This additive reader is stricter than
    ``ReportData.load_json``, whose existing json.loads behavior is unchanged.
    Duplicate object keys retain Python's last-value behavior. No gate, schema,
    freshness or publication check follows parsing.
    """
    source = Path(path)
    try:
        contents = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        return FileRead(source, "missing")
    except UnicodeError as exc:
        return FileRead(source, "invalid", error_type=type(exc).__name__, reason="decode_error")
    except OSError as exc:
        return FileRead(source, "invalid", error_type=type(exc).__name__, reason="read_error")
    if not contents.strip(" \t\r\n"):
        return FileRead(source, "invalid", error_type="JSONDecodeError", reason="empty_document")
    try:
        value = json.loads(contents, parse_constant=_reject_json_constant, parse_float=_finite_json_float)
    except (ValueError, RecursionError) as exc:
        return FileRead(source, "invalid", error_type=type(exc).__name__, reason="parse_error")
    return FileRead(source, "present", value)


def _exact_number(value: Real) -> Fraction:
    if isinstance(value, Integral | np.bool_):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        return value
    return Fraction(*value.as_integer_ratio())


def _finite_number(value: Real) -> bool:
    return True if isinstance(value, Integral | Fraction | np.bool_) else bool(np.isfinite(value))


def nearest_row(
    frame: pd.DataFrame | None,
    *,
    key: str,
    at: float,
    max_distance: float | None = None,
) -> pd.Series | None:
    """Select the first nearest row without interpolation or changing its index.

    Parameters
    ----------
    frame
        Table to search, or None when unavailable. The returned Series retains
        its original row label, including a duplicated index label.
    key
        Unique column used for distance. Numeric strings are converted with
        pandas; missing, unconvertible and nonfinite keys are skipped.
    at
        Finite real query value. A nonfinite query has no nearest row.
    max_distance
        Optional finite, nonnegative bound on absolute key distance. Equality
        is included. With no bound, a query outside the grid selects its nearest
        endpoint; it does not interpolate or establish support for that query.

    Returns
    -------
    pandas.Series or None
        First row in input order attaining the minimum distance, or None when
        the frame/key is absent, no finite keys remain, or the bound is exceeded.

    Notes
    -----
    Integer keys and integral queries use exact distances, avoiding unsigned
    subtraction, signed overflow and rounded integer queries. Floating keys
    with floating queries retain pandas' normal arithmetic; an overflow falls
    back to exact distances between the represented values.
    Other columns and the input table are not modified. Duplicated key columns
    are ambiguous and raise; scientific visibility remains the caller's choice.
    """
    if max_distance is not None:
        if isinstance(max_distance, bool | np.bool_) or not isinstance(max_distance, Real):
            raise TypeError("max_distance must be a real number")
        if not _finite_number(max_distance) or max_distance < 0:
            raise ValueError("max_distance must be finite and nonnegative")
    if not isinstance(at, Real | np.bool_):
        raise TypeError("at must be a real number")
    if not _finite_number(at) or frame is None:
        return None
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame or None")
    if frame.empty or key not in frame.columns:
        return None
    column = frame[key]
    if not isinstance(column, pd.Series):
        raise ValueError("key must identify one column")
    keys = pd.to_numeric(column, errors="coerce")
    if keys.dtype.kind == "c":
        raise TypeError("key values must be real numbers")
    positions = np.flatnonzero(np.isfinite(keys).fillna(False).to_numpy(dtype=bool))
    if not len(positions):
        return None
    candidates = keys.iloc[positions]
    if keys.dtype.kind in "iub" or isinstance(at, Integral | Fraction | np.bool_):
        query = _exact_number(at)
        distances = [abs(_exact_number(value) - query) for value in candidates]
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            distances = (candidates - at).abs().to_numpy()
        if not np.isfinite(distances).all():
            query = _exact_number(at)
            distances = [abs(_exact_number(value) - query) for value in candidates]
    position = min(range(len(positions)), key=distances.__getitem__)
    if max_distance is not None and distances[position] > max_distance:
        return None
    return frame.iloc[positions[position]]
