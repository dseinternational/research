# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest

from dse_research_utils.statistics.transforms import (
    convert_to_categorical,
    invlogit,
    logit,
    standardize,
)


def test_standardize_zero_mean_unit_sd():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = standardize(x)
    assert np.mean(z) == pytest.approx(0.0, abs=1e-12)
    assert np.std(z) == pytest.approx(1.0)


def test_standardize_constant_returns_centred():
    x = np.array([3.0, 3.0, 3.0])
    z = standardize(x)
    assert np.allclose(z, 0.0)


def test_logit_invlogit_roundtrip():
    p = np.array([0.1, 0.4, 0.5, 0.9])
    assert np.allclose(invlogit(logit(p)), p)


def test_convert_to_categorical_series_and_frame():
    s = pd.Series(["a", "b", "a", "c"])
    codes = convert_to_categorical(s)
    assert list(codes) == [0, 1, 0, 2]
    df = pd.DataFrame({"x": ["a", "b"], "y": ["q", "q"]})
    out = convert_to_categorical(df)
    assert list(out["x"]) == [0, 1]
    assert list(out["y"]) == [0, 0]


def test_standardise_returns_invertible_transform():
    from dse_research_utils.statistics.transforms import standardise

    x = np.array([1.0, 2.0, np.nan, 4.0, 8.0])
    z, scaler = standardise(x)
    # NaN-aware: statistics computed over the finite values only.
    assert np.nanmean(z) == pytest.approx(0.0, abs=1e-12)
    assert np.nanstd(z, ddof=1) == pytest.approx(1.0)
    assert np.isnan(z[2])
    # The fitted transform round-trips exactly.
    assert np.allclose(scaler.inverse(z[[0, 1, 3, 4]]), x[[0, 1, 3, 4]])
    assert np.allclose(scaler(x[[0, 1, 3, 4]]), z[[0, 1, 3, 4]])


def test_standardise_raises_on_constant_input():
    from dse_research_utils.statistics.transforms import standardise

    with pytest.raises(ValueError, match="positive"):
        standardise(np.array([3.0, 3.0, 3.0]))


def test_haldane_logit_finite_at_bounds_and_nan_passthrough():
    from dse_research_utils.statistics.transforms import haldane_logit

    n = 10
    out = haldane_logit(np.array([0.0, 5.0, 10.0, np.nan]), n)
    assert np.isfinite(out[0]) and np.isfinite(out[2])
    assert out[1] == pytest.approx(0.0)  # symmetric at y = n/2
    assert out[0] == pytest.approx(-out[2])  # symmetric about the midpoint
    assert np.isnan(out[3])
    # Matches the log((y+.5)/(n-y+.5)) definition.
    assert out[0] == pytest.approx(np.log(0.5 / 10.5))
