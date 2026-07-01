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
