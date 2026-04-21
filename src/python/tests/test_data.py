# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest

from dse_research_utils.statistics.models.data import BinomialModelData, ModelData


class TestModelData:
    def test_constructs_with_arrays(self) -> None:
        m = ModelData(X_obs=np.array([[1.0], [2.0]]), y_obs=np.array([0, 1]))
        assert m.X_obs.shape == (2, 1)
        assert m.y_obs.shape == (2,)


class TestBinomialModelData:
    def test_accepts_valid_input(self) -> None:
        m = BinomialModelData(
            X_obs=np.array([[1.0], [2.0], [3.0]]),
            y_obs=np.array([1, 2, 3]),
            n_trials=10,
        )
        assert m.n_trials == 10
        assert isinstance(m.X_obs, np.ndarray)
        assert isinstance(m.y_obs, np.ndarray)

    def test_converts_lists_to_arrays(self) -> None:
        m = BinomialModelData(X_obs=[[1.0], [2.0]], y_obs=[0, 1], n_trials=5)
        assert isinstance(m.X_obs, np.ndarray)
        assert isinstance(m.y_obs, np.ndarray)

    def test_rejects_1d_X(self) -> None:
        with pytest.raises(ValueError, match=r"X_obs.*2D"):
            BinomialModelData(X_obs=np.array([1.0, 2.0]), y_obs=np.array([0, 1]), n_trials=5)

    def test_rejects_X_with_wrong_columns(self) -> None:
        with pytest.raises(ValueError, match=r"X_obs"):
            BinomialModelData(
                X_obs=np.array([[1.0, 2.0], [3.0, 4.0]]),
                y_obs=np.array([0, 1]),
                n_trials=5,
            )

    def test_rejects_y_with_wrong_length(self) -> None:
        with pytest.raises(ValueError, match=r"y_obs"):
            BinomialModelData(
                X_obs=np.array([[1.0], [2.0], [3.0]]),
                y_obs=np.array([0, 1]),
                n_trials=5,
            )

    def test_rejects_y_below_zero(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, n_trials\]"):
            BinomialModelData(
                X_obs=np.array([[1.0], [2.0]]),
                y_obs=np.array([-1, 0]),
                n_trials=5,
            )

    def test_rejects_y_above_n_trials(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, n_trials\]"):
            BinomialModelData(
                X_obs=np.array([[1.0], [2.0]]),
                y_obs=np.array([0, 6]),
                n_trials=5,
            )
