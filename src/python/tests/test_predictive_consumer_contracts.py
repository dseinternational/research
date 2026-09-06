# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small downstream output contracts captured independently of these adapters.

The fixture records source commits, file hashes and complete golden tables from
actual downstream functions. Tests use only synthetic inputs and local files;
no downstream checkout, fitted model or network access is required.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dse_research_utils.statistics.predictive import predictive_observation_checks
from dse_research_utils.statistics.samples import sample_matrix


@pytest.fixture(scope="module")
def consumer_contract():
    return json.loads((Path(__file__).parent / "fixtures" / "predictive_consumer_contracts.json").read_text())


def _golden_table(contract, name):
    table = contract[name]
    return pd.DataFrame(table["rows"], columns=table["columns"]).astype(table["dtypes"])


def _labelled_inputs(contract, observed_key):
    inputs = contract["inputs"]
    # Neither sample nor observation labels are sorted. Their declared order is
    # part of the output contract, independent of values or a missing-row mask.
    replicated = xr.DataArray(
        np.asarray(inputs["replicated"], dtype=inputs["draw_dtype"]).T[None, :, :],
        dims=("chain", "draw", "obs_id"),
        coords={"chain": [7], "draw": inputs["draw_ids"], "obs_id": inputs["observation_ids"]},
    )
    observed = xr.DataArray(
        np.asarray(inputs[observed_key], dtype=float),
        dims="obs_id",
        coords={"obs_id": inputs["observation_ids"]},
    )
    matrix = sample_matrix(replicated, sample_dims=("chain", "draw"), observation_dims=("obs_id",))
    return matrix, matrix.observed_values(observed)


def _vocabulary_table(observed, draws, checks, age_bands):
    """Keep consumer-owned grouping, display schema and zero-count summaries."""
    labels = np.asarray(age_bands)
    zero_rate = np.mean(draws == 0, axis=1)
    # The existing consumer subtracts native float32 bounds before storing each
    # width as float64. Casting bounds first would change the golden table.
    widths = (checks.upper - checks.lower).astype(float)
    rows = []
    for label in ("all", *dict.fromkeys(age_bands)):
        keep = np.ones(observed.size, dtype=bool) if label == "all" else labels == label
        y = observed[keep]
        pit = checks.midpoint_pit[keep]
        shared = {
            "age_band_months": label,
            "n_observations": int(keep.sum()),
            "observed_mean": float(y.mean()),
            "predictive_mean": float(checks.predictive_mean[keep].mean()),
            # Preserve difference-of-means arithmetic, rather than averaging
            # observation-level errors before the group reduction.
            "mean_error": float(checks.predictive_mean[keep].mean() - y.mean()),
            "observed_zero_rate": float(np.mean(y == 0)),
            "predictive_zero_rate": float(zero_rate[keep].mean()),
            "mid_pit_mean": float(pit.mean()),
            "mid_pit_variance": float(pit.var()),
            "expected_mid_pit_variance": float(checks.expected_midpoint_pit_variance[keep].mean()),
            "mid_pit_extreme_rate": float(np.mean((pit < 0.05) | (pit > 0.95))),
        }
        for column, prob in enumerate(checks.interval_probs):
            rows.append(
                {
                    **shared,
                    "interval_probability": prob,
                    "empirical_coverage": float(checks.inside[keep, column].mean()),
                    "expected_coverage": float(checks.predictive_mass[keep, column].mean()),
                    "mean_interval_width": float(widths[keep, column].mean()),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_vocabulary_age_band_and_overall_tables_match_downstream_golden(consumer_contract, chunk_size):
    matrix, observed = _labelled_inputs(consumer_contract, "vg_observed")
    checks = predictive_observation_checks(
        observed,
        matrix.values,
        interval_probs=consumer_contract["inputs"]["interval_probs"],
        pit_method="midpoint",
        observation_chunk_size=chunk_size,
    )
    actual = _vocabulary_table(observed, matrix.values, checks, consumer_contract["inputs"]["age_bands"])
    pd.testing.assert_frame_equal(actual, _golden_table(consumer_contract, "vg_calibration"), check_exact=True)


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_reading_calibration_and_coverage_keep_missing_row_contract(consumer_contract, chunk_size):
    matrix, observed = _labelled_inputs(consumer_contract, "lrp_observed")
    # LRP owns this filter. Apply it to identities, observations and draws once.
    keep = np.isfinite(observed)
    assert matrix.observation_index[keep].tolist() == ["row-3", "row-4", "row-2"]
    observed = observed[keep]
    checks = predictive_observation_checks(
        observed,
        matrix.values[keep],
        interval_probs=consumer_contract["inputs"]["interval_probs"],
        observation_chunk_size=chunk_size,
    )
    column = checks.interval_probs.index(0.89)
    calibration = pd.DataFrame(
        {
            "observed": observed,
            "pp_median": checks.predictive_median,
            "pp_lo": checks.lower[:, column],
            "pp_hi": checks.upper[:, column],
            "inside": checks.inside[:, column],
        }
    )
    rows = []
    for column, prob in enumerate(checks.interval_probs):
        n_inside = int(np.count_nonzero(checks.inside[:, column]))
        rows.append(
            {
                "mode": "count_interval",
                "node": "y_post",
                "unit": "observations",
                "quantity": "observed score",
                "level": prob,
                "level_pct": round(prob * 100),
                "n_total": len(observed),
                "n_inside": n_inside,
                "coverage": n_inside / len(observed),
            }
        )
    pd.testing.assert_frame_equal(calibration, _golden_table(consumer_contract, "lrp_calibration"), check_exact=True)
    pd.testing.assert_frame_equal(
        pd.DataFrame(rows), _golden_table(consumer_contract, "lrp_coverage"), check_exact=True
    )
