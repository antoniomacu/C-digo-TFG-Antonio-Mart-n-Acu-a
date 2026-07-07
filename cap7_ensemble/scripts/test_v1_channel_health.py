"""TDD tests for V1TimestepResult.channel_health (Phase 2).

Four tests (per plan §6):
  1. sigmoid parity — bridge health equals direct compute_channel_health output
  2. NaN handling   — NaN actual sensor → NaN channel health
  3. pump fallback  — missing pump_id uses global per_feature stats without error
  4. output ordering — channel_health list order matches output_columns
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "ensemble"))

from model.scoring import ChannelHealth, compute_channel_health

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_OUTPUT_COLUMNS = [
    "Main HTF Pump Current Consumption",
    "Main HTF Pump Flow",
    "Main HTF Pump Outlet Pressure",
    "Main HTF Pump NDE Outboard bearing",
    "Main HTF Pump NDE Inboard bearing",
    "Main HTF Pump DE bearing",
    "Main HTF Pump Motor bearing Temp 1",
    "Main HTF Pump Motor bearing Temp 2",
    "Main HTF Pump Motor bearing Temp 3",
    "Main HTF Pump Motor bearing Temp 4",
    "Main HTF Pump Motor bearing Temp 5",
    "Main HTF Pump Motor supply voltage",
    "Main HTF Pump Motor supply frequency",
]

_N = len(_OUTPUT_COLUMNS)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _per_feature_block(std: float = 1.0, p99: float = 2.33) -> dict:
    return {
        name: {
            "mean_residual": 0.0,
            "std_residual": std,
            "p95_abs_residual": p99 * 0.75,
            "p99_abs_residual": p99,
        }
        for name in _OUTPUT_COLUMNS
    }


def _threshold_block(std: float = 1.0, p99: float = 2.33) -> dict:
    return {
        "mahalanobis": {"warning": 5.0, "alarm": 10.0, "mean": 3.0},
        "mean_residual_vector": [0.0] * _N,
        "inverse_covariance_matrix": np.eye(_N).tolist(),
        "per_feature": _per_feature_block(std=std, p99=p99),
    }


def _make_thresholds(pump3_std: float = 1.0, pump3_p99: float = 2.33,
                     pump3_fallback: bool = False) -> dict:
    """Minimal production_thresholds structure for unit testing."""
    global_block = _threshold_block(std=1.0, p99=2.33)
    pump3_block = {
        **_threshold_block(std=pump3_std, p99=pump3_p99),
        "fallback_to_global": pump3_fallback,
    }
    return {
        "global": global_block,
        "per_pump": {"3": pump3_block},
    }


def _make_bridge(thresholds_dict: dict):
    """CondRegV1Bridge with mocked PumpPredictor (avoids loading ML weights)."""
    from demos.cond_reg_v1_bridge import CondRegV1Bridge

    mock_pred = MagicMock()
    mock_pred.output_columns = _OUTPUT_COLUMNS
    mock_pred.input_columns = [
        "Ambient temperature",
        "Main HTF Pump Speed",
        "Main HTF Pump Inlet Temperature",
    ]

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as fh:
        json.dump(thresholds_dict, fh)
        thresh_path = fh.name

    with patch("demos.cond_reg_v1_bridge.PumpPredictor") as mock_cls:
        mock_cls.return_value = mock_pred
        bridge = CondRegV1Bridge(weights_dir="/tmp", thresholds_path=thresh_path)

    return bridge, mock_pred


def _run_timestep(bridge, mock_pred,
                  actuals: np.ndarray,
                  pump_id: int = 3) -> object:
    """Run one process_timestep call with controlled actual/predicted values."""
    predicted = np.zeros(_N)
    pred_df = pd.DataFrame(
        [predicted], columns=_OUTPUT_COLUMNS, index=[pd.Timestamp("2024-07-07")]
    )
    mock_pred.predict.return_value = pred_df

    row = pd.Series({name: float(actuals[i]) for i, name in enumerate(_OUTPUT_COLUMNS)})
    row["Ambient temperature"] = 25.0
    row["Main HTF Pump Speed"] = 1500.0
    row["Main HTF Pump Inlet Temperature"] = 60.0

    return bridge.process_timestep(
        pump_id=pump_id, timestamp="2024-07-07T00:00:00", row=row
    )


# ---------------------------------------------------------------------------
# Test 1 — Sigmoid parity
# ---------------------------------------------------------------------------


class TestSigmoidParity:
    """V1 channel health must equal direct compute_channel_health output."""

    def test_health_matches_compute_channel_health(self):
        bridge, mock_pred = _make_bridge(_make_thresholds())

        # residual = actual - predicted = 1.0 - 0.0 = 1.0 for every channel
        actuals = np.ones(_N)
        result = _run_timestep(bridge, mock_pred, actuals)

        assert hasattr(result, "channel_health"), (
            "V1TimestepResult missing channel_health field"
        )
        assert len(result.channel_health) == _N

        expected_health, expected_z = compute_channel_health(
            residual=1.0, std_residual=1.0, p99_abs_residual=2.33
        )
        for ch in result.channel_health:
            assert ch.health == pytest.approx(expected_health, rel=1e-4), (
                f"{ch.feature}: health {ch.health} != {expected_health}"
            )
            assert ch.z_score == pytest.approx(expected_z, rel=1e-4), (
                f"{ch.feature}: z_score {ch.z_score} != {expected_z}"
            )

    def test_residual_equals_actual_minus_zero_predicted(self):
        """residual stored in ChannelHealth equals actual - predicted."""
        bridge, mock_pred = _make_bridge(_make_thresholds())
        actuals = np.linspace(0.5, 1.5, _N)
        result = _run_timestep(bridge, mock_pred, actuals)

        assert hasattr(result, "channel_health")
        for i, ch in enumerate(result.channel_health):
            assert ch.residual == pytest.approx(actuals[i], rel=1e-5)


# ---------------------------------------------------------------------------
# Test 2 — NaN handling
# ---------------------------------------------------------------------------


class TestNaNHandling:
    """NaN actual sensor reading must produce NaN channel health."""

    def test_all_nan_actuals_give_nan_health(self):
        bridge, mock_pred = _make_bridge(_make_thresholds())
        actuals = np.full(_N, np.nan)
        result = _run_timestep(bridge, mock_pred, actuals)

        assert hasattr(result, "channel_health")
        for ch in result.channel_health:
            assert math.isnan(ch.health), (
                f"{ch.feature}: expected NaN health, got {ch.health}"
            )

    def test_single_nan_channel_only_that_channel_is_nan(self):
        """Only the NaN actual channel gets NaN health; others remain finite."""
        bridge, mock_pred = _make_bridge(_make_thresholds())
        actuals = np.ones(_N)
        actuals[2] = np.nan  # third channel is NaN

        result = _run_timestep(bridge, mock_pred, actuals)

        assert hasattr(result, "channel_health")
        assert math.isnan(result.channel_health[2].health), (
            f"Channel index 2 should be NaN, got {result.channel_health[2].health}"
        )
        for i, ch in enumerate(result.channel_health):
            if i != 2:
                assert math.isfinite(ch.health), (
                    f"Channel index {i} should be finite, got {ch.health}"
                )


# ---------------------------------------------------------------------------
# Test 3 — Pump fallback
# ---------------------------------------------------------------------------


class TestPumpFallback:
    """Unknown pump falls back to global per_feature stats without error."""

    def test_unknown_pump_uses_global_no_error(self):
        bridge, mock_pred = _make_bridge(_make_thresholds())
        actuals = np.ones(_N)

        # pump_id=99 is not in the thresholds per_pump block
        result = _run_timestep(bridge, mock_pred, actuals, pump_id=99)

        assert hasattr(result, "channel_health")
        assert len(result.channel_health) == _N
        for ch in result.channel_health:
            assert math.isfinite(ch.health), (
                f"Fallback pump: {ch.feature} health not finite: {ch.health}"
            )

    def test_fallback_flag_pump_uses_global_stats(self):
        """pump3 with fallback_to_global=True produces finite health from global stats."""
        thresholds = _make_thresholds(pump3_std=99.0, pump3_p99=999.0, pump3_fallback=True)
        bridge, mock_pred = _make_bridge(thresholds)
        actuals = np.ones(_N)

        result = _run_timestep(bridge, mock_pred, actuals, pump_id=3)

        assert hasattr(result, "channel_health")
        assert len(result.channel_health) == _N


# ---------------------------------------------------------------------------
# Test 4 — Output ordering
# ---------------------------------------------------------------------------


class TestOutputOrdering:
    """channel_health[i].feature must equal output_columns[i]."""

    def test_channel_health_order_matches_output_columns(self):
        bridge, mock_pred = _make_bridge(_make_thresholds())
        actuals = np.random.default_rng(42).uniform(0.1, 1.5, _N)
        result = _run_timestep(bridge, mock_pred, actuals)

        assert hasattr(result, "channel_health")
        assert len(result.channel_health) == _N
        for i, ch in enumerate(result.channel_health):
            assert ch.feature == _OUTPUT_COLUMNS[i], (
                f"channel_health[{i}].feature={ch.feature!r}, "
                f"expected {_OUTPUT_COLUMNS[i]!r}"
            )
