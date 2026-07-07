"""Tests for heatmap helper functions in comparison_plotting_demo.py.

Run from ensemble/:
    uv run python -m pytest scripts/test_plotting_heatmap.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from model.scoring import ChannelHealth
from demos.comparison_plotting_demo import (
    _build_heatmap_array,
    _health_row_from_channel_list,
    SENSOR_COLUMNS,
)


def _make_ch(feature: str, health: float) -> ChannelHealth:
    return ChannelHealth(feature=feature, health=health, z_score=0.0, residual=0.0)


CHAN_A = "Main HTF Pump Current Consumption"
CHAN_B = "Main HTF Pump Flow"
CHAN_C = "Main HTF Pump Outlet Pressure"


class TestHealthRowFromChannelList:
    def test_values_in_order(self) -> None:
        channel_order = [CHAN_A, CHAN_B, CHAN_C]
        ch_health = [_make_ch(CHAN_A, 90.0), _make_ch(CHAN_B, 45.0), _make_ch(CHAN_C, 10.0)]
        row = _health_row_from_channel_list(ch_health, channel_order)
        assert row.shape == (3,)
        np.testing.assert_allclose(row, [90.0, 45.0, 10.0])

    def test_missing_channel_is_nan(self) -> None:
        channel_order = [CHAN_A, CHAN_B, CHAN_C]
        ch_health = [_make_ch(CHAN_A, 80.0)]  # CHAN_B and CHAN_C missing
        row = _health_row_from_channel_list(ch_health, channel_order)
        assert np.isfinite(row[0])
        assert np.isnan(row[1])
        assert np.isnan(row[2])

    def test_nan_health_value_preserved(self) -> None:
        channel_order = [CHAN_A]
        ch_health = [_make_ch(CHAN_A, float("nan"))]
        row = _health_row_from_channel_list(ch_health, channel_order)
        assert np.isnan(row[0])

    def test_empty_channel_health_all_nan(self) -> None:
        channel_order = [CHAN_A, CHAN_B]
        row = _health_row_from_channel_list([], channel_order)
        assert row.shape == (2,)
        assert np.all(np.isnan(row))

    def test_order_follows_channel_order_not_input_order(self) -> None:
        channel_order = [CHAN_C, CHAN_A, CHAN_B]
        ch_health = [_make_ch(CHAN_A, 10.0), _make_ch(CHAN_B, 20.0), _make_ch(CHAN_C, 30.0)]
        row = _health_row_from_channel_list(ch_health, channel_order)
        np.testing.assert_allclose(row, [30.0, 10.0, 20.0])


class TestBuildHeatmapArray:
    def test_shape_is_2_by_n_channels(self) -> None:
        channel_order = [CHAN_A, CHAN_B, CHAN_C]
        v1 = [_make_ch(CHAN_A, 10.0), _make_ch(CHAN_B, 20.0), _make_ch(CHAN_C, 30.0)]
        ens = [_make_ch(CHAN_A, 80.0), _make_ch(CHAN_B, 90.0), _make_ch(CHAN_C, 70.0)]
        arr = _build_heatmap_array(v1, ens, channel_order)
        assert arr.shape == (2, 3)

    def test_row0_is_v1_row1_is_ens(self) -> None:
        channel_order = [CHAN_A, CHAN_B]
        v1 = [_make_ch(CHAN_A, 5.0), _make_ch(CHAN_B, 10.0)]
        ens = [_make_ch(CHAN_A, 85.0), _make_ch(CHAN_B, 90.0)]
        arr = _build_heatmap_array(v1, ens, channel_order)
        np.testing.assert_allclose(arr[0], [5.0, 10.0])
        np.testing.assert_allclose(arr[1], [85.0, 90.0])

    def test_nan_propagated_for_missing_channels(self) -> None:
        channel_order = [CHAN_A, CHAN_B, CHAN_C]
        v1 = [_make_ch(CHAN_A, 5.0)]
        ens = [_make_ch(CHAN_A, 85.0)]
        arr = _build_heatmap_array(v1, ens, channel_order)
        assert np.isfinite(arr[0, 0])
        assert np.isnan(arr[0, 1])
        assert np.isfinite(arr[1, 0])
        assert np.isnan(arr[1, 1])


class TestSensorColumns:
    def test_has_13_channels(self) -> None:
        assert len(SENSOR_COLUMNS) == 13

    def test_all_from_short_names(self) -> None:
        from demos.streaming_demo import SHORT_NAMES
        assert set(SENSOR_COLUMNS) == set(SHORT_NAMES.keys())
