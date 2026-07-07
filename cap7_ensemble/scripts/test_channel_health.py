"""Tests for per-variable health indicators."""

from __future__ import annotations

import math
import pytest

from model.scoring import compute_channel_health, ChannelHealth, ChannelHealthSummary


class TestComputeChannelHealth:
    """Unit tests for compute_channel_health."""

    def test_zero_residual_gives_high_health(self):
        """Residual = 0 means perfect prediction → health near 95."""
        health, z = compute_channel_health(
            residual=0.0,
            std_residual=1.0,
            p99_abs_residual=2.33,
        )
        assert z == pytest.approx(0.0)
        assert health > 90.0

    def test_p99_residual_gives_mid_health(self):
        """Residual at the channel's own 99th-percentile deviation → health ~50.

        The band is anchored to the channel's normal envelope: reaching the
        deviation it only exceeds 1% of the time during normal operation is
        the edge of normal, hence borderline health, not a full alarm.
        """
        health, z = compute_channel_health(
            residual=2.33,
            std_residual=1.0,
            p99_abs_residual=2.33,
        )
        assert z == pytest.approx(2.33, rel=1e-3)
        assert 40.0 < health < 60.0

    def test_double_p99_residual_gives_low_health(self):
        """Residual at twice the 99th-percentile deviation → health near 0 (alarm)."""
        health, z = compute_channel_health(
            residual=4.66,
            std_residual=1.0,
            p99_abs_residual=2.33,
        )
        assert health < 10.0

    def test_residual_centered_on_mean_gives_high_health(self):
        """A residual equal to the calibrated mean residual is zero deviation → green.

        The detector's Mahalanobis path subtracts the mean residual vector, so a
        channel's known systematic bias must not count as ill-health here either.
        """
        health, z = compute_channel_health(
            residual=1.4,
            std_residual=3.15,
            p99_abs_residual=8.93,
            mean_residual=1.4,
        )
        assert z == pytest.approx(0.0, abs=1e-9)
        assert health > 90.0

    def test_steady_mild_offset_reads_green(self):
        """The DE Bearing screenshot case: steady ~1.5-sigma under-prediction reads green.

        Numbers from pump_2_2025-12-10: residual ~4.8, std 3.15, p99 8.93,
        calibrated mean_residual 0.61. The covariance-whitened detector treats
        this as NORMAL; the per-channel diagnostic must agree.
        """
        health, z = compute_channel_health(
            residual=4.8,
            std_residual=3.15,
            p99_abs_residual=8.93,
            mean_residual=0.61,
        )
        assert health >= 70.0

    def test_mean_residual_defaults_to_zero(self):
        """Omitting mean_residual reproduces uncentered behaviour (backward compatible)."""
        with_default = compute_channel_health(2.0, 1.0, 2.33)
        explicit_zero = compute_channel_health(2.0, 1.0, 2.33, mean_residual=0.0)
        assert with_default == explicit_zero

    def test_negative_residual_same_as_positive(self):
        """Health is symmetric — sign of residual doesn't matter."""
        h_pos, z_pos = compute_channel_health(1.5, 1.0, 2.33)
        h_neg, z_neg = compute_channel_health(-1.5, 1.0, 2.33)
        assert h_pos == pytest.approx(h_neg)
        assert z_pos == pytest.approx(z_neg)

    def test_very_large_residual_clamps_to_zero(self):
        """Residual far beyond alarm → health near 0, not negative."""
        health, z = compute_channel_health(
            residual=100.0,
            std_residual=1.0,
            p99_abs_residual=2.33,
        )
        assert health >= 0.0
        assert health < 1.0

    def test_near_zero_std_no_crash(self):
        """std_residual ≈ 0 must not cause division by zero."""
        health, z = compute_channel_health(
            residual=0.5,
            std_residual=0.0,
            p99_abs_residual=0.0,
        )
        assert math.isfinite(health)
        assert math.isfinite(z)
        assert health >= 0.0

    def test_return_types(self):
        """Returns (float, float)."""
        health, z = compute_channel_health(0.5, 1.0, 2.33)
        assert isinstance(health, float)
        assert isinstance(z, float)


class TestChannelHealthDataclass:
    """Verify ChannelHealth dataclass basics."""

    def test_construction(self):
        ch = ChannelHealth(
            feature="Main HTF Pump Flow",
            health=85.0,
            z_score=0.5,
            residual=-2.3,
        )
        assert ch.feature == "Main HTF Pump Flow"
        assert ch.health == 85.0
        assert ch.z_score == 0.5
        assert ch.residual == -2.3


class TestChannelHealthSummaryDataclass:
    """Verify ChannelHealthSummary dataclass basics."""

    def test_construction(self):
        chs = ChannelHealthSummary(
            feature="Main HTF Pump Current Consumption",
            mean_health=72.5,
            min_health=15.0,
            mean_z_score=1.1,
            max_z_score=3.8,
            n_timesteps_below_50=12,
        )
        assert chs.feature == "Main HTF Pump Current Consumption"
        assert chs.min_health == 15.0
        assert chs.n_timesteps_below_50 == 12


class TestStreamingChannelHealth:
    """Verify channel_health is populated in StreamingTimestepResult."""

    def test_streaming_result_has_channel_health_field(self):
        from model.streaming import StreamingTimestepResult

        result = StreamingTimestepResult(
            timestep=0,
            timestamp="2025-10-06T00:00:00",
            pump_id=1,
            l1_mahalanobis=3.0,
            l1_status="NORMAL",
            l1_health=90.0,
            l1_z_scores={},
            l1_actual={},
            l1_predicted={},
            l1_residuals={},
            l1_anomalous_features=[],
        )
        assert hasattr(result, "channel_health")
        assert result.channel_health == []


class TestBatchChannelHealthAggregation:
    """Verify ChannelHealthSummary aggregation logic."""

    def test_aggregate_single_timestep(self):
        from model.scoring import ChannelHealth
        from model.level1_detector import _aggregate_channel_health

        per_timestep = {
            "Feature_A": [ChannelHealth("Feature_A", 80.0, 0.5, 0.3)],
        }
        summaries = _aggregate_channel_health(per_timestep)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.feature == "Feature_A"
        assert s.mean_health == pytest.approx(80.0)
        assert s.min_health == pytest.approx(80.0)
        assert s.mean_z_score == pytest.approx(0.5)
        assert s.max_z_score == pytest.approx(0.5)
        assert s.n_timesteps_below_50 == 0

    def test_aggregate_multiple_timesteps(self):
        from model.scoring import ChannelHealth
        from model.level1_detector import _aggregate_channel_health

        per_timestep = {
            "Feature_A": [
                ChannelHealth("Feature_A", 80.0, 0.5, 0.3),
                ChannelHealth("Feature_A", 30.0, 2.0, 1.5),
                ChannelHealth("Feature_A", 60.0, 1.0, 0.8),
            ],
        }
        summaries = _aggregate_channel_health(per_timestep)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.mean_health == pytest.approx((80.0 + 30.0 + 60.0) / 3.0)
        assert s.min_health == pytest.approx(30.0)
        assert s.mean_z_score == pytest.approx((0.5 + 2.0 + 1.0) / 3.0)
        assert s.max_z_score == pytest.approx(2.0)
        assert s.n_timesteps_below_50 == 1  # only the 30.0 one

    def test_aggregate_empty(self):
        from model.level1_detector import _aggregate_channel_health

        summaries = _aggregate_channel_health({})
        assert summaries == []


class TestEnsemblePropagation:
    """Verify channel_health_summary propagates to PumpEnsembleResult."""

    def test_pump_ensemble_result_has_channel_health_summary(self):
        from model.scoring import (
            Level1Result,
            PumpEnsembleResult,
            ChannelHealthSummary,
        )

        summary = ChannelHealthSummary(
            feature="Flow",
            mean_health=70.0,
            min_health=30.0,
            mean_z_score=1.2,
            max_z_score=2.8,
            n_timesteps_below_50=5,
        )
        l1 = Level1Result(
            pump_id=1,
            date="2025-10-06",
            status="NORMAL",
            day_mean_mahalanobis=3.0,
            day_max_mahalanobis=5.0,
            fraction_above_warning=0.1,
            fraction_above_alarm=0.0,
            normalized_severity=0.5,
            n_timesteps=100,
            channel_health_summary=[summary],
        )
        pr = PumpEnsembleResult(
            pump_id=1,
            date="2025-10-06",
            overall_status="NORMAL",
            overall_severity=0.5,
            level1=l1,
            level2=None,
            ensemble_reasoning="test",
            channel_health_summary=l1.channel_health_summary,
        )
        assert len(pr.channel_health_summary) == 1
        assert pr.channel_health_summary[0].feature == "Flow"


class TestJsonSerialization:
    """Verify channel health appears in EnsembleReport.to_dict() output."""

    def test_to_dict_includes_channel_health_summary(self):
        from model.scoring import (
            EnsembleReport,
            PumpEnsembleResult,
            Level1Result,
            ChannelHealthSummary,
        )

        summary = ChannelHealthSummary(
            feature="Flow",
            mean_health=70.0,
            min_health=30.0,
            mean_z_score=1.2,
            max_z_score=2.8,
            n_timesteps_below_50=5,
        )
        l1 = Level1Result(
            pump_id=1,
            date="2025-10-06",
            status="NORMAL",
            day_mean_mahalanobis=3.0,
            day_max_mahalanobis=5.0,
            fraction_above_warning=0.1,
            fraction_above_alarm=0.0,
            normalized_severity=0.5,
            n_timesteps=100,
            channel_health_summary=[summary],
        )
        pr = PumpEnsembleResult(
            pump_id=1,
            date="2025-10-06",
            overall_status="NORMAL",
            overall_severity=0.5,
            level1=l1,
            level2=None,
            ensemble_reasoning="test",
            channel_health_summary=[summary],
        )
        report = EnsembleReport(
            overall_status="NORMAL",
            overall_severity=0.5,
            pump_results=[pr],
        )
        d = report.to_dict()
        pr_dict = d["pump_results"][0]
        assert "channel_health_summary" in pr_dict
        assert len(pr_dict["channel_health_summary"]) == 1
        assert pr_dict["channel_health_summary"][0]["feature"] == "Flow"
        assert pr_dict["channel_health_summary"][0]["min_health"] == 30.0
