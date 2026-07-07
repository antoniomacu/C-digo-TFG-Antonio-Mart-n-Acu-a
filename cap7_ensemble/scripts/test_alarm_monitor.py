"""Unit tests for AlarmMonitor (spike filter + rate limiter)."""
import pytest
from datetime import datetime, timezone

from model.monitoring import AlarmMonitor


def _ts(offset_seconds: float = 0.0) -> str:
    """Return an ISO timestamp starting from a fixed epoch plus offset."""
    base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return (base.replace(second=0) if offset_seconds == 0 else
            datetime.fromtimestamp(base.timestamp() + offset_seconds, tz=timezone.utc)).isoformat()


class TestSpikeFilter:
    def test_single_warning_suppressed(self):
        mon = AlarmMonitor(min_alarm_duration=2)
        status, fired = mon.check("WARNING", _ts(0))
        assert status == "NORMAL"
        assert not fired

    def test_single_alarm_suppressed(self):
        mon = AlarmMonitor(min_alarm_duration=2)
        status, fired = mon.check("ALARM", _ts(0))
        assert status == "NORMAL"
        assert not fired

    def test_two_consecutive_warnings_pass(self):
        mon = AlarmMonitor(min_alarm_duration=2)
        mon.check("WARNING", _ts(0))
        status, _ = mon.check("WARNING", _ts(1))
        assert status == "WARNING"

    def test_two_consecutive_alarms_pass(self):
        mon = AlarmMonitor(min_alarm_duration=2)
        mon.check("ALARM", _ts(0))
        status, _ = mon.check("ALARM", _ts(1))
        assert status == "ALARM"

    def test_normal_resets_counter(self):
        mon = AlarmMonitor(min_alarm_duration=2)
        mon.check("WARNING", _ts(0))
        mon.check("NORMAL", _ts(1))
        # counter reset — next WARNING is alone again
        status, _ = mon.check("WARNING", _ts(2))
        assert status == "NORMAL"

    def test_normal_passthrough(self):
        mon = AlarmMonitor(min_alarm_duration=2)
        status, fired = mon.check("NORMAL", _ts(0))
        assert status == "NORMAL"
        assert not fired

    def test_min_duration_one_allows_single_window(self):
        mon = AlarmMonitor(min_alarm_duration=1)
        status, _ = mon.check("ALARM", _ts(0))
        assert status == "ALARM"

    def test_three_consecutive_with_min_two(self):
        mon = AlarmMonitor(min_alarm_duration=2)
        mon.check("ALARM", _ts(0))
        s1, _ = mon.check("ALARM", _ts(1))
        s2, _ = mon.check("ALARM", _ts(2))
        assert s1 == "ALARM"
        assert s2 == "ALARM"


class TestRateLimiter:
    def test_first_alarm_fires(self):
        mon = AlarmMonitor(min_alarm_duration=1, rate_limit_seconds=3600.0)
        _, fired = mon.check("ALARM", _ts(0))
        assert fired

    def test_second_alarm_within_limit_does_not_fire(self):
        mon = AlarmMonitor(min_alarm_duration=1, rate_limit_seconds=3600.0)
        mon.check("ALARM", _ts(0))
        _, fired = mon.check("ALARM", _ts(10))
        assert not fired

    def test_alarm_fires_after_rate_limit_elapsed(self):
        mon = AlarmMonitor(min_alarm_duration=1, rate_limit_seconds=3600.0)
        mon.check("ALARM", _ts(0))
        # 3600 seconds later — exactly at boundary
        _, fired = mon.check("ALARM", _ts(3600))
        assert fired

    def test_warning_does_not_fire_alert(self):
        mon = AlarmMonitor(min_alarm_duration=1, rate_limit_seconds=3600.0)
        _, fired = mon.check("WARNING", _ts(0))
        assert not fired

    def test_alert_fires_again_after_normal_gap(self):
        mon = AlarmMonitor(min_alarm_duration=1, rate_limit_seconds=3600.0)
        mon.check("ALARM", _ts(0))
        mon.check("NORMAL", _ts(10))
        # Normal resets in_sustained_alarm; next alarm should re-fire
        _, fired = mon.check("ALARM", _ts(20))
        assert fired


class TestReset:
    def test_reset_clears_counter(self):
        mon = AlarmMonitor(min_alarm_duration=3)
        mon.check("ALARM", _ts(0))
        mon.check("ALARM", _ts(1))
        mon.reset()
        status, _ = mon.check("ALARM", _ts(2))
        assert status == "NORMAL"

    def test_reset_clears_rate_limit(self):
        mon = AlarmMonitor(min_alarm_duration=1, rate_limit_seconds=3600.0)
        mon.check("ALARM", _ts(0))
        mon.reset()
        _, fired = mon.check("ALARM", _ts(10))
        assert fired
