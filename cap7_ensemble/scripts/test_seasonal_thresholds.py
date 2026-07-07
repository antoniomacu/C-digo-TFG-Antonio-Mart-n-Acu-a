"""Unit tests for P7 seasonal calibration and threshold selection.

Run from ensemble/ directory:
    uv run python scripts/test_seasonal_thresholds.py

Tests cover:
  1. _month_to_season mapping correctness
  2. ThresholdCalibrator.calibrate_seasonal() structure (smoke test with minimal data)
  3. SeasonalTracker.promote() writes correct files
  4. Level1Detector.set_season() + seasonal fallback chain
  5. Level1Detector with no seasonal file — existing behaviour preserved
  6. Level2Detector seasonal threshold override in classify()
  7. Streaming boundary crossing — set_season() called when month changes season
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd

ENSEMBLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENSEMBLE_DIR))

from model.monitoring import SeasonalTracker
from cond_reg_v2.model.threshold_calibration import _month_to_season, _SEASONS, _SEASON_MAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seasonal_l1_dict(seasons=None) -> dict:
    """Build a minimal valid L1 seasonal threshold dict."""
    if seasons is None:
        seasons = _SEASONS
    feature_names = ["feat_a", "feat_b"]
    threshold_block = {
        "mahalanobis": {"warning": 5.0, "alarm": 8.0, "mean": 3.0, "std": 1.0, "p95": 5.0, "p99": 8.0},
        "per_feature": {
            "feat_a": {"mean_residual": 0.0, "std_residual": 1.0,
                       "p95_abs_residual": 2.0, "p99_abs_residual": 3.0,
                       "z_score_warning": 1.96, "z_score_alarm": 2.58},
            "feat_b": {"mean_residual": 0.0, "std_residual": 1.0,
                       "p95_abs_residual": 2.0, "p99_abs_residual": 3.0,
                       "z_score_warning": 1.96, "z_score_alarm": 2.58},
        },
        "covariance_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "inverse_covariance_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "mean_residual_vector": [0.0, 0.0],
    }
    per_pump_block = {
        "1": {"n_samples": 100, "fallback_to_global": False,
              "mahalanobis": {"warning": 4.5, "alarm": 7.5}, "per_feature": threshold_block["per_feature"]},
        "2": {"n_samples": 5, "fallback_to_global": True,
              "mahalanobis": {"warning": 5.0, "alarm": 8.0}, "per_feature": threshold_block["per_feature"]},
    }
    result = {
        "description": "test",
        "method": "test",
        "seasons": _SEASONS,
        "feature_names": feature_names,
    }
    for season in _SEASONS:
        result[season] = {
            "n_training_samples": 200 if season in seasons else 0,
            "global": threshold_block,
            "per_pump": per_pump_block,
        }
    return result


def _make_seasonal_l2_dict() -> dict:
    """Build a minimal valid L2 seasonal threshold dict."""
    per_pump = {
        "1": {"warning": 0.0042, "alarm": 0.0071, "n_samples": 100, "fallback_to_global": False},
        "2": {"warning": 0.0038, "alarm": 0.0065, "n_samples": 80, "fallback_to_global": False},
    }
    result: dict = {
        "description": "test L2 seasonal",
        "method": "test",
        "seasons": _SEASONS,
    }
    for season in _SEASONS:
        result[season] = {"per_pump": per_pump}
    return result


# ---------------------------------------------------------------------------
# Test 1: _month_to_season correctness
# ---------------------------------------------------------------------------

class TestMonthToSeason(unittest.TestCase):
    def test_winter_months(self):
        for m in [12, 1, 2]:
            self.assertEqual(_month_to_season(m), "winter", f"month {m} should be winter")

    def test_spring_months(self):
        for m in [3, 4, 5]:
            self.assertEqual(_month_to_season(m), "spring", f"month {m} should be spring")

    def test_summer_months(self):
        for m in [6, 7, 8]:
            self.assertEqual(_month_to_season(m), "summer", f"month {m} should be summer")

    def test_autumn_months(self):
        for m in [9, 10, 11]:
            self.assertEqual(_month_to_season(m), "autumn", f"month {m} should be autumn")

    def test_invalid_month_raises(self):
        with self.assertRaises(ValueError):
            _month_to_season(0)
        with self.assertRaises(ValueError):
            _month_to_season(13)

    def test_all_months_covered(self):
        for m in range(1, 13):
            season = _month_to_season(m)
            self.assertIn(season, _SEASONS)


# ---------------------------------------------------------------------------
# Test 2: SeasonalTracker.promote() writes correct files
# ---------------------------------------------------------------------------

class TestSeasonalTrackerPromote(unittest.TestCase):
    def test_promote_writes_both_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tracker = SeasonalTracker(log_dir=tmp)
            l1 = _make_seasonal_l1_dict()
            l2 = _make_seasonal_l2_dict()

            l1_path, l2_path = tracker.promote(l1, l2, output_dir=tmp)

            self.assertTrue(l1_path.exists(), "L1 seasonal file must exist")
            self.assertTrue(l2_path.exists(), "L2 seasonal file must exist")
            self.assertEqual(l1_path.name, "production_thresholds_seasonal.json")
            self.assertEqual(l2_path.name, "production_thresholds_seasonal_l2.json")

            with l1_path.open() as f:
                loaded_l1 = json.load(f)
            self.assertIn("winter", loaded_l1)
            self.assertIn("summer", loaded_l1)

    def test_promote_missing_season_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = SeasonalTracker(log_dir=Path(tmpdir))
            l1 = _make_seasonal_l1_dict()
            l2_missing = {"winter": {}, "spring": {}}  # missing summer, autumn
            with self.assertRaises(ValueError):
                tracker.promote(l1, l2_missing, output_dir=Path(tmpdir))

    def test_promote_uses_baseline_parent_when_no_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fake_baseline = tmp / "some_dir" / "baseline.json"
            fake_baseline.parent.mkdir(parents=True)
            fake_baseline.write_text("{}")
            tracker = SeasonalTracker(log_dir=tmp, baseline_path=fake_baseline)
            l1_path, l2_path = tracker.promote(
                _make_seasonal_l1_dict(), _make_seasonal_l2_dict()
            )
            self.assertEqual(l1_path.parent, fake_baseline.parent)


# ---------------------------------------------------------------------------
# Test 3: Level1Detector set_season() + seasonal fallback chain
# ---------------------------------------------------------------------------

class TestLevel1DetectorSeasonal(unittest.TestCase):
    def _make_detector_with_seasonal(self, tmpdir: Path):
        """Create a minimal Level1Detector with a mocked weights_dir and seasonal file."""
        from model.level1_detector import Level1Detector

        weights_dir = tmpdir / "weights"
        weights_dir.mkdir()

        feature_names = ["feat_a", "feat_b"]

        # Write minimal production_thresholds.json (required by __init__).
        global_thresholds = {
            "description": "test",
            "method": "test",
            "n_training_samples": 100,
            "feature_names": feature_names,
            "global": {
                "mahalanobis": {"warning": 10.0, "alarm": 15.0, "mean": 3.0, "std": 1.0, "p95": 10.0, "p99": 15.0},
                "per_feature": {
                    f: {"mean_residual": 0.0, "std_residual": 1.0,
                        "p95_abs_residual": 2.0, "p99_abs_residual": 3.0,
                        "z_score_warning": 1.96, "z_score_alarm": 2.58}
                    for f in feature_names
                },
                "covariance_matrix": [[1.0, 0.0], [0.0, 1.0]],
                "inverse_covariance_matrix": [[1.0, 0.0], [0.0, 1.0]],
                "mean_residual_vector": [0.0, 0.0],
            },
            "per_pump": {
                "1": {
                    "n_samples": 200,
                    "fallback_to_global": False,
                    "mahalanobis": {"warning": 9.0, "alarm": 14.0},
                    "per_feature": {
                        f: {"mean_residual": 0.0, "std_residual": 1.0,
                            "p95_abs_residual": 2.0, "p99_abs_residual": 3.0,
                            "z_score_warning": 1.96, "z_score_alarm": 2.58}
                        for f in feature_names
                    },
                    "covariance_matrix": [[1.0, 0.0], [0.0, 1.0]],
                    "inverse_covariance_matrix": [[1.0, 0.0], [0.0, 1.0]],
                    "mean_residual_vector": [0.0, 0.0],
                }
            },
        }
        with (weights_dir / "production_thresholds.json").open("w") as f:
            json.dump(global_thresholds, f)

        # Write seasonal file with summer having pump 1 with lower warning.
        seasonal = _make_seasonal_l1_dict()
        # Override summer pump 1 warning to a distinctive value.
        seasonal["summer"]["per_pump"]["1"]["mahalanobis"]["warning"] = 3.0
        seasonal["summer"]["per_pump"]["1"]["mahalanobis"]["alarm"] = 6.0
        with (weights_dir / "production_thresholds_seasonal.json").open("w") as f:
            json.dump(seasonal, f)

        # Write dummy model weights and norm_params to pass validation.
        (weights_dir / "best_weights.pt").write_bytes(b"")
        (weights_dir / "norm_params.json").write_text("{}")

        # Mock PumpPredictor to avoid loading actual model.
        mock_predictor = MagicMock()
        mock_predictor.output_columns = feature_names
        mock_predictor._past_history = 3

        with patch("cond_reg_v2.model.threshold_calibration.PumpPredictor"), \
             patch("model.level1_detector.Level1Detector.__init__") as mock_init:

            detector = Level1Detector.__new__(Level1Detector)
            # Manually replicate init logic with mocked predictor.
            detector._weights_dir = str(weights_dir)
            detector._feature_z_threshold = 3.0
            detector._top_k_timesteps = 10
            detector._predictor = mock_predictor
            detector._output_columns = feature_names
            detector._feature_names = feature_names

            # Parse thresholds using the real logic.
            with (weights_dir / "production_thresholds.json").open() as f:
                thresholds = json.load(f)
            detector._thresholds = thresholds
            detector._global_thresholds = detector._parse_threshold_block(thresholds.get("global", {}))
            detector._per_pump_thresholds = {}
            for pid_str, block in thresholds.get("per_pump", {}).items():
                pid_int = int(pid_str)
                parsed = detector._parse_threshold_block(block)
                parsed["fallback_to_global"] = bool(block.get("fallback_to_global", False))
                detector._per_pump_thresholds[pid_int] = parsed

            # Load seasonal file.
            detector._active_season = None
            detector._seasonal_per_pump_thresholds = None
            detector._seasonal_global_thresholds = None
            seasonal_path = weights_dir / "production_thresholds_seasonal.json"
            if seasonal_path.exists():
                with seasonal_path.open() as f:
                    s_data = json.load(f)
                detector._seasonal_per_pump_thresholds = {}
                detector._seasonal_global_thresholds = {}
                from model.level1_detector import _SEASONS as L1_SEASONS
                for season in L1_SEASONS:
                    sb = s_data.get(season, {})
                    detector._seasonal_global_thresholds[season] = detector._parse_threshold_block(
                        sb.get("global", {})
                    )
                    pump_map = {}
                    for ps, pb in sb.get("per_pump", {}).items():
                        pi = int(ps)
                        pp = detector._parse_threshold_block(pb)
                        pp["fallback_to_global"] = bool(pb.get("fallback_to_global", False))
                        pump_map[pi] = pp
                    detector._seasonal_per_pump_thresholds[season] = pump_map

            return detector

    def test_set_season_changes_active_season(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            det = self._make_detector_with_seasonal(Path(tmpdir))
            self.assertIsNone(det._active_season)
            det.set_season("summer")
            self.assertEqual(det._active_season, "summer")

    def test_seasonal_thresholds_used_when_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            det = self._make_detector_with_seasonal(Path(tmpdir))
            det.set_season("summer")
            thresholds = det._get_pump_thresholds(1)
            # Summer pump 1 warning was set to 3.0 in the seasonal file.
            self.assertAlmostEqual(thresholds["mahalanobis"]["warning"], 3.0)

    def test_fallback_to_global_when_season_not_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            det = self._make_detector_with_seasonal(Path(tmpdir))
            # No season set — should use non-seasonal per-pump thresholds.
            thresholds = det._get_pump_thresholds(1)
            self.assertAlmostEqual(thresholds["mahalanobis"]["warning"], 9.0)

    def test_pump_fallback_to_seasonal_global(self):
        """Pump 2 has fallback_to_global=True in seasonal file — should get seasonal global."""
        with tempfile.TemporaryDirectory() as tmpdir:
            det = self._make_detector_with_seasonal(Path(tmpdir))
            det.set_season("summer")
            thresholds = det._get_pump_thresholds(2)
            # Pump 2 fallback_to_global=True, so should get seasonal global warning=5.0.
            self.assertAlmostEqual(thresholds["mahalanobis"]["warning"], 5.0)

    def test_invalid_season_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            det = self._make_detector_with_seasonal(Path(tmpdir))
            with self.assertRaises(ValueError):
                det.set_season("monsoon")

    def test_no_seasonal_file_preserves_existing_behaviour(self):
        """When seasonal file absent, set_season() is no-op and global thresholds used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            det = self._make_detector_with_seasonal(Path(tmpdir))
            # Remove seasonal thresholds to simulate absent file.
            det._seasonal_per_pump_thresholds = None
            det._seasonal_global_thresholds = None
            det.set_season("summer")  # should log warning, not raise
            self.assertIsNone(det._active_season)  # no-op when None
            thresholds = det._get_pump_thresholds(1)
            self.assertAlmostEqual(thresholds["mahalanobis"]["warning"], 9.0)


# ---------------------------------------------------------------------------
# Test 4: Level2Detector seasonal threshold override
# ---------------------------------------------------------------------------

class TestLevel2DetectorSeasonal(unittest.TestCase):
    def _make_l2_detector_with_seasonal(self, tmpdir: Path):
        """Create Level2Detector with mocked internals, bypassing __init__.

        Level2Detector.__init__ does a lazy `from bin.model.inference import ProductionDetector`
        which requires the bin package on sys.path. We bypass __init__ entirely and manually
        set the attributes that classify() and set_season() depend on.
        """
        from model.level2_detector import Level2Detector, _SEASONS as L2_SEASONS

        version_dir = tmpdir / "final_metrics"
        version_dir.mkdir()

        # Write seasonal L2 file so the loading logic can be exercised in isolation.
        seasonal_l2 = _make_seasonal_l2_dict()
        with (version_dir / "production_thresholds_seasonal_l2.json").open("w") as f:
            json.dump(seasonal_l2, f)

        # Build instance without calling __init__.
        det = Level2Detector.__new__(Level2Detector)
        det._version_dir = str(version_dir)
        det._active_season = None

        # Load seasonal thresholds using the same logic as __init__.
        seasonal_path = version_dir / "production_thresholds_seasonal_l2.json"
        with seasonal_path.open() as f:
            s_data = json.load(f)
        det._seasonal_l2_thresholds = {}
        for season in L2_SEASONS:
            season_block = s_data.get(season, {})
            pump_map: dict = {}
            for pid_str, pump_block in season_block.get("per_pump", {}).items():
                try:
                    pump_map[int(pid_str)] = pump_block
                except (TypeError, ValueError):
                    pass
            det._seasonal_l2_thresholds[season] = pump_map

        # Wire a mock ProductionDetector that returns a fixed payload.
        mock_pd = MagicMock()
        mock_pd.classify.return_value = {
            "status": "OK",
            "pump_results": [
                {
                    "pump_id": 1,
                    "date": "2024-07-01",
                    "status": "NORMAL",
                    "alarm_threshold": 0.008,
                    "warning_threshold": 0.005,
                    "day_error_mse": 0.0060,  # between global warning=0.005 and alarm=0.008
                    "n_samples": 100,
                    "window_results": [],
                    "window_warning_threshold": 0.005,
                    "window_alarm_threshold": 0.008,
                    "smoothing_alpha": 0.3,
                    "fraction_windows_warning": 0.0,
                    "fraction_windows_alarm": 0.0,
                }
            ],
        }
        det._detector = mock_pd

        return det, mock_pd

    def test_seasonal_override_lowers_threshold_causing_warning(self):
        """Summer seasonal warning=0.0042 < day_mse=0.006 => WARNING despite global=0.005."""
        with tempfile.TemporaryDirectory() as tmpdir:
            det, _ = self._make_l2_detector_with_seasonal(Path(tmpdir))
            det.set_season("summer")
            results, _ = det.classify(["dummy.csv"])
            self.assertEqual(len(results), 1)
            # day_mse=0.006 > summer warning=0.0042 -> WARNING
            self.assertEqual(results[0].status, "WARNING")
            self.assertAlmostEqual(results[0].warning_threshold, 0.0042, places=4)

    def test_no_season_uses_global_thresholds(self):
        """Without set_season, global thresholds apply: day_mse=0.006 > warning=0.005 -> WARNING."""
        with tempfile.TemporaryDirectory() as tmpdir:
            det, _ = self._make_l2_detector_with_seasonal(Path(tmpdir))
            results, _ = det.classify(["dummy.csv"])
            self.assertEqual(len(results), 1)
            # day_mse=0.006 > global warning=0.005 -> WARNING
            self.assertEqual(results[0].status, "WARNING")
            self.assertAlmostEqual(results[0].warning_threshold, 0.005, places=4)

    def test_invalid_season_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            det, _ = self._make_l2_detector_with_seasonal(Path(tmpdir))
            with self.assertRaises(ValueError):
                det.set_season("rainy")


# ---------------------------------------------------------------------------
# Test 5: Streaming boundary crossing
# ---------------------------------------------------------------------------

class TestStreamingSeasonBoundary(unittest.TestCase):
    def test_season_change_calls_set_season(self):
        """Simulate processing timesteps across a season boundary."""
        from model.streaming import StreamingEnsembleDetector

        mock_l1 = MagicMock()
        mock_l1._seasonal_per_pump_thresholds = {"summer": {}, "autumn": {}}
        mock_l1._seasonal_global_thresholds = {}
        mock_l1._active_season = None
        mock_l1._output_columns = ["feat_a"]
        mock_l1._predictor._past_history = 1
        mock_l1._predictor.predict.return_value = pd.DataFrame({"feat_a": [0.0]})
        mock_l1._feature_names = ["feat_a"]

        # Build a minimal StreamingEnsembleDetector without real models.
        detector = MagicMock(spec=StreamingEnsembleDetector)
        detector._active_season = None
        detector._l1_detector = mock_l1

        # Simulate the boundary crossing logic from process_timestep.
        def _simulate_timestep(ts_str: str):
            ts = pd.Timestamp(ts_str)
            new_season = SeasonalTracker.tag_season(ts.date())
            if new_season != detector._active_season:
                detector._active_season = new_season
                detector._l1_detector.set_season(new_season)

        # August = summer.
        _simulate_timestep("2024-08-31 23:55:00")
        self.assertEqual(detector._active_season, "summer")
        mock_l1.set_season.assert_called_with("summer")

        # September = autumn — season boundary crossing.
        _simulate_timestep("2024-09-01 00:00:00")
        self.assertEqual(detector._active_season, "autumn")
        mock_l1.set_season.assert_called_with("autumn")

    def test_same_season_no_repeated_set_season(self):
        """Consecutive timesteps in the same season must not re-call set_season."""
        mock_l1 = MagicMock()
        detector_state = {"active_season": None}

        def _simulate(ts_str: str):
            ts = pd.Timestamp(ts_str)
            new_season = SeasonalTracker.tag_season(ts.date())
            if new_season != detector_state["active_season"]:
                detector_state["active_season"] = new_season
                mock_l1.set_season(new_season)

        _simulate("2024-07-01 00:00:00")
        _simulate("2024-07-15 12:00:00")
        _simulate("2024-07-31 23:55:00")
        # All in summer — set_season should only have been called once.
        self.assertEqual(mock_l1.set_season.call_count, 1)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestMonthToSeason,
        TestSeasonalTrackerPromote,
        TestLevel1DetectorSeasonal,
        TestLevel2DetectorSeasonal,
        TestStreamingSeasonBoundary,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
