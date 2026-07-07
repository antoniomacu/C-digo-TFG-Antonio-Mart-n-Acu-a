"""Unit and integration tests for P5 conformal prediction wrapper.

Run from ensemble/ directory:
    uv run pytest scripts/test_conformal_calibration.py -v

Tests cover:
  1. Conformal quantile formula correctness (finite-sample)
  2. Coverage guarantee holds empirically on synthetic data
  3. Per-pump fallback to global when n < MIN_SAMPLES
  4. JSON injection preserves existing keys
  5. End-to-end with synthetic calibration data
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ENSEMBLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENSEMBLE_DIR))


class TestConformalQuantileFormula(unittest.TestCase):
    """Test the core quantile computation."""

    def test_quantile_index_n19_alpha005(self):
        """n=19, α=0.05: ⌈(1-0.05)*(19+1)⌉ = ⌈19⌉ = 19 → index 18 (0-based) → largest score."""
        from cond_reg_v2.model.conformal_calibration import conformal_quantile

        scores = np.arange(1.0, 20.0)  # [1, 2, ..., 19]
        result = conformal_quantile(scores, alpha=0.05)
        self.assertAlmostEqual(result, 19.0)

    def test_quantile_index_n100_alpha005(self):
        """n=100, α=0.05: ⌈0.95*101⌉ = ⌈95.95⌉ = 96 → 96th smallest."""
        from cond_reg_v2.model.conformal_calibration import conformal_quantile

        scores = np.arange(1.0, 101.0)
        result = conformal_quantile(scores, alpha=0.05)
        self.assertAlmostEqual(result, 96.0)

    def test_quantile_index_n1000_alpha005(self):
        """n=1000, α=0.05: ⌈0.95*1001⌉ = ⌈950.95⌉ = 951 → 951st smallest."""
        from cond_reg_v2.model.conformal_calibration import conformal_quantile

        scores = np.arange(1.0, 1001.0)
        result = conformal_quantile(scores, alpha=0.05)
        self.assertAlmostEqual(result, 951.0)

    def test_quantile_single_sample(self):
        """n=1: ⌈0.95*2⌉ = ⌈1.9⌉ = 2 → but only 1 score, so clamp to max."""
        from cond_reg_v2.model.conformal_calibration import conformal_quantile

        scores = np.array([5.0])
        result = conformal_quantile(scores, alpha=0.05)
        self.assertAlmostEqual(result, 5.0)

    def test_quantile_all_identical(self):
        """All scores identical — quantile equals that value."""
        from cond_reg_v2.model.conformal_calibration import conformal_quantile

        scores = np.full(50, 3.14)
        result = conformal_quantile(scores, alpha=0.05)
        self.assertAlmostEqual(result, 3.14)

    def test_quantile_alpha010(self):
        """Different alpha: n=100, α=0.10: ⌈0.90*101⌉ = ⌈90.9⌉ = 91 → 91st."""
        from cond_reg_v2.model.conformal_calibration import conformal_quantile

        scores = np.arange(1.0, 101.0)
        result = conformal_quantile(scores, alpha=0.10)
        self.assertAlmostEqual(result, 91.0)


class TestCoverageGuarantee(unittest.TestCase):
    """Empirical coverage: conformal quantile controls exceedance rate."""

    def test_coverage_holds_on_exchangeable_data(self):
        """Generate calibration + test from same distribution; verify exceedance ≤ α."""
        from cond_reg_v2.model.conformal_calibration import conformal_quantile

        rng = np.random.default_rng(42)
        alpha = 0.05
        n_trials = 2000
        exceedances = 0

        for _ in range(n_trials):
            samples = rng.standard_normal(101)
            cal_scores = np.abs(samples[:100])
            test_score = abs(samples[100])
            q = conformal_quantile(cal_scores, alpha=alpha)
            if test_score > q:
                exceedances += 1

        empirical_rate = exceedances / n_trials
        self.assertLessEqual(
            empirical_rate,
            0.07,
            f"Empirical exceedance rate {empirical_rate:.4f} exceeds tolerance",
        )
        self.assertGreaterEqual(
            empirical_rate,
            0.02,
            f"Empirical exceedance rate {empirical_rate:.4f} is too conservative",
        )


class TestFiniteSampleCoverage(unittest.TestCase):
    """Test the finite_sample_coverage computation."""

    def test_coverage_value_n100(self):
        """n=100, α=0.05: coverage = ⌈0.95*101⌉/101 = 96/101 ≈ 0.9505."""
        from cond_reg_v2.model.conformal_calibration import finite_sample_coverage

        cov = finite_sample_coverage(n=100, alpha=0.05)
        expected = math.ceil(0.95 * 101) / 101
        self.assertAlmostEqual(cov, expected, places=6)

    def test_coverage_always_ge_one_minus_alpha(self):
        """Coverage is always ≥ (1 - α) by construction."""
        from cond_reg_v2.model.conformal_calibration import finite_sample_coverage

        for n in [1, 5, 19, 50, 100, 1000]:
            for alpha in [0.01, 0.05, 0.10, 0.20]:
                cov = finite_sample_coverage(n=n, alpha=alpha)
                self.assertGreaterEqual(
                    cov, 1.0 - alpha, f"n={n}, α={alpha}: coverage {cov} < {1 - alpha}"
                )


class TestPerPumpFallback(unittest.TestCase):
    """Test fallback logic when pump has insufficient samples."""

    def test_pump_below_min_samples_uses_global(self):
        """Pump with < 30 samples should get fallback_to_global=True."""
        from cond_reg_v2.model.conformal_calibration import ConformalCalibrator

        calibrator = ConformalCalibrator.__new__(ConformalCalibrator)
        calibrator.alpha = 0.05
        calibrator.min_samples = 30

        pump_scores = {
            "global": np.abs(np.random.default_rng(0).standard_normal(200)),
            "1": np.abs(np.random.default_rng(1).standard_normal(100)),
            "2": np.abs(np.random.default_rng(2).standard_normal(10)),
        }

        result = calibrator._build_all_conformal_blocks(pump_scores)

        self.assertFalse(result["global"]["fallback_to_global"])
        self.assertFalse(result["1"]["fallback_to_global"])
        self.assertTrue(result["2"]["fallback_to_global"])
        self.assertAlmostEqual(
            result["2"]["threshold"],
            result["global"]["threshold"],
        )


class TestJsonInjection(unittest.TestCase):
    """Test that inject_into_thresholds merges correctly."""

    def test_preserves_existing_keys(self):
        """Injection must not remove any existing keys."""
        from cond_reg_v2.model.conformal_calibration import ConformalCalibrator

        existing = {
            "description": "test",
            "global": {
                "mahalanobis": {
                    "warning": 7.0,
                    "alarm": 10.0,
                    "mean": 4.0,
                    "std": 2.0,
                    "p95": 7.0,
                    "p99": 10.0,
                },
                "per_feature": {},
                "covariance_matrix": [[1.0]],
                "inverse_covariance_matrix": [[1.0]],
                "mean_residual_vector": [0.0],
            },
            "per_pump": {
                "1": {
                    "n_samples": 100,
                    "fallback_to_global": False,
                    "mahalanobis": {
                        "warning": 6.5,
                        "alarm": 9.0,
                        "mean": 3.5,
                        "std": 1.5,
                        "p95": 6.5,
                        "p99": 9.0,
                    },
                    "per_feature": {},
                }
            },
        }

        conformal_blocks = {
            "global": {
                "alpha": 0.05,
                "threshold": 7.2,
                "n_calibration": 200,
                "finite_sample_coverage": 0.9502,
                "fallback_to_global": False,
                "guarantee": "P(false alarm) <= 0.05",
            },
            "1": {
                "alpha": 0.05,
                "threshold": 6.8,
                "n_calibration": 100,
                "finite_sample_coverage": 0.9505,
                "fallback_to_global": False,
                "guarantee": "P(false alarm) <= 0.05",
            },
        }

        result = ConformalCalibrator.inject_into_thresholds(existing, conformal_blocks)

        self.assertEqual(result["description"], "test")
        self.assertEqual(result["global"]["mahalanobis"]["warning"], 7.0)
        self.assertEqual(result["global"]["mahalanobis"]["alarm"], 10.0)
        self.assertEqual(result["per_pump"]["1"]["mahalanobis"]["warning"], 6.5)

        self.assertIn("conformal_quantile", result["global"]["mahalanobis"])
        self.assertAlmostEqual(
            result["global"]["mahalanobis"]["conformal_quantile"]["threshold"], 7.2
        )
        self.assertIn("conformal_quantile", result["per_pump"]["1"]["mahalanobis"])
        self.assertAlmostEqual(
            result["per_pump"]["1"]["mahalanobis"]["conformal_quantile"]["threshold"],
            6.8,
        )

    def test_pump_not_in_conformal_blocks_unchanged(self):
        """Pumps without conformal data should remain unchanged."""
        from cond_reg_v2.model.conformal_calibration import ConformalCalibrator

        existing = {
            "global": {
                "mahalanobis": {"warning": 7.0, "alarm": 10.0, "p95": 7.0, "p99": 10.0},
            },
            "per_pump": {
                "1": {"mahalanobis": {"warning": 6.5, "alarm": 9.0}},
                "3": {"mahalanobis": {"warning": 7.5, "alarm": 11.0}},
            },
        }
        conformal_blocks = {
            "global": {
                "alpha": 0.05,
                "threshold": 7.1,
                "n_calibration": 50,
                "finite_sample_coverage": 0.95,
                "fallback_to_global": False,
                "guarantee": "test",
            },
            "1": {
                "alpha": 0.05,
                "threshold": 6.6,
                "n_calibration": 30,
                "finite_sample_coverage": 0.95,
                "fallback_to_global": False,
                "guarantee": "test",
            },
        }

        result = ConformalCalibrator.inject_into_thresholds(existing, conformal_blocks)
        self.assertNotIn("conformal_quantile", result["per_pump"]["3"]["mahalanobis"])


class TestCLIConformalMode(unittest.TestCase):
    """Integration test: --mode conformal with a mocked calibrator."""

    def test_conformal_mode_injects_blocks(self):
        """Simulate --mode conformal end-to-end with synthetic thresholds."""
        from cond_reg_v2.model.conformal_calibration import (
            ConformalCalibrator,
        )

        # Create a minimal production_thresholds.json
        thresholds = {
            "description": "test",
            "method": "test",
            "n_training_samples": 100,
            "feature_names": ["feat_a", "feat_b"],
            "global": {
                "mahalanobis": {
                    "warning": 5.0,
                    "alarm": 8.0,
                    "mean": 3.0,
                    "std": 1.0,
                    "p95": 5.0,
                    "p99": 8.0,
                },
                "per_feature": {},
                "covariance_matrix": [[1.0, 0.0], [0.0, 1.0]],
                "inverse_covariance_matrix": [[1.0, 0.0], [0.0, 1.0]],
                "mean_residual_vector": [0.0, 0.0],
            },
            "per_pump": {
                "1": {
                    "n_samples": 50,
                    "fallback_to_global": False,
                    "mahalanobis": {
                        "warning": 4.5,
                        "alarm": 7.0,
                        "mean": 2.8,
                        "std": 0.9,
                        "p95": 4.5,
                        "p99": 7.0,
                    },
                    "per_feature": {},
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            thresholds_path = Path(tmpdir) / "production_thresholds.json"
            with thresholds_path.open("w") as f:
                json.dump(thresholds, f)

            # Simulate calibration scores (bypass actual model inference)
            rng = np.random.default_rng(123)
            pump_scores = {
                "global": np.abs(rng.standard_normal(200)) * 3,
                "1": np.abs(rng.standard_normal(80)) * 2.5,
            }

            calibrator = ConformalCalibrator.__new__(ConformalCalibrator)
            calibrator.alpha = 0.05
            calibrator.min_samples = 30
            calibrator.thresholds_path = thresholds_path

            conformal_blocks = calibrator._build_all_conformal_blocks(pump_scores)

            # Load and inject
            with thresholds_path.open("r") as f:
                loaded = json.load(f)

            result = ConformalCalibrator.inject_into_thresholds(
                loaded, conformal_blocks
            )

            # Write back
            with thresholds_path.open("w") as f:
                json.dump(result, f, indent=2)

            # Verify output
            with thresholds_path.open("r") as f:
                final = json.load(f)

            # Global conformal block exists
            gcq = final["global"]["mahalanobis"]["conformal_quantile"]
            self.assertEqual(gcq["alpha"], 0.05)
            self.assertGreater(gcq["threshold"], 0)
            self.assertEqual(gcq["n_calibration"], 200)
            self.assertGreaterEqual(gcq["finite_sample_coverage"], 0.95)
            self.assertIn("guarantee", gcq)

            # Per-pump conformal block
            pcq = final["per_pump"]["1"]["mahalanobis"]["conformal_quantile"]
            self.assertEqual(pcq["alpha"], 0.05)
            self.assertEqual(pcq["n_calibration"], 80)

            # Original keys intact
            self.assertEqual(final["global"]["mahalanobis"]["warning"], 5.0)
            self.assertEqual(final["per_pump"]["1"]["mahalanobis"]["alarm"], 7.0)


if __name__ == "__main__":
    unittest.main()
