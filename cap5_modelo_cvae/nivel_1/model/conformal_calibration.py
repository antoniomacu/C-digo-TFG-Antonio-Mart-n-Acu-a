"""Split conformal prediction on L1 Mahalanobis distance scores.

Provides finite-sample valid thresholds (coverage ≥ 1-α) under the
exchangeability assumption between calibration and test data.

Reference: Vovk et al. (2005), Lei et al. (2018).

Usage (CLI):
    uv run python -m cond_reg_v2.model.threshold_calibration --mode conformal

Usage (API):
    from cond_reg_v2.model.conformal_calibration import ConformalCalibrator
    calibrator = ConformalCalibrator(alpha=0.05)
    conformal_blocks = calibrator.calibrate()
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .inference import PumpPredictor

MIN_SAMPLES_CONFORMAL = 30


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute the split conformal quantile with finite-sample correction.

    Given n calibration scores, returns the ⌈(1-α)(n+1)⌉-th smallest score.
    This guarantees P(new_score > quantile) ≤ α under exchangeability.

    Args:
        scores: 1-D array of calibration nonconformity scores (Mahalanobis distances).
        alpha: Target miscoverage rate (e.g. 0.05 for 95% coverage).

    Returns:
        The conformal threshold value.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot compute conformal quantile on empty score array")

    sorted_scores = np.sort(scores)
    raw_index = math.ceil((1 - alpha) * (n + 1))
    index = min(raw_index, n) - 1  # Convert to 0-based, clamp to valid range
    return float(sorted_scores[index])


def finite_sample_coverage(n: int, alpha: float) -> float:
    """Compute the exact finite-sample coverage level.

    The conformal guarantee provides coverage = ⌈(1-α)(n+1)⌉ / (n+1),
    which is always ≥ (1-α).

    Args:
        n: Number of calibration samples.
        alpha: Target miscoverage rate.

    Returns:
        Exact coverage probability (≥ 1-α).
    """
    # raw_index = ceil((1-α)(n+1)) ≥ (1-α)(n+1), so raw_index/(n+1) ≥ 1-α always.
    # Clamping is only needed for array indexing in conformal_quantile, not here.
    raw_index = math.ceil((1 - alpha) * (n + 1))
    return raw_index / (n + 1)


class ConformalCalibrator:
    """Compute conformal prediction thresholds on L1 Mahalanobis scores.

    Reads the existing production_thresholds.json for covariance parameters
    (frozen from --mode global), computes Mahalanobis distances on the
    calibration data, then produces conformal quantile blocks.
    """

    def __init__(
        self,
        weights_dir: str | Path | None = None,
        train_path: str | Path | None = None,
        thresholds_path: str | Path | None = None,
        alpha: float = 0.05,
    ):
        self.alpha = alpha
        self.min_samples = MIN_SAMPLES_CONFORMAL

        default_weights = Path(__file__).parent / "weights"
        self.weights_dir = Path(weights_dir) if weights_dir else default_weights

        self.thresholds_path = (
            Path(thresholds_path)
            if thresholds_path
            else self.weights_dir / "production_thresholds.json"
        )

        self.predictor = PumpPredictor(weights_dir=self.weights_dir)
        self.feature_names = list(self.predictor.output_columns)

        self.train_path = self._resolve_train_path(train_path or "../../data/train/")

    @staticmethod
    def _resolve_train_path(train_path: str | Path) -> Path:
        path = Path(train_path)
        if path.is_absolute():
            return path
        if path.exists():
            return path.resolve()
        module_base = Path(__file__).resolve().parents[1]
        return (module_base / path).resolve()

    def _load_existing_thresholds(self) -> dict[str, Any]:
        """Load existing production_thresholds.json; raise if missing."""
        if not self.thresholds_path.exists():
            raise FileNotFoundError(
                f"production_thresholds.json not found at {self.thresholds_path}. "
                "Run --mode global first to generate covariance parameters."
            )
        with self.thresholds_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _compute_calibration_scores(
        self, thresholds: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """Compute Mahalanobis distances on calibration data, grouped by pump."""
        global_block = thresholds["global"]
        inv_cov = np.asarray(global_block["inverse_covariance_matrix"], dtype=float)
        mean_residual = np.asarray(global_block["mean_residual_vector"], dtype=float)

        csv_files = sorted(self.train_path.glob("*.csv"))
        if not csv_files:
            csv_files = sorted(self.train_path.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found under calibration path: {self.train_path}"
            )

        rows: list[pd.DataFrame] = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            missing = [c for c in self.feature_names if c not in df.columns]
            if missing:
                continue

            if "pump_id" in df.columns:
                pump_ids = pd.to_numeric(df["pump_id"], errors="coerce")
            else:
                match = re.search(r"pump[_-]?(\d+)", csv_path.stem, flags=re.IGNORECASE)
                if match:
                    pump_ids = pd.Series(
                        np.full(len(df), int(match.group(1))), dtype=float
                    )
                else:
                    continue

            try:
                predictions = self.predictor.predict(df)
            except Exception:
                continue

            actual = df[self.feature_names].apply(pd.to_numeric, errors="coerce")
            predicted = predictions[self.feature_names].apply(
                pd.to_numeric, errors="coerce"
            )
            residual = (actual - predicted).to_numpy(dtype=float)

            frame = pd.DataFrame({"pump_id": pump_ids.values})
            frame[self.feature_names] = residual
            frame = frame.dropna()
            frame["pump_id"] = frame["pump_id"].astype(int)
            rows.append(frame)

        if not rows:
            raise ValueError(
                f"No valid calibration data found under: {self.train_path}"
            )

        all_residuals = pd.concat(rows, ignore_index=True)

        residual_matrix = all_residuals[self.feature_names].to_numpy(dtype=float)
        centered = residual_matrix - mean_residual
        mahal_sq = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
        mahal_sq = np.clip(mahal_sq, a_min=0.0, a_max=None)
        mahal = np.sqrt(mahal_sq)

        all_residuals["_mahal"] = mahal

        pump_scores: dict[str, np.ndarray] = {"global": mahal}
        for pump_id, group in all_residuals.groupby("pump_id"):
            pump_scores[str(int(pump_id))] = group["_mahal"].to_numpy()

        return pump_scores

    def _build_all_conformal_blocks(
        self, pump_scores: dict[str, np.ndarray]
    ) -> dict[str, dict[str, Any]]:
        """Build conformal quantile blocks for global + each pump."""
        global_scores = pump_scores["global"]
        global_block = self._build_single_block(global_scores)
        global_block["fallback_to_global"] = False

        result: dict[str, dict[str, Any]] = {"global": global_block}

        for key, scores in pump_scores.items():
            if key == "global":
                continue
            if len(scores) < self.min_samples:
                result[key] = {
                    **global_block,
                    "n_calibration": int(len(scores)),
                    "fallback_to_global": True,
                }
            else:
                block = self._build_single_block(scores)
                block["fallback_to_global"] = False
                result[key] = block

        return result

    def _build_single_block(self, scores: np.ndarray) -> dict[str, Any]:
        """Build a single conformal quantile block from scores."""
        n = len(scores)
        threshold = conformal_quantile(scores, self.alpha)
        coverage = finite_sample_coverage(n, self.alpha)

        return {
            "alpha": self.alpha,
            "threshold": threshold,
            "n_calibration": n,
            "finite_sample_coverage": round(coverage, 6),
            "guarantee": (
                f"P(false alarm on normal pump-day) <= {self.alpha} "
                f"under exchangeability (finite-sample valid, n={n})"
            ),
        }

    def calibrate(self) -> dict[str, dict[str, Any]]:
        """Run full conformal calibration pipeline."""
        thresholds = self._load_existing_thresholds()
        pump_scores = self._compute_calibration_scores(thresholds)
        return self._build_all_conformal_blocks(pump_scores)

    @staticmethod
    def inject_into_thresholds(
        thresholds: dict[str, Any],
        conformal_blocks: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Inject conformal_quantile blocks into existing thresholds dict.

        Adds a "conformal_quantile" key inside each mahalanobis block
        without modifying any existing keys.
        """
        if "global" in conformal_blocks and "global" in thresholds:
            global_mahal = thresholds["global"].get("mahalanobis", {})
            block = conformal_blocks["global"].copy()
            block.pop("fallback_to_global", None)
            global_mahal["conformal_quantile"] = block
            thresholds["global"]["mahalanobis"] = global_mahal

        per_pump = thresholds.get("per_pump", {})
        for pump_id, pump_block in per_pump.items():
            if pump_id in conformal_blocks:
                pump_mahal = pump_block.get("mahalanobis", {})
                block = conformal_blocks[pump_id].copy()
                block.pop("fallback_to_global", None)
                pump_mahal["conformal_quantile"] = block
                pump_block["mahalanobis"] = pump_mahal

        return thresholds
