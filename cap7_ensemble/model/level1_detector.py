"""
Level 1 Detector - Instantaneous anomaly detection via digital twin.

Wraps cond_reg.model.inference.PumpPredictor, computes residuals between
predicted and actual sensor readings, and flags anomalies using per-feature
Z-scores and Mahalanobis distance.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .scoring import (
    FeatureAnomaly,
    TimestepResult,
    Level1Result,
    ChannelHealth,
    ChannelHealthSummary,
    classify_status,
    normalize_severity,
    compute_channel_health,
    STATUS_NORMAL,
    STATUS_WARNING,
    STATUS_ALARM,
)

# Default paths
DEFAULT_WEIGHTS_DIR = (
    Path(__file__).resolve().parent.parent / "cond_reg_v2" / "model" / "weights"
)

logger = logging.getLogger(__name__)

_SEASONS = ["winter", "spring", "summer", "autumn"]
_SEASON_MAP: dict[str, list[int]] = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}


def _month_to_season(month: int) -> str:
    for season, months in _SEASON_MAP.items():
        if month in months:
            return season
    raise ValueError(f"Invalid month: {month}")


def _aggregate_channel_health(
    per_timestep: dict[str, list[ChannelHealth]],
) -> list[ChannelHealthSummary]:
    """Aggregate per-timestep ChannelHealth into day-level summaries."""
    summaries: list[ChannelHealthSummary] = []
    for feature, entries in sorted(per_timestep.items()):
        if not entries:
            continue
        healths = [e.health for e in entries]
        z_scores = [e.z_score for e in entries]
        summaries.append(
            ChannelHealthSummary(
                feature=feature,
                mean_health=sum(healths) / len(healths),
                min_health=min(healths),
                mean_z_score=sum(z_scores) / len(z_scores),
                max_z_score=max(z_scores),
                n_timesteps_below_50=sum(1 for h in healths if h < 50.0),
            )
        )
    return summaries


class Level1Detector:
    """
    Instantaneous anomaly detector - digital twin approach.

    For each timestep:
    1. Predicts expected sensor values from operating conditions
    2. Computes residuals (actual - predicted)
    3. Scores each feature via Z-score (using per-pump training statistics)
    4. Computes Mahalanobis distance on the residual vector
    5. Classifies as NORMAL/WARNING/ALARM

    Parameters
    ----------
    weights_dir : path to cond_reg weights (best_weights.pt, norm_params.json, production_thresholds.json)
    device : "auto", "cpu", or "cuda"
    feature_z_threshold : Z-score threshold to flag individual features (default 3.0)
    top_k_timesteps : Number of most anomalous timesteps to include in results (default 10)
    """

    def __init__(
        self,
        weights_dir=None,
        device="auto",
        feature_z_threshold=3.0,
        top_k_timesteps=10,
    ):
        resolved_weights_dir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
        self._validate_weights_dir(resolved_weights_dir)

        # Lazy import to avoid hard dependency failures at import time.
        from cond_reg_v2.model.inference import PumpPredictor

        self._predictor = PumpPredictor(weights_dir=resolved_weights_dir, device=device)
        self._weights_dir = str(resolved_weights_dir)
        self._feature_z_threshold = float(feature_z_threshold)
        self._top_k_timesteps = int(top_k_timesteps)

        thresholds_path = resolved_weights_dir / "production_thresholds.json"
        with open(thresholds_path, "r", encoding="utf-8") as f:
            self._thresholds = json.load(f)

        self._output_columns: list[str] = list(self._predictor.output_columns)
        feature_names = self._thresholds.get("feature_names")
        if isinstance(feature_names, list) and feature_names:
            self._feature_names = list(feature_names)
        else:
            self._feature_names = list(self._output_columns)

        if set(self._feature_names) != set(self._output_columns):
            raise ValueError(
                "Mismatch between threshold feature_names and PumpPredictor output_columns. "
                f"thresholds={self._feature_names}, predictor={self._output_columns}"
            )

        self._global_thresholds = self._parse_threshold_block(
            self._thresholds.get("global", {})
        )

        self._per_pump_thresholds: dict[int, dict] = {}
        for pump_id_str, block in self._thresholds.get("per_pump", {}).items():
            try:
                pump_id_int = int(pump_id_str)
            except (TypeError, ValueError):
                continue
            parsed = self._parse_threshold_block(block)
            parsed["fallback_to_global"] = bool(block.get("fallback_to_global", False))
            self._per_pump_thresholds[pump_id_int] = parsed

        # Optional seasonal thresholds — four sets of per-pump thresholds keyed by season name.
        self._active_season: str | None = None
        self._seasonal_per_pump_thresholds: dict[str, dict[int, dict]] | None = None
        self._seasonal_global_thresholds: dict[str, dict] | None = None
        seasonal_path = resolved_weights_dir / "production_thresholds_seasonal.json"
        if seasonal_path.exists():
            try:
                with seasonal_path.open("r", encoding="utf-8") as f:
                    seasonal_data = json.load(f)
                self._seasonal_per_pump_thresholds = {}
                self._seasonal_global_thresholds = {}
                for season in _SEASONS:
                    season_block = seasonal_data.get(season, {})
                    # Parse global block for this season.
                    self._seasonal_global_thresholds[season] = (
                        self._parse_threshold_block(season_block.get("global", {}))
                    )
                    # Parse per-pump blocks for this season.
                    season_pump_map: dict[int, dict] = {}
                    for pid_str, pump_block in season_block.get("per_pump", {}).items():
                        try:
                            pid_int = int(pid_str)
                        except (TypeError, ValueError):
                            continue
                        parsed_pump = self._parse_threshold_block(pump_block)
                        parsed_pump["fallback_to_global"] = bool(
                            pump_block.get("fallback_to_global", False)
                        )
                        season_pump_map[pid_int] = parsed_pump
                    self._seasonal_per_pump_thresholds[season] = season_pump_map
                logger.debug("Loaded seasonal L1 thresholds from %s", seasonal_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load seasonal L1 thresholds from %s: %s",
                    seasonal_path,
                    exc,
                )
                self._seasonal_per_pump_thresholds = None
                self._seasonal_global_thresholds = None

    @property
    def weights_dir(self) -> str:
        return self._weights_dir

    def set_season(self, season: str) -> None:
        """Set the active season for threshold selection.

        If no seasonal threshold file was loaded, this is a no-op (logs warning once).
        """
        if season not in _SEASONS:
            raise ValueError(f"Invalid season '{season}'. Must be one of {_SEASONS}")
        if self._seasonal_per_pump_thresholds is None:
            logger.warning(
                "set_season('%s') called but no seasonal L1 thresholds loaded — using global thresholds",
                season,
            )
            return
        self._active_season = season

    @staticmethod
    def _validate_weights_dir(weights_dir: Path) -> None:
        if not weights_dir.exists() or not weights_dir.is_dir():
            raise FileNotFoundError(f"Level 1 weights_dir not found: {weights_dir}")

        required_files = [
            "norm_params.json",
            "production_thresholds.json",
        ]
        missing = [name for name in required_files if not (weights_dir / name).exists()]
        has_supported_model_weights = any(
            (weights_dir / name).exists()
            for name in ("best_weights.pt", "model_weights.ckpt")
        )
        if not has_supported_model_weights:
            missing.append("best_weights.pt or model_weights.ckpt")

        if missing:
            missing_joined = ", ".join(missing)
            raise FileNotFoundError(
                f"Level 1 weights_dir is missing required files: {missing_joined} "
                f"(weights_dir={weights_dir})"
            )

    def _parse_threshold_block(self, block: dict) -> dict:
        per_feature = block.get("per_feature", {})

        means: list[float] = []
        stds: list[float] = []
        for feature in self._feature_names:
            values = per_feature.get(feature, {})
            means.append(float(values.get("mean_residual", 0.0)))
            std_val = float(values.get("std_residual", 1.0))
            stds.append(std_val if std_val > 1e-12 else 1e-12)

        mean_residual_vector = block.get("mean_residual_vector")
        if isinstance(mean_residual_vector, list) and len(mean_residual_vector) == len(
            self._feature_names
        ):
            mean_vector = np.asarray(mean_residual_vector, dtype=float)
        else:
            mean_vector = np.asarray(means, dtype=float)

        inverse_covariance_matrix = block.get("inverse_covariance_matrix")
        if isinstance(inverse_covariance_matrix, list) and inverse_covariance_matrix:
            cov_inv = np.asarray(inverse_covariance_matrix, dtype=float)
        else:
            cov_inv = None

        covariance_matrix = block.get("covariance_matrix")
        if cov_inv is not None:
            pass
        elif isinstance(covariance_matrix, list) and covariance_matrix:
            cov = np.asarray(covariance_matrix, dtype=float)
            cov_inv = np.linalg.pinv(cov)
        else:
            # Per-pump blocks may omit covariance; use per-feature variance instead of identity.
            # Identity assumes unit variance for all residuals and can inflate Mahalanobis scores.
            var_vector = np.square(np.asarray(stds, dtype=float))
            var_vector = np.where(var_vector > 1e-12, var_vector, 1e-12)
            cov_inv = np.diag(1.0 / var_vector)

        maha = block.get("mahalanobis", {})
        warning_threshold = float(maha.get("warning", np.inf))
        alarm_threshold = float(maha.get("alarm", np.inf))
        if alarm_threshold < warning_threshold:
            alarm_threshold = warning_threshold

        return {
            "mahalanobis": {
                "warning": warning_threshold,
                "alarm": alarm_threshold,
            },
            "mean_vector": mean_vector,
            "std_vector": np.asarray(stds, dtype=float),
            "cov_inv": cov_inv,
            "per_feature_raw": per_feature,
        }

    def classify(self, df: pd.DataFrame) -> tuple[list[Level1Result], dict]:
        """
        Run Level 1 detection on a DataFrame containing pump sensor data.

        The DataFrame must contain:
        - Timestamp index (or 'timestamp' column)
        - "pump_id" column (int 1-4)
        - "Ambient temperature", "Main HTF Pump Speed", "Main HTF Pump Inlet Temperature"
        - All 13 output sensor columns (for residual computation)

        Processing:
        1. Group data by (pump_id, date)
        2. For each group, run PumpPredictor on operating conditions
        3. Compute residuals for each timestep
        4. Score each timestep (Z-scores + Mahalanobis)
        5. Aggregate to day-level statistics
        6. Return Level1Result per pump-day

        Returns
        -------
        results : list of Level1Result
        timing : dict
        """
        t_start = time.perf_counter()

        t_pre_start = time.perf_counter()
        prepared = self._prepare_dataframe(df)
        t_preprocessing = time.perf_counter() - t_pre_start

        prepared["_date"] = prepared.index.date

        t_inf_start = time.perf_counter()
        results: list[Level1Result] = []
        skipped_nan_timesteps = 0
        scored_timesteps = 0
        n_groups = 0

        grouped = prepared.groupby(["pump_id", "_date"], sort=True)
        _prev_season: str | None = None
        for (pump_id_raw, date_obj), group_df in grouped:
            n_groups += 1
            pump_id = int(pump_id_raw)

            # Auto-select seasonal thresholds for this pump-day.
            try:
                season = _month_to_season(date_obj.month)
                if season != _prev_season:
                    self.set_season(season)
                    _prev_season = season
            except Exception:
                pass

            input_cols = [
                "Ambient temperature",
                "Main HTF Pump Speed",
                "Main HTF Pump Inlet Temperature",
                "pump_id",
            ]
            predicted = self._predictor.predict(group_df[input_cols])
            actual = group_df[self._output_columns]
            residuals = actual - predicted

            per_channel_accum: dict[str, list[ChannelHealth]] = {}
            timestep_results: list[TimestepResult] = []
            per_feature_stats = self._get_per_feature_stats(pump_id)

            for idx, residual_series in residuals.iterrows():
                residual_row = residual_series.to_numpy(dtype=float)
                if np.isnan(residual_row).any():
                    skipped_nan_timesteps += 1
                    continue

                actual_row = actual.loc[idx].to_numpy(dtype=float)
                predicted_row = predicted.loc[idx].to_numpy(dtype=float)

                timestep_result = self._score_timestep(
                    residual_row=residual_row,
                    pump_id=pump_id,
                    timestamp=pd.Timestamp(idx).isoformat(),
                    actual_row=actual_row,
                    predicted_row=predicted_row,
                )
                timestep_results.append(timestep_result)
                scored_timesteps += 1

                for i, feature_name in enumerate(self._feature_names):
                    fs = per_feature_stats.get(feature_name)
                    if fs is None:
                        continue
                    h, z = compute_channel_health(
                        residual=float(residual_row[i]),
                        std_residual=float(fs.get("std_residual", 1.0)),
                        p99_abs_residual=float(fs.get("p99_abs_residual", 1.0)),
                        mean_residual=float(fs.get("mean_residual", 0.0)),
                    )
                    per_channel_accum.setdefault(feature_name, []).append(
                        ChannelHealth(
                            feature=feature_name,
                            health=h,
                            z_score=z,
                            residual=float(residual_row[i]),
                        )
                    )

            if not timestep_results:
                continue

            thresholds = self._get_pump_thresholds(pump_id)
            warning_threshold = float(thresholds["mahalanobis"]["warning"])
            alarm_threshold = float(thresholds["mahalanobis"]["alarm"])

            mahalanobis_values = np.asarray(
                [ts.mahalanobis for ts in timestep_results], dtype=float
            )
            day_mean = float(np.mean(mahalanobis_values))
            day_max = float(np.max(mahalanobis_values))

            count_warning_or_alarm = sum(
                1
                for ts in timestep_results
                if ts.status in {STATUS_WARNING, STATUS_ALARM}
            )
            count_alarm = sum(1 for ts in timestep_results if ts.status == STATUS_ALARM)
            n_timesteps = len(timestep_results)

            alarm_fraction = count_alarm / n_timesteps
            if day_max >= alarm_threshold and alarm_fraction >= 0.02:
                # Sustained alarm: ≥2% of timesteps above alarm threshold
                day_status = STATUS_ALARM
            elif day_max >= alarm_threshold:
                # Single spike above alarm → downgrade to WARNING
                day_status = STATUS_WARNING
            elif day_max >= warning_threshold:
                day_status = STATUS_WARNING
            else:
                day_status = STATUS_NORMAL
            top_timesteps = sorted(
                timestep_results,
                key=lambda ts: ts.mahalanobis,
                reverse=True,
            )[: self._top_k_timesteps]

            results.append(
                Level1Result(
                    pump_id=pump_id,
                    date=str(date_obj),
                    status=day_status,
                    day_mean_mahalanobis=day_mean,
                    day_max_mahalanobis=day_max,
                    fraction_above_warning=float(count_warning_or_alarm / n_timesteps),
                    fraction_above_alarm=float(count_alarm / n_timesteps),
                    normalized_severity=normalize_severity(day_max, alarm_threshold),
                    n_timesteps=n_timesteps,
                    top_anomalous_timesteps=top_timesteps,
                    all_timestep_results=timestep_results,
                    channel_health_summary=_aggregate_channel_health(per_channel_accum),
                )
            )

        t_inference = time.perf_counter() - t_inf_start
        t_total = time.perf_counter() - t_start

        timing = {
            "preprocessing_seconds": round(t_preprocessing, 4),
            "inference_seconds": round(t_inference, 4),
            "level1_seconds": round(t_total, 4),
            "n_groups": n_groups,
            "n_scored_timesteps": scored_timesteps,
            "n_skipped_nan_timesteps": skipped_nan_timesteps,
        }
        return results, timing

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        prepared = df.copy()

        if "timestamp" in prepared.columns:
            ts = pd.to_datetime(prepared["timestamp"], errors="coerce")
            if ts.isna().any():
                raise ValueError("'timestamp' column contains invalid datetime values")
            prepared = prepared.drop(columns=["timestamp"])
            prepared.index = ts
        elif isinstance(prepared.index, pd.DatetimeIndex):
            if prepared.index.isna().any():
                raise ValueError("Datetime index contains NaT values")
        else:
            raise ValueError(
                "DataFrame must have a DatetimeIndex or a 'timestamp' column"
            )

        if "pump_id" not in prepared.columns:
            raise ValueError("DataFrame must contain 'pump_id' column")

        required_inputs = [
            "Ambient temperature",
            "Main HTF Pump Speed",
            "Main HTF Pump Inlet Temperature",
        ]
        missing_inputs = [col for col in required_inputs if col not in prepared.columns]
        if missing_inputs:
            raise ValueError(f"Missing required input columns: {missing_inputs}")

        missing_outputs = [
            col for col in self._output_columns if col not in prepared.columns
        ]
        if missing_outputs:
            raise ValueError(f"Missing required output columns: {missing_outputs}")

        prepared["pump_id"] = pd.to_numeric(prepared["pump_id"], errors="coerce")
        if prepared["pump_id"].isna().any():
            raise ValueError("Column 'pump_id' contains non-numeric values")
        prepared["pump_id"] = prepared["pump_id"].astype(int)

        return prepared.sort_index()

    def _score_timestep(
        self,
        residual_row: np.ndarray,
        pump_id: int,
        timestamp: str,
        actual_row: Optional[np.ndarray] = None,
        predicted_row: Optional[np.ndarray] = None,
    ) -> TimestepResult:
        """Score a single timestep's residual vector."""
        thresholds = self._get_pump_thresholds(pump_id)

        mean_vec = thresholds["mean_vector"]
        std_vec = thresholds["std_vector"]
        cov_inv = thresholds["cov_inv"]
        warning_threshold = float(thresholds["mahalanobis"]["warning"])
        alarm_threshold = float(thresholds["mahalanobis"]["alarm"])

        z_scores = (residual_row - mean_vec) / std_vec
        mahalanobis = self._compute_mahalanobis(residual_row, mean_vec, cov_inv)
        status = classify_status(mahalanobis, warning_threshold, alarm_threshold)

        if actual_row is None:
            actual_row = np.full_like(residual_row, np.nan, dtype=float)
        if predicted_row is None:
            predicted_row = np.full_like(residual_row, np.nan, dtype=float)

        feature_scores = [
            FeatureAnomaly(
                feature=feature,
                actual=float(actual_row[i]),
                predicted=float(predicted_row[i]),
                residual=float(residual_row[i]),
                z_score=float(z_scores[i]),
                is_anomalous=bool(abs(float(z_scores[i])) > self._feature_z_threshold),
            )
            for i, feature in enumerate(self._feature_names)
        ]

        return TimestepResult(
            timestamp=timestamp,
            mahalanobis=float(mahalanobis),
            status=status,
            feature_scores=feature_scores,
        )

    def _get_pump_thresholds(self, pump_id: int) -> dict:
        """Get thresholds for a specific pump using the seasonal fallback chain.

        Fallback order:
        1. Seasonal per-pump (if season set and file loaded and pump not flagged fallback)
        2. Seasonal global for active season
        3. Non-seasonal per-pump
        4. Non-seasonal global
        """
        pid = int(pump_id)

        if (
            self._active_season is not None
            and self._seasonal_per_pump_thresholds is not None
        ):
            season_pump_map = self._seasonal_per_pump_thresholds.get(
                self._active_season, {}
            )
            pump_block = season_pump_map.get(pid)
            if pump_block is not None and not pump_block.get(
                "fallback_to_global", False
            ):
                return pump_block
            # Fall to seasonal global for this season.
            if self._seasonal_global_thresholds is not None:
                season_global = self._seasonal_global_thresholds.get(
                    self._active_season
                )
                if season_global is not None:
                    return season_global

        # Non-seasonal path (original behaviour).
        pump_thresholds = self._per_pump_thresholds.get(pid)
        if not pump_thresholds:
            return self._global_thresholds
        if pump_thresholds.get("fallback_to_global", False):
            return self._global_thresholds
        return pump_thresholds

    def _get_per_feature_stats(self, pump_id: int) -> dict[str, dict]:
        """Get raw per-feature stats for per-channel health computation.

        Falls back to global stats if per-pump data is unavailable.
        """
        thresholds = self._get_pump_thresholds(pump_id)
        per_feature = thresholds.get("per_feature_raw", {})
        if per_feature:
            return per_feature
        return self._global_thresholds.get("per_feature_raw", {})

    def _compute_mahalanobis(
        self, residual: np.ndarray, mean_vec: np.ndarray, cov_inv: np.ndarray
    ) -> float:
        """Compute Mahalanobis distance for a single residual vector."""
        diff = residual - mean_vec
        dist_sq = float(diff @ cov_inv @ diff)
        if dist_sq < 0.0:
            dist_sq = 0.0
        return float(np.sqrt(dist_sq))
