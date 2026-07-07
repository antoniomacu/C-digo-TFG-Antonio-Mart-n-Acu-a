"""Threshold calibration for cond_reg_v2 production anomaly detection."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .inference import PumpPredictor

MIN_SAMPLES_PER_PUMP = 30
REG_EPS = 1e-6

_SEASON_MAP: dict[str, list[int]] = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}
_SEASONS: list[str] = ["winter", "spring", "summer", "autumn"]


def _month_to_season(month: int) -> str:
    for season, months in _SEASON_MAP.items():
        if month in months:
            return season
    raise ValueError(f"Invalid month: {month}")


class ThresholdCalibrator:
    def __init__(self, weights_dir=None, train_path=None):
        """
        Args:
            weights_dir: Path to model weights (best_weights.pt + norm_params.json)
            train_path: Path to training CSV files for calibration
        """
        self.predictor = PumpPredictor(weights_dir=weights_dir)
        self.train_path = self._resolve_train_path(train_path or "../../data/train/")
        self.feature_names = list(self.predictor.output_columns)

    @staticmethod
    def _resolve_train_path(train_path: str | Path) -> Path:
        path = Path(train_path)
        if path.is_absolute():
            return path

        if path.exists():
            return path.resolve()

        module_base = Path(__file__).resolve().parents[1]
        return (module_base / path).resolve()

    def calibrate(self) -> dict:
        """Run full calibration pipeline. Returns thresholds dict."""
        residuals_df = self._compute_residuals()
        if residuals_df.empty:
            raise ValueError(
                f"No residuals computed from train path: {self.train_path}"
            )

        global_per_feature = self._compute_per_feature_stats(residuals_df)
        global_mahal = self._compute_mahalanobis(residuals_df)
        global_contrib = self._compute_per_feature_mahalanobis_contribution(
            residuals_df,
            np.asarray(global_mahal["inverse_covariance_matrix"], dtype=float),
            np.asarray(global_mahal["mean_residual_vector"], dtype=float),
        )

        global_per_feature = self._merge_feature_sections(
            global_per_feature, global_contrib
        )

        per_pump = {}
        expected_pumps = {1, 2, 3, 4}
        observed_pumps = {int(pid) for pid in residuals_df["pump_id"].dropna().unique()}
        all_pumps = sorted(expected_pumps | observed_pumps)

        for pump_id in all_pumps:
            pump_residuals = residuals_df[residuals_df["pump_id"] == pump_id]
            pump_id_str = str(int(pump_id))
            n_samples = int(len(pump_residuals))

            if n_samples < MIN_SAMPLES_PER_PUMP:
                fallback_global = {
                    "n_samples": n_samples,
                    "fallback_to_global": True,
                    "mahalanobis": copy.deepcopy(global_mahal["mahalanobis"]),
                    "per_feature": copy.deepcopy(global_per_feature),
                }
                per_pump[pump_id_str] = fallback_global
                continue

            pump_feature_stats = self._compute_per_feature_stats(pump_residuals)
            pump_mahal = self._compute_mahalanobis(pump_residuals)
            pump_contrib = self._compute_per_feature_mahalanobis_contribution(
                pump_residuals,
                np.asarray(pump_mahal["inverse_covariance_matrix"], dtype=float),
                np.asarray(pump_mahal["mean_residual_vector"], dtype=float),
            )

            per_pump[pump_id_str] = {
                "n_samples": n_samples,
                "fallback_to_global": False,
                "mahalanobis": pump_mahal["mahalanobis"],
                "per_feature": self._merge_feature_sections(
                    pump_feature_stats, pump_contrib
                ),
            }

        thresholds = {
            "description": "Production thresholds for real-time anomaly detection based on residual distributions from normal training data.",
            "method": "per-feature Z-score + Mahalanobis distance on residual vectors",
            "n_training_samples": int(len(residuals_df)),
            "feature_names": self.feature_names,
            "global": {
                "mahalanobis": global_mahal["mahalanobis"],
                "per_feature": global_per_feature,
                "covariance_matrix": global_mahal["covariance_matrix"],
                "inverse_covariance_matrix": global_mahal["inverse_covariance_matrix"],
                "mean_residual_vector": global_mahal["mean_residual_vector"],
            },
            "per_pump": per_pump,
        }
        return thresholds

    def _compute_residuals(self) -> pd.DataFrame:
        """Run predictor on all training files, return residuals DataFrame.

        For each CSV file:
        1. Read the file
        2. Run predictor.predict(df)
        3. Compute residual = actual - predicted for each output column
        4. Track pump_id and timestamp for each row

        Returns DataFrame with columns:
            timestamp, pump_id, + residual columns for each output variable
        """
        csv_files = sorted(self.train_path.glob("*.csv"))
        if not csv_files:
            csv_files = sorted(self.train_path.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found under train path: {self.train_path}"
            )

        rows: list[pd.DataFrame] = []

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                print(
                    f"[threshold_calibration] Skipping {csv_path.name}: read error: {exc}"
                )
                continue

            missing_outputs = [
                col for col in self.feature_names if col not in df.columns
            ]
            if missing_outputs:
                print(
                    f"[threshold_calibration] Skipping {csv_path.name}: missing output columns {missing_outputs}"
                )
                continue

            if "timestamp" in df.columns:
                timestamps = df["timestamp"].copy()
            else:
                timestamps = pd.Series(np.arange(len(df)), name="timestamp")

            pump_id_series = self._extract_pump_ids(df, csv_path)
            if pump_id_series is None:
                print(
                    f"[threshold_calibration] Skipping {csv_path.name}: unable to infer pump_id"
                )
                continue

            try:
                predictions = self.predictor.predict(df)
            except Exception as exc:
                print(
                    f"[threshold_calibration] Skipping {csv_path.name}: prediction error: {exc}"
                )
                continue

            actual = df[self.feature_names].apply(pd.to_numeric, errors="coerce")
            predicted = predictions[self.feature_names].apply(
                pd.to_numeric, errors="coerce"
            )
            residual = actual - predicted

            frame = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "pump_id": pump_id_series,
                }
            )
            frame[self.feature_names] = residual
            frame = frame.dropna(subset=["pump_id"] + self.feature_names)
            frame["pump_id"] = frame["pump_id"].astype(int)
            rows.append(frame)

        if not rows:
            return pd.DataFrame(columns=["timestamp", "pump_id", *self.feature_names])

        residuals_df = pd.concat(rows, ignore_index=True)
        return residuals_df

    @staticmethod
    def _extract_pump_ids(df: pd.DataFrame, csv_path: Path) -> pd.Series | None:
        if "pump_id" in df.columns:
            pump_ids = pd.to_numeric(df["pump_id"], errors="coerce")
            return pump_ids

        match = re.search(r"pump[_-]?(\d+)", csv_path.stem, flags=re.IGNORECASE)
        if match:
            pump_id = int(match.group(1))
            return pd.Series(np.full(len(df), pump_id), dtype=float)

        return None

    def _compute_per_feature_stats(self, residuals_df) -> dict:
        """Compute per-feature residual statistics.

        For each output feature:
        - mean_residual: mean of residuals
        - std_residual: std of residuals
        - p95_abs_residual: 95th percentile of |residual|
        - p99_abs_residual: 99th percentile of |residual|
        - z_score_warning: P95 of |z-score| (for per-feature anomaly flagging)
        - z_score_alarm: P99 of |z-score|
        """
        result = {}

        for feature in self.feature_names:
            values = pd.to_numeric(residuals_df[feature], errors="coerce").to_numpy(
                dtype=float
            )
            values = values[~np.isnan(values)]

            if values.size == 0:
                result[feature] = {
                    "mean_residual": 0.0,
                    "std_residual": 0.0,
                    "p95_abs_residual": 0.0,
                    "p99_abs_residual": 0.0,
                    "z_score_warning": 0.0,
                    "z_score_alarm": 0.0,
                }
                continue

            mean = float(np.mean(values))
            std = float(np.std(values))
            abs_values = np.abs(values)

            if std < 1e-12:
                z_abs = np.zeros_like(values)
            else:
                z_abs = np.abs((values - mean) / std)

            result[feature] = {
                "mean_residual": mean,
                "std_residual": std,
                "p95_abs_residual": float(np.percentile(abs_values, 95)),
                "p99_abs_residual": float(np.percentile(abs_values, 99)),
                "z_score_warning": float(np.percentile(z_abs, 95)),
                "z_score_alarm": float(np.percentile(z_abs, 99)),
            }

        return result

    def _compute_mahalanobis(self, residuals_df) -> dict:
        """Compute Mahalanobis distance statistics.

        1. Build residual matrix [N, 13]
        2. Compute mean residual vector [13]
        3. Compute covariance matrix [13, 13]
        4. Regularize: cov + 1e-6 * I (prevent singularity)
        5. Compute inverse covariance
        6. Compute Mahalanobis distance for each sample
        7. Get statistics: mean, std, P95 (warning), P99 (alarm)

        Returns dict with:
            mahalanobis stats, covariance_matrix, inv_covariance, mean_residual_vector
        """
        residual_matrix = residuals_df[self.feature_names].to_numpy(dtype=float)

        if residual_matrix.ndim != 2 or residual_matrix.shape[0] == 0:
            raise ValueError("Residual matrix is empty or malformed")

        mean_residual = np.mean(residual_matrix, axis=0)
        covariance_matrix = np.cov(residual_matrix, rowvar=False)

        if np.isscalar(covariance_matrix):
            covariance_matrix = np.array([[float(covariance_matrix)]], dtype=float)

        covariance_matrix = np.asarray(covariance_matrix, dtype=float)
        if covariance_matrix.shape != (
            len(self.feature_names),
            len(self.feature_names),
        ):
            covariance_matrix = np.eye(len(self.feature_names), dtype=float)

        regularized_cov = covariance_matrix + REG_EPS * np.eye(
            len(self.feature_names), dtype=float
        )
        inv_covariance = np.linalg.pinv(regularized_cov)

        centered = residual_matrix - mean_residual
        mahal_sq = np.einsum("ij,jk,ik->i", centered, inv_covariance, centered)
        mahal_sq = np.clip(mahal_sq, a_min=0.0, a_max=None)
        mahal = np.sqrt(mahal_sq)

        return {
            "mahalanobis": {
                "warning": float(np.percentile(mahal, 95)),
                "alarm": float(np.percentile(mahal, 99)),
                "mean": float(np.mean(mahal)),
                "std": float(np.std(mahal)),
                "p95": float(np.percentile(mahal, 95)),
                "p99": float(np.percentile(mahal, 99)),
            },
            "covariance_matrix": covariance_matrix.tolist(),
            "inverse_covariance_matrix": inv_covariance.tolist(),
            "mean_residual_vector": mean_residual.tolist(),
        }

    def _compute_per_feature_mahalanobis_contribution(
        self,
        residuals_df,
        inv_cov,
        mean_residual,
    ) -> dict:
        """Decompose Mahalanobis distance by feature contribution.

        For each feature i:
            contribution_i = (r_i - μ_i) * Σ_j (Σ^{-1}_{ij} * (r_j - μ_j))

        Total Mahal² = Σ_i contribution_i

        Store per-feature:
        - mahalanobis_contribution_mean: average contribution
        - mahalanobis_contribution_p95: 95th percentile contribution
        - top_correlated_features: top-3 most correlated features (from covariance matrix)
        """
        residual_matrix = residuals_df[self.feature_names].to_numpy(dtype=float)
        centered = residual_matrix - mean_residual

        weighted = centered @ inv_cov.T
        contributions = centered * weighted

        cov = np.cov(residual_matrix, rowvar=False)
        if np.isscalar(cov):
            cov = np.array([[float(cov)]], dtype=float)
        cov = np.asarray(cov, dtype=float)

        std = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
        denom = std[:, None] * std[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 1e-12, cov / denom, 0.0)
        corr = np.nan_to_num(corr)

        feature_payload = {}
        for i, feature in enumerate(self.feature_names):
            corr_row = np.abs(corr[i]).copy()
            corr_row[i] = -np.inf
            top_idx = np.argsort(corr_row)[-3:][::-1]
            top_features = [
                self.feature_names[j] for j in top_idx if np.isfinite(corr_row[j])
            ]

            feature_payload[feature] = {
                "mahalanobis_contribution_mean": float(np.mean(contributions[:, i])),
                "mahalanobis_contribution_p95": float(
                    np.percentile(contributions[:, i], 95)
                ),
                "top_correlated_features": top_features,
            }

        return feature_payload

    @staticmethod
    def _merge_feature_sections(
        base_stats: dict[str, dict], extra_stats: dict[str, dict]
    ) -> dict[str, dict]:
        merged: dict[str, dict] = {}
        for feature, feature_stats in base_stats.items():
            merged[feature] = {**feature_stats, **extra_stats.get(feature, {})}
        return merged

    def calibrate_seasonal(self) -> dict:
        """Compute per-season P95/P99 Mahalanobis thresholds from normal training residuals.

        Uses the northern-hemisphere calendar month mapping from _SEASON_MAP.
        Season/pump combinations with fewer than MIN_SAMPLES_PER_PUMP residuals fall
        back to the corresponding global per-pump block.

        Returns a dict with top-level keys for each season plus metadata.
        """
        residuals_df = self._compute_residuals()
        if residuals_df.empty:
            raise ValueError(
                f"No residuals computed from train path: {self.train_path}"
            )

        # Compute global blocks used as fallback for under-sampled season/pump combos.
        global_per_feature = self._compute_per_feature_stats(residuals_df)
        global_mahal = self._compute_mahalanobis(residuals_df)
        global_contrib = self._compute_per_feature_mahalanobis_contribution(
            residuals_df,
            np.asarray(global_mahal["inverse_covariance_matrix"], dtype=float),
            np.asarray(global_mahal["mean_residual_vector"], dtype=float),
        )
        global_per_feature = self._merge_feature_sections(
            global_per_feature, global_contrib
        )

        global_block = {
            "mahalanobis": global_mahal["mahalanobis"],
            "per_feature": global_per_feature,
            "covariance_matrix": global_mahal["covariance_matrix"],
            "inverse_covariance_matrix": global_mahal["inverse_covariance_matrix"],
            "mean_residual_vector": global_mahal["mean_residual_vector"],
        }

        expected_pumps = {1, 2, 3, 4}
        observed_pumps = {int(pid) for pid in residuals_df["pump_id"].dropna().unique()}
        all_pumps = sorted(expected_pumps | observed_pumps)

        # Build global per-pump fallback blocks (same as calibrate() per-pump logic).
        global_per_pump: dict[str, dict] = {}
        for pump_id in all_pumps:
            pump_residuals = residuals_df[residuals_df["pump_id"] == pump_id]
            pump_id_str = str(int(pump_id))
            n_samples = int(len(pump_residuals))
            if n_samples < MIN_SAMPLES_PER_PUMP:
                global_per_pump[pump_id_str] = {
                    "n_samples": n_samples,
                    "fallback_to_global": True,
                    "mahalanobis": copy.deepcopy(global_mahal["mahalanobis"]),
                    "per_feature": copy.deepcopy(global_per_feature),
                }
                continue
            pump_feature_stats = self._compute_per_feature_stats(pump_residuals)
            pump_mahal = self._compute_mahalanobis(pump_residuals)
            pump_contrib = self._compute_per_feature_mahalanobis_contribution(
                pump_residuals,
                np.asarray(pump_mahal["inverse_covariance_matrix"], dtype=float),
                np.asarray(pump_mahal["mean_residual_vector"], dtype=float),
            )
            global_per_pump[pump_id_str] = {
                "n_samples": n_samples,
                "fallback_to_global": False,
                "mahalanobis": pump_mahal["mahalanobis"],
                "per_feature": self._merge_feature_sections(
                    pump_feature_stats, pump_contrib
                ),
            }

        # Assign season labels to each residual row via the timestamp column.
        try:
            dates = pd.to_datetime(residuals_df["timestamp"], errors="coerce").dt.date
        except Exception:
            dates = pd.Series([None] * len(residuals_df), index=residuals_df.index)

        season_labels = []
        for d in dates:
            try:
                season_labels.append(
                    _month_to_season(d.month)
                    if d is not None and not pd.isnull(d)
                    else None
                )
            except Exception:
                season_labels.append(None)

        residuals_df = residuals_df.copy()
        residuals_df["_season"] = season_labels

        seasonal_out: dict[str, dict] = {}

        for season in _SEASONS:
            season_df = residuals_df[residuals_df["_season"] == season].copy()
            n_season = int(len(season_df))

            if n_season < MIN_SAMPLES_PER_PUMP:
                # Entire season under-sampled — use full global as season-level stats.
                seasonal_global = copy.deepcopy(global_block)
                season_per_pump = {
                    pid_str: {**copy.deepcopy(block), "fallback_to_global": True}
                    for pid_str, block in global_per_pump.items()
                }
            else:
                s_per_feature = self._compute_per_feature_stats(season_df)
                s_mahal = self._compute_mahalanobis(season_df)
                s_contrib = self._compute_per_feature_mahalanobis_contribution(
                    season_df,
                    np.asarray(s_mahal["inverse_covariance_matrix"], dtype=float),
                    np.asarray(s_mahal["mean_residual_vector"], dtype=float),
                )
                s_per_feature = self._merge_feature_sections(s_per_feature, s_contrib)
                seasonal_global = {
                    "mahalanobis": s_mahal["mahalanobis"],
                    "per_feature": s_per_feature,
                    "covariance_matrix": s_mahal["covariance_matrix"],
                    "inverse_covariance_matrix": s_mahal["inverse_covariance_matrix"],
                    "mean_residual_vector": s_mahal["mean_residual_vector"],
                }

                season_per_pump = {}
                for pump_id in all_pumps:
                    pump_id_str = str(int(pump_id))
                    sp_df = season_df[season_df["pump_id"] == pump_id]
                    n_sp = int(len(sp_df))

                    if n_sp < MIN_SAMPLES_PER_PUMP:
                        # Fall back to global per-pump for this (season, pump) combo.
                        fallback = global_per_pump.get(pump_id_str, {})
                        season_per_pump[pump_id_str] = {
                            "n_samples": n_sp,
                            "fallback_to_global": True,
                            "mahalanobis": copy.deepcopy(
                                fallback.get("mahalanobis", global_mahal["mahalanobis"])
                            ),
                            "per_feature": copy.deepcopy(
                                fallback.get("per_feature", global_per_feature)
                            ),
                        }
                        continue

                    sp_feature_stats = self._compute_per_feature_stats(sp_df)
                    sp_mahal = self._compute_mahalanobis(sp_df)
                    sp_contrib = self._compute_per_feature_mahalanobis_contribution(
                        sp_df,
                        np.asarray(sp_mahal["inverse_covariance_matrix"], dtype=float),
                        np.asarray(sp_mahal["mean_residual_vector"], dtype=float),
                    )
                    season_per_pump[pump_id_str] = {
                        "n_samples": n_sp,
                        "fallback_to_global": False,
                        "mahalanobis": sp_mahal["mahalanobis"],
                        "per_feature": self._merge_feature_sections(
                            sp_feature_stats, sp_contrib
                        ),
                    }

            seasonal_out[season] = {
                "n_training_samples": n_season,
                "global": seasonal_global,
                "per_pump": season_per_pump,
            }

        return {
            "description": (
                "Per-season L1 thresholds calibrated from normal training residuals. "
                "Season mapping uses northern-hemisphere calendar months (winter=Dec-Feb, "
                "spring=Mar-May, summer=Jun-Aug, autumn=Sep-Nov)."
            ),
            "method": "seasonal-bucketed per-feature Z-score + Mahalanobis on training residuals",
            "seasons": _SEASONS,
            "feature_names": self.feature_names,
            **{season: seasonal_out[season] for season in _SEASONS},
        }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate anomaly detection thresholds"
    )
    parser.add_argument("--train-path", default="../../data/train/")
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--output", default="model/weights/production_thresholds.json")
    parser.add_argument(
        "--mode",
        choices=["global", "seasonal", "conformal"],
        default="global",
        help=(
            "global (default): write production_thresholds.json; "
            "seasonal: write production_thresholds_seasonal.json; "
            "conformal: add conformal_quantile blocks into existing production_thresholds.json"
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Conformal miscoverage rate (default: 0.05). Only used with --mode conformal.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "conformal":
        from .conformal_calibration import ConformalCalibrator

        calibrator = ConformalCalibrator(
            weights_dir=args.weights_dir,
            train_path=args.train_path,
            thresholds_path=output_path,
            alpha=args.alpha,
        )
        thresholds = calibrator._load_existing_thresholds()
        conformal_blocks = calibrator.calibrate()
        updated = calibrator.inject_into_thresholds(thresholds, conformal_blocks)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2)
        print(f"Conformal quantiles (α={args.alpha}) injected into {output_path}")

    elif args.mode == "seasonal":
        calibrator = ThresholdCalibrator(
            weights_dir=args.weights_dir, train_path=args.train_path
        )
        thresholds = calibrator.calibrate_seasonal()
        seasonal_path = output_path.parent / "production_thresholds_seasonal.json"
        with seasonal_path.open("w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=2)
        print(f"Seasonal thresholds saved to {seasonal_path}")
    else:
        calibrator = ThresholdCalibrator(
            weights_dir=args.weights_dir, train_path=args.train_path
        )
        thresholds = calibrator.calibrate()
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=2)
        print(f"Thresholds saved to {output_path}")


if __name__ == "__main__":
    main()
