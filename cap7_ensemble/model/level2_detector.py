"""
Level 2 Detector - Temporal degradation detection via VAE reconstruction.

Wraps bin.model.inference.ProductionDetector and converts its output
to the standard ensemble data format (Level2Result).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .scoring import Level2Result, WindowResult, classify_status, normalize_severity

# Default location of the trained Level 2 model
DEFAULT_VERSION_DIR = Path(__file__).resolve().parent.parent.parent / "bin" / "final_metrics"

logger = logging.getLogger(__name__)

_SEASONS = ["winter", "spring", "summer", "autumn"]


class Level2Detector:
    """
    Temporal degradation detector using a trained VAE.

    Reconstructs 3-hour sliding windows and detects anomalies
    via reconstruction error (MSE) against calibrated thresholds.

    Parameters
    ----------
    version_dir : str or Path, optional
        Directory containing model_weights.ckpt, norm_params.json,
        production_thresholds.json, hparams.yaml.
        Defaults to bin/final_metrics/.
    """

    def __init__(self, version_dir: str | Path | None = None):
        resolved_version_dir = Path(version_dir) if version_dir else DEFAULT_VERSION_DIR
        self._validate_version_dir(resolved_version_dir)

        # Import here to avoid hard dependency at import time
        try:
            from bin.model.inference import ProductionDetector
        except ImportError as e:
            raise ImportError(
                "Could not import bin.model.inference. Ensure the 'pumps-model' package "
                "is installed or the workspace root is on PYTHONPATH."
            ) from e

        self._detector = ProductionDetector(str(resolved_version_dir))
        self._version_dir = str(resolved_version_dir)

        # Optional seasonal thresholds — per-season per-pump warning/alarm overrides.
        self._active_season: str | None = None
        self._seasonal_l2_thresholds: dict[str, dict[int, dict]] | None = None
        seasonal_path = resolved_version_dir / "production_thresholds_seasonal_l2.json"
        if seasonal_path.exists():
            try:
                with seasonal_path.open("r", encoding="utf-8") as f:
                    seasonal_data = json.load(f)
                self._seasonal_l2_thresholds = {}
                for season in _SEASONS:
                    season_block = seasonal_data.get(season, {})
                    pump_map: dict[int, dict] = {}
                    for pid_str, pump_block in season_block.get("per_pump", {}).items():
                        try:
                            pid_int = int(pid_str)
                        except (TypeError, ValueError):
                            continue
                        pump_map[pid_int] = pump_block
                    self._seasonal_l2_thresholds[season] = pump_map
                logger.debug("Loaded seasonal L2 thresholds from %s", seasonal_path)
            except Exception as exc:
                logger.warning("Failed to load seasonal L2 thresholds from %s: %s", seasonal_path, exc)
                self._seasonal_l2_thresholds = None

    @property
    def version_dir(self) -> str:
        return self._version_dir

    def set_season(self, season: str) -> None:
        """Set the active season for threshold selection.

        If no seasonal threshold file was loaded, this is a no-op (logs warning once).
        """
        if season not in _SEASONS:
            raise ValueError(f"Invalid season '{season}'. Must be one of {_SEASONS}")
        if self._seasonal_l2_thresholds is None:
            logger.warning(
                "set_season('%s') called but no seasonal L2 thresholds loaded — using production thresholds",
                season,
            )
            return
        self._active_season = season

    @staticmethod
    def _validate_version_dir(version_dir: Path) -> None:
        if not version_dir.exists() or not version_dir.is_dir():
            raise FileNotFoundError(f"Level 2 version_dir not found: {version_dir}")

        required_files = [
            "model_weights.ckpt",
            "norm_params.json",
            "production_thresholds.json",
            "hparams.yaml",
        ]
        missing = [name for name in required_files if not (version_dir / name).exists()]
        if missing:
            missing_joined = ", ".join(missing)
            raise FileNotFoundError(
                f"Level 2 version_dir is missing required files: {missing_joined} "
                f"(version_dir={version_dir})"
            )

    @staticmethod
    def _resolve_status(
        day_mse: float,
        warning_threshold: float,
        alarm_threshold: float,
        fallback_status: str,
    ) -> str:
        """Use ensemble scoring helper, with safe fallback for signature mismatches."""
        try:
            return classify_status(day_mse, warning_threshold, alarm_threshold)
        except TypeError:
            try:
                return classify_status(
                    day_error_mse=day_mse,
                    warning_threshold=warning_threshold,
                    alarm_threshold=alarm_threshold,
                )
            except Exception:
                return fallback_status
        except Exception:
            return fallback_status

    def classify(self, csv_paths: list[str]) -> tuple[list[Level2Result], dict]:
        """
        Run Level 2 detection on pump-day CSV files.

        Parameters
        ----------
        csv_paths : list of str
            Paths to CSV files (one per pump-day).

        Returns
        -------
        results : list of Level2Result
            One result per successfully processed pump.
        timing : dict
            Wall-clock timing information.
        """
        t_start = time.perf_counter()

        raw = self._detector.classify(csv_paths)

        t_total = time.perf_counter() - t_start

        raw_timing = raw.get("timing", {}) if isinstance(raw, dict) else {}
        timing = {
            **raw_timing,
            "level2_seconds": round(t_total, 4),
        }

        # Upstream detector-level error payload.
        if raw.get("status") == "ERROR":
            timing["detector_status"] = "ERROR"
            if "message" in raw:
                timing["detector_message"] = str(raw["message"])
            return [], timing

        results: list[Level2Result] = []
        for pr in raw.get("pump_results", []):
            # Pump-level errors are skipped so callers only receive valid Level2Result objects.
            if pr.get("status") == "ERROR":
                continue

            try:
                alarm_threshold = float(pr["alarm_threshold"])
                warning_threshold = float(pr["warning_threshold"])
                day_mse = float(pr["day_error_mse"])
            except (KeyError, TypeError, ValueError):
                continue

            # Apply seasonal threshold override when active and not flagged as fallback.
            if self._active_season is not None and self._seasonal_l2_thresholds is not None:
                pump_id_for_season = int(pr.get("pump_id", -1))
                season_pump_map = self._seasonal_l2_thresholds.get(self._active_season, {})
                override = season_pump_map.get(pump_id_for_season)
                if override is not None and not override.get("fallback_to_global", True):
                    warning_threshold = float(override["warning"])
                    alarm_threshold = float(override["alarm"])

            status = self._resolve_status(
                day_mse=day_mse,
                warning_threshold=warning_threshold,
                alarm_threshold=alarm_threshold,
                fallback_status=str(pr.get("status", "ERROR")),
            )

            # Parse per-window results (empty list if not present = backward compatible)
            window_results_raw = pr.get("window_results", [])
            window_results = []
            for wr in window_results_raw:
                try:
                    window_results.append(
                        WindowResult(
                            window_index=int(wr["window_index"]),
                            timestamp=str(wr["timestamp"]),
                            raw_mse=float(wr["raw_mse"]),
                            smoothed_mse=float(wr["smoothed_mse"]),
                            status=str(wr["status"]),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue

            results.append(
                Level2Result(
                    pump_id=int(pr["pump_id"]),
                    date=str(pr["date"]),
                    status=status,
                    day_error_mse=day_mse,
                    warning_threshold=warning_threshold,
                    alarm_threshold=alarm_threshold,
                    normalized_severity=normalize_severity(day_mse, alarm_threshold),
                    n_samples=int(pr.get("n_samples", 0)),
                    window_results=window_results,
                    # Seasonal override applies to day-level thresholds only;
                    # window-level thresholds keep the ProductionDetector values.
                    window_warning_threshold=float(pr.get("window_warning_threshold", 0.0)),
                    window_alarm_threshold=float(pr.get("window_alarm_threshold", 0.0)),
                    smoothing_alpha=float(pr.get("smoothing_alpha", 0.3)),
                    fraction_windows_warning=float(pr.get("fraction_windows_warning", 0.0)),
                    fraction_windows_alarm=float(pr.get("fraction_windows_alarm", 0.0)),
                )
            )

        return results, timing
