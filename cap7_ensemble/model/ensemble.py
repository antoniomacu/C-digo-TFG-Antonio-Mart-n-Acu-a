"""
Ensemble Detector - combines Level 1 (instantaneous) and Level 2 (temporal)
anomaly detection for comprehensive pump health monitoring.
"""

from __future__ import annotations

import bisect
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .scoring import (
    PumpEnsembleResult,
    EnsembleReport,
    Level1Result,
    Level2Result,
    WindowEnsembleResult,
    fuse_statuses,
    compute_severity,
    normalize_severity,
    STATUS_NORMAL,
)
from .level1_detector import Level1Detector
from .level2_detector import Level2Detector


class EnsembleDetector:
    """
    Two-level ensemble anomaly detector for industrial pump monitoring.

    Level 1 catches sudden, instantaneous anomalies (sensor deviates from expected RIGHT NOW).
    Level 2 catches gradual degradation (temporal patterns drift over 3-hour windows).

    The ensemble fuses both signals: when both levels agree on an anomaly, severity escalates.
    When only one level flags, the alert still propagates with context about which level triggered.

    Parameters
    ----------
    level1_weights_dir : path to cond_reg_v2 weights directory
    level2_version_dir : path to bin model version directory
    device : "auto", "cpu", or "cuda" (for Level 1 PyTorch model)
    feature_z_threshold : Z-score threshold for Level 1 per-feature flagging
    top_k_timesteps : Number of top anomalous timesteps to report from Level 1
    """

    def __init__(
        self,
        level1_weights_dir: str | Path | None = None,
        level2_version_dir: str | Path | None = None,
        device: str = "auto",
        feature_z_threshold: float = 3.0,
        top_k_timesteps: int = 10,
    ):
        self._level1 = Level1Detector(
            weights_dir=level1_weights_dir,
            device=device,
            feature_z_threshold=feature_z_threshold,
            top_k_timesteps=top_k_timesteps,
        )
        self._level2 = Level2Detector(version_dir=level2_version_dir)

    def classify(
        self, csv_paths: list[str], df: pd.DataFrame | None = None
    ) -> EnsembleReport:
        """
        Run both detection levels and return a fused ensemble report.

        Parameters
        ----------
        csv_paths : list of str
            Paths to pump-day CSV files. Used by BOTH levels:
            - Level 2 passes these directly to ProductionDetector
            - Level 1 loads them into a DataFrame (if df not provided)
        df : DataFrame, optional
            Pre-loaded DataFrame for Level 1. If None, CSVs are loaded and
            concatenated automatically. Must have DatetimeIndex, pump_id,
            and all sensor columns.

        Returns
        -------
        EnsembleReport
        """
        t_start = time.perf_counter()

        l1_results: list[Level1Result] = []
        l2_results: list[Level2Result] = []
        level1_timing: dict = {}
        level2_timing: dict = {}

        try:
            level1_input_df = df
            if level1_input_df is None:
                level1_input_df = self._load_level1_dataframe(csv_paths)
            l1_results, level1_timing = self._level1.classify(level1_input_df)
        except Exception as exc:
            l1_results = []
            level1_timing = {
                "level1_error": str(exc),
                "level1_seconds": 0.0,
            }

        try:
            l2_results, level2_timing = self._level2.classify(csv_paths)
        except Exception as exc:
            l2_results = []
            level2_timing = {
                "level2_error": str(exc),
                "level2_seconds": 0.0,
            }

        l1_lookup = {(res.pump_id, res.date): res for res in l1_results}
        l2_lookup = {(res.pump_id, res.date): res for res in l2_results}

        all_keys = sorted(set(l1_lookup.keys()) | set(l2_lookup.keys()))

        pump_results: list[PumpEnsembleResult] = []
        for pump_id, date in all_keys:
            l1_result = l1_lookup.get((pump_id, date))
            l2_result = l2_lookup.get((pump_id, date))

            if l1_result is None:
                l1_result = self._build_fallback_level1_result(
                    pump_id=pump_id, date=date
                )

            ensemble_status, reasoning = fuse_statuses(
                l1_result.status,
                l2_result.status if l2_result else None,
            )
            severity = compute_severity(
                l1_result.normalized_severity,
                l2_result.normalized_severity if l2_result else None,
            )

            pump_result = PumpEnsembleResult(
                pump_id=pump_id,
                date=date,
                overall_status=ensemble_status,
                overall_severity=severity,
                level1=l1_result,
                level2=l2_result,
                ensemble_reasoning=reasoning,
            )
            # Propagate per-channel health from L1 to ensemble result
            pump_result.channel_health_summary = l1_result.channel_health_summary

            # Per-window ensemble alignment
            if l2_result and l2_result.window_results:
                pump_result.window_ensemble_results = self._align_windows_to_timesteps(
                    l1_result,
                    l2_result,
                )

            pump_results.append(pump_result)

        overall_status = self._compute_overall_status(pump_results)
        overall_severity = max(
            (pr.overall_severity for pr in pump_results), default=0.0
        )

        timing = {
            **level1_timing,
            **level2_timing,
            "ensemble_seconds": round(time.perf_counter() - t_start, 4),
            "n_pump_days": len(pump_results),
        }

        model_versions = {
            "level1_weights_dir": self._level1.weights_dir,
            "level2_version_dir": self._level2.version_dir,
        }

        return EnsembleReport(
            overall_status=overall_status,
            overall_severity=float(overall_severity),
            pump_results=pump_results,
            timing=timing,
            model_versions=model_versions,
        )

    @staticmethod
    def _load_level1_dataframe(csv_paths: list[str]) -> pd.DataFrame:
        """Load and concatenate pump-day CSVs for Level 1 inference."""
        if not csv_paths:
            raise ValueError("csv_paths must not be empty")

        frames: list[pd.DataFrame] = []
        for csv_path in csv_paths:
            path = Path(csv_path)
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            frame = pd.read_csv(path)
            if "timestamp" not in frame.columns:
                raise ValueError(f"CSV missing required 'timestamp' column: {csv_path}")

            ts = pd.to_datetime(frame["timestamp"], errors="coerce")
            if ts.isna().any():
                raise ValueError(f"CSV has invalid timestamp values: {csv_path}")

            frame = frame.copy()
            frame.index = ts
            frames.append(frame)

        combined = pd.concat(frames, axis=0, ignore_index=False, sort=False)
        return combined.sort_index()

    @staticmethod
    def _build_fallback_level1_result(pump_id: int, date: str) -> Level1Result:
        """Fallback Level 1 result when only Level 2 is available for a pump-day."""
        return Level1Result(
            pump_id=int(pump_id),
            date=str(date),
            status=STATUS_NORMAL,
            day_mean_mahalanobis=0.0,
            day_max_mahalanobis=0.0,
            fraction_above_warning=0.0,
            fraction_above_alarm=0.0,
            normalized_severity=0.0,
            n_timesteps=0,
            top_anomalous_timesteps=[],
            all_timestep_results=[],
        )

    @staticmethod
    def _compute_overall_status(pump_results: list[PumpEnsembleResult]) -> str:
        """Return worst status across all pump-day ensemble results."""
        if not pump_results:
            return STATUS_NORMAL

        status_rank = {
            STATUS_NORMAL: 0,
            "WARNING": 1,
            "ALARM": 2,
        }
        return max(
            (pr.overall_status for pr in pump_results),
            key=lambda status: status_rank.get(status, -1),
        )

    @staticmethod
    def _align_windows_to_timesteps(
        l1_result: Level1Result,
        l2_result: Level2Result,
        past_history: int = 36,
        sampling_minutes: int = 5,
    ) -> list[WindowEnsembleResult]:
        """
        Align Level 2 per-window results with Level 1 per-timestep results.

        For each L2 window, use the L1 timestep nearest the window end timestamp
        (bounded to that window's range), then fuse with the L2 window status.
        This preserves Level 1's instantaneous behavior and avoids 3-hour
        "ghost alarms" from max-over-window aggregation.
        """
        if not l2_result.window_results:
            return []

        l1_timesteps = l1_result.all_timestep_results
        if not l1_timesteps:
            # No L1 data: produce window results with L1 NORMAL fallback.
            results = []
            for wr in l2_result.window_results:
                ens_status, _ = fuse_statuses(STATUS_NORMAL, wr.status)
                l2_sev = normalize_severity(wr.smoothed_mse, l2_result.alarm_threshold)
                severity = compute_severity(0.0, l2_sev)
                results.append(
                    WindowEnsembleResult(
                        window_index=wr.window_index,
                        timestamp=wr.timestamp,
                        level2_smoothed_mse=wr.smoothed_mse,
                        level2_status=wr.status,
                        level1_max_mahalanobis=0.0,
                        level1_status=STATUS_NORMAL,
                        ensemble_status=ens_status,
                        ensemble_severity=severity,
                    )
                )
            return results

        l1_parsed = []
        for ts_result in l1_timesteps:
            try:
                dt = datetime.fromisoformat(ts_result.timestamp)
                l1_parsed.append((dt, ts_result))
            except (ValueError, TypeError):
                continue

        l1_parsed.sort(key=lambda x: x[0])
        l1_times = [pair[0] for pair in l1_parsed]

        window_span = timedelta(minutes=(past_history - 1) * sampling_minutes)

        results = []
        for wr in l2_result.window_results:
            try:
                window_end = datetime.fromisoformat(wr.timestamp)
            except (ValueError, TypeError):
                continue

            window_start = window_end - window_span

            # Select the L1 timestep nearest to window_end, constrained to this window.
            left_idx = bisect.bisect_right(l1_times, window_end) - 1
            right_idx = bisect.bisect_left(l1_times, window_end)

            candidates: list[tuple[datetime, object]] = []
            if 0 <= left_idx < len(l1_parsed):
                left_dt, left_ts = l1_parsed[left_idx]
                if window_start <= left_dt <= window_end:
                    candidates.append((left_dt, left_ts))
            if 0 <= right_idx < len(l1_parsed):
                right_dt, right_ts = l1_parsed[right_idx]
                if window_start <= right_dt <= window_end:
                    candidates.append((right_dt, right_ts))

            if candidates:
                _, chosen_ts = min(
                    candidates,
                    key=lambda pair: abs((pair[0] - window_end).total_seconds()),
                )
                current_mahal = float(chosen_ts.mahalanobis)
                l1_status = chosen_ts.status
            else:
                current_mahal = 0.0
                l1_status = STATUS_NORMAL

            ens_status, _ = fuse_statuses(l1_status, wr.status)

            l1_sev = (
                l1_result.normalized_severity
                * (current_mahal / l1_result.day_max_mahalanobis)
                if l1_result.day_max_mahalanobis > 0
                else 0.0
            )
            l2_sev = normalize_severity(wr.smoothed_mse, l2_result.alarm_threshold)
            severity = compute_severity(l1_sev, l2_sev)

            results.append(
                WindowEnsembleResult(
                    window_index=wr.window_index,
                    timestamp=wr.timestamp,
                    level2_smoothed_mse=wr.smoothed_mse,
                    level2_status=wr.status,
                    level1_max_mahalanobis=current_mahal,
                    level1_status=l1_status,
                    ensemble_status=ens_status,
                    ensemble_severity=severity,
                )
            )

        return results


__all__ = ["EnsembleDetector"]
