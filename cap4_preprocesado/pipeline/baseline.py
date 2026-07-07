from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import load_csv
from .period_detector import detect_periods, get_steady_state_mask
from .system_config import SystemConfig


@dataclass
class PumpBaseline:
    # Operating state statistics (median and std of per-day means)
    speed_median: float
    speed_std: float
    current_median: float
    current_std: float
    pressure_on_median: float
    pressure_on_std: float
    pressure_off_median: float
    pressure_off_std: float
    pressure_off_var_median: float
    pressure_off_var_std: float
    pressure_response_hires_median: float
    pressure_response_hires_std: float
    pressure_response_lores_median: float
    pressure_response_lores_std: float
    current_off_median: float
    current_off_std: float
    # Per-column temperature baselines
    temp_medians: dict[str, float]
    temp_stds: dict[str, float]
    temp_var_medians: dict[str, float]
    temp_var_stds: dict[str, float]
    temp_var_q1: dict[str, float]
    # Per-column proximitor baselines
    prox_medians: dict[str, float]
    prox_stds: dict[str, float]
    prox_maxes: dict[str, float]
    # Correlation baseline
    current_speed_ratio_median: float
    current_speed_ratio_std: float


def _to_array(values: list[float]) -> np.ndarray:
    if not values:
        return np.array([], dtype=float)
    return np.asarray(values, dtype=float)


def _robust_median_std(values: list[float], std_floor: float = 0.0) -> tuple[float, float]:
    arr = _to_array(values)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), max(std_floor, 0.0)

    median = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - median)))
    robust_std = 1.4826 * mad
    robust_std = max(robust_std, std_floor)
    return median, float(robust_std)


def compute_baselines(
    file_list: list[tuple[date, int, Path]],
    cfg: SystemConfig,
    exclude_files: set[Path] | None = None,
) -> dict[int, PumpBaseline]:
    """Compute per-pump robust baseline statistics from historical files."""
    per_pump: dict[int, dict[str, object]] = {}
    speed_col = cfg.columns.speed_col
    current_col = cfg.columns.current_col
    pressure_col = cfg.columns.pressure_col
    pressure_off_col = cfg.columns.pressure_off_col
    temp_cols = cfg.columns.temp_cols
    prox_cols = cfg.columns.prox_cols
    all_baseline_cols = temp_cols + cfg.columns.baseline_extra_cols
    min_period_duration = cfg.thresholds.min_period_duration

    def _get_bucket(pump_id: int) -> dict[str, object]:
        if pump_id not in per_pump:
            per_pump[pump_id] = {
                "speed": [],
                "current": [],
                "pressure_on": [],
                "pressure_off": [],
                "pressure_off_var": [],
                "pressure_response_hires": [],
                "pressure_response_lores": [],
                "current_off": [],
                "ratio": [],
                "temp": {col: [] for col in all_baseline_cols},
                "temp_var": {col: [] for col in all_baseline_cols},
                "prox_mean": {col: [] for col in prox_cols},
                "prox_max": {col: [] for col in prox_cols},
            }
        return per_pump[pump_id]

    total = len(file_list)
    for idx, (_, pump_id, path) in enumerate(file_list, start=1):
        if exclude_files and path in exclude_files:
            continue

        if idx % 100 == 0:
            print(f"[baseline] processed {idx}/{total} files")

        df = load_csv(path, cfg)
        timestamp_col = cfg.columns.timestamp_col
        if df is None or df.empty or timestamp_col not in df.columns:
            continue

        sampling_interval = float(df.attrs.get("sampling_interval_sec", 5.0))
        if not np.isfinite(sampling_interval) or sampling_interval <= 0:
            sampling_interval = 5.0

        periods = detect_periods(df, sampling_interval_sec=sampling_interval, cfg=cfg)
        if not periods:
            continue

        bucket = _get_bucket(pump_id)
        ts = df[timestamp_col]

        near_any_period = pd.Series(False, index=df.index)
        for period in periods:
            near_start = period.start - pd.Timedelta(minutes=5)
            near_end = period.end + pd.Timedelta(minutes=5)
            near_any_period = near_any_period | ((ts >= near_start) & (ts <= near_end))

        speed_zero_mask = (
            df.get(speed_col, pd.Series(np.nan, index=df.index)).fillna(np.nan) == 0
        )
        off_mask = speed_zero_mask & (~near_any_period)
        if off_mask.any():
            if pressure_off_col is not None:
                pressure_off_val = (
                    float(np.nanmean(df.loc[off_mask, pressure_off_col]))
                    if pressure_off_col in df.columns
                    else float("nan")
                )
                if np.isfinite(pressure_off_val):
                    bucket["pressure_off"].append(pressure_off_val)  # type: ignore[index]

            current_off_val = (
                float(np.nanmean(df.loc[off_mask, current_col]))
                if current_col in df.columns
                else float("nan")
            )
            if np.isfinite(current_off_val):
                bucket["current_off"].append(current_off_val)  # type: ignore[index]

        for period in periods:
            p_off_before = float("nan")
            if pressure_off_col is not None and pressure_off_col in df.columns:
                before_start = period.start - pd.Timedelta(minutes=30)
                after_end = period.end + pd.Timedelta(minutes=30)
                off_window_mask = (
                    (((ts >= before_start) & (ts < period.start)) | ((ts > period.end) & (ts <= after_end)))
                    & speed_zero_mask
                )
                if int(off_window_mask.sum()) >= 2:
                    off_window_std = float(np.nanstd(df.loc[off_window_mask, pressure_off_col]))
                    if np.isfinite(off_window_std):
                        bucket["pressure_off_var"].append(off_window_std)  # type: ignore[index]

                # Pressure before start (for response tracking)
                before_off_mask = (ts >= before_start) & (ts < period.start) & speed_zero_mask
                if int(before_off_mask.sum()) >= 2:
                    p_off_before = float(np.nanmean(df.loc[before_off_mask, pressure_off_col]))

            if period.duration_seconds < min_period_duration:
                continue

            steady_mask = get_steady_state_mask(df, period, cfg)
            if int(steady_mask.sum()) == 0:
                continue

            steady_df = df.loc[steady_mask]

            speed_mean = (
                float(np.nanmean(steady_df[speed_col]))
                if speed_col in steady_df.columns
                else float("nan")
            )
            current_mean = (
                float(np.nanmean(steady_df[current_col]))
                if current_col in steady_df.columns
                else float("nan")
            )
            pressure_on_mean = float("nan")
            if pressure_col is not None:
                pressure_on_mean = (
                    float(np.nanmean(steady_df[pressure_col]))
                    if pressure_col in steady_df.columns
                    else float("nan")
                )

            if np.isfinite(speed_mean):
                bucket["speed"].append(speed_mean)  # type: ignore[index]
            if np.isfinite(current_mean):
                bucket["current"].append(current_mean)  # type: ignore[index]
            if np.isfinite(pressure_on_mean):
                bucket["pressure_on"].append(pressure_on_mean)  # type: ignore[index]
            if np.isfinite(pressure_on_mean) and np.isfinite(p_off_before):
                resp_key = "pressure_response_hires" if sampling_interval <= 30 else "pressure_response_lores"
                bucket[resp_key].append(pressure_on_mean - p_off_before)  # type: ignore[index]

            if np.isfinite(speed_mean) and abs(speed_mean) > 1e-6 and np.isfinite(current_mean):
                bucket["ratio"].append(current_mean / speed_mean)  # type: ignore[index]

            for col in all_baseline_cols:
                if col not in steady_df.columns:
                    continue
                col_mean = float(np.nanmean(steady_df[col]))
                col_std = float(np.nanstd(steady_df[col]))
                if np.isfinite(col_mean):
                    bucket["temp"][col].append(col_mean)  # type: ignore[index]
                if np.isfinite(col_std):
                    bucket["temp_var"][col].append(col_std)  # type: ignore[index]

            for col in prox_cols:
                if col not in steady_df.columns:
                    continue
                col_mean = float(np.nanmean(steady_df[col]))
                col_max = float(np.nanmax(steady_df[col]))
                if np.isfinite(col_mean):
                    bucket["prox_mean"][col].append(col_mean)  # type: ignore[index]
                if np.isfinite(col_max):
                    bucket["prox_max"][col].append(col_max)  # type: ignore[index]

    baselines: dict[int, PumpBaseline] = {}

    for pump_id, bucket in per_pump.items():
        speed_median, speed_std = _robust_median_std(bucket["speed"])  # type: ignore[arg-type]
        current_median, current_std = _robust_median_std(bucket["current"])  # type: ignore[arg-type]

        if pressure_col is not None:
            pressure_on_median, pressure_on_std = _robust_median_std(bucket["pressure_on"])  # type: ignore[arg-type]
            pressure_off_median, pressure_off_std = _robust_median_std(bucket["pressure_off"])  # type: ignore[arg-type]
            pressure_off_var_median, pressure_off_var_std = _robust_median_std(bucket["pressure_off_var"])  # type: ignore[arg-type]
            pressure_response_hires_median, pressure_response_hires_std = _robust_median_std(
                bucket["pressure_response_hires"], std_floor=0.5
            )
            pressure_response_lores_median, pressure_response_lores_std = _robust_median_std(
                bucket["pressure_response_lores"], std_floor=0.5
            )
        else:
            pressure_on_median = float("nan")
            pressure_on_std = float("nan")
            pressure_off_median = float("nan")
            pressure_off_std = float("nan")
            pressure_off_var_median = float("nan")
            pressure_off_var_std = float("nan")
            pressure_response_hires_median = float("nan")
            pressure_response_hires_std = float("nan")
            pressure_response_lores_median = float("nan")
            pressure_response_lores_std = float("nan")

        current_off_median, current_off_std = _robust_median_std(bucket["current_off"])  # type: ignore[arg-type]
        ratio_median, ratio_std = _robust_median_std(bucket["ratio"])  # type: ignore[arg-type]

        temp_medians: dict[str, float] = {}
        temp_stds: dict[str, float] = {}
        temp_var_medians: dict[str, float] = {}
        temp_var_stds: dict[str, float] = {}
        temp_var_q1: dict[str, float] = {}
        for col in all_baseline_cols:
            median, std = _robust_median_std(bucket["temp"][col], std_floor=0.01)  # type: ignore[index]
            var_median, var_std = _robust_median_std(bucket["temp_var"][col], std_floor=0.01)  # type: ignore[index]
            temp_medians[col] = median
            temp_stds[col] = std
            temp_var_medians[col] = var_median
            temp_var_stds[col] = var_std
            # 25th percentile of variability — resistant to upward contamination
            var_arr = _to_array(bucket["temp_var"][col])  # type: ignore[index]
            var_arr = var_arr[np.isfinite(var_arr)]
            temp_var_q1[col] = float(np.nanpercentile(var_arr, 25)) if var_arr.size > 0 else float("nan")

        prox_medians: dict[str, float] = {}
        prox_stds: dict[str, float] = {}
        prox_maxes: dict[str, float] = {}
        for col in prox_cols:
            median, std = _robust_median_std(bucket["prox_mean"][col], std_floor=0.005)  # type: ignore[index]
            prox_medians[col] = median
            prox_stds[col] = std

            max_arr = _to_array(bucket["prox_max"][col])  # type: ignore[index]
            max_arr = max_arr[np.isfinite(max_arr)]
            if max_arr.size == 0:
                prox_maxes[col] = float("nan")
            else:
                prox_maxes[col] = float(np.nanpercentile(max_arr, 95))

        baselines[pump_id] = PumpBaseline(
            speed_median=speed_median,
            speed_std=speed_std,
            current_median=current_median,
            current_std=current_std,
            pressure_on_median=pressure_on_median,
            pressure_on_std=pressure_on_std,
            pressure_off_median=pressure_off_median,
            pressure_off_std=pressure_off_std,
            pressure_off_var_median=pressure_off_var_median,
            pressure_off_var_std=pressure_off_var_std,
            pressure_response_hires_median=pressure_response_hires_median,
            pressure_response_hires_std=pressure_response_hires_std,
            pressure_response_lores_median=pressure_response_lores_median,
            pressure_response_lores_std=pressure_response_lores_std,
            current_off_median=current_off_median,
            current_off_std=current_off_std,
            temp_medians=temp_medians,
            temp_stds=temp_stds,
            temp_var_medians=temp_var_medians,
            temp_var_stds=temp_var_stds,
            temp_var_q1=temp_var_q1,
            prox_medians=prox_medians,
            prox_stds=prox_stds,
            prox_maxes=prox_maxes,
            current_speed_ratio_median=ratio_median,
            current_speed_ratio_std=ratio_std,
        )

    return baselines
