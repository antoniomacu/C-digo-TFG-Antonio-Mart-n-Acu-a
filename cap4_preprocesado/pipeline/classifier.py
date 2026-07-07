from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .baseline import PumpBaseline
from .period_detector import OperationPeriod, get_steady_state_mask
from .system_config import SystemConfig


@dataclass
class ClassificationResult:
    classification: str  # "normal" or "abnormal"
    reasons: list[str]


DetectorFn = Callable[[pd.DataFrame, OperationPeriod, PumpBaseline, SystemConfig], tuple[bool, str]]


def _safe_sigma_threshold(std: float, min_std: float = 1e-6) -> float:
    if not np.isfinite(std):
        return min_std
    return max(float(std), min_std)


def _steady_df(df: pd.DataFrame, period: OperationPeriod, cfg: SystemConfig) -> pd.DataFrame:
    steady_mask = get_steady_state_mask(df, period, cfg)
    if not steady_mask.any():
        return df.iloc[0:0]
    return df.loc[steady_mask]


def _check_speed_stability(
    df: pd.DataFrame,
    period: OperationPeriod,
    baseline: PumpBaseline,
    cfg: SystemConfig,
) -> tuple[bool, str]:
    speed_col = cfg.columns.speed_col
    steady = _steady_df(df, period, cfg)
    if steady.empty or speed_col not in steady.columns:
        return False, ""

    speed_vals = steady[speed_col].to_numpy(dtype=float)
    if np.isfinite(speed_vals).sum() < 2:
        return False, ""

    speed_std = float(np.nanstd(speed_vals))

    baseline_std = _safe_sigma_threshold(baseline.speed_std)
    unstable = speed_std > max(cfg.thresholds.anomaly_sigma * baseline_std, 50.0)

    if unstable:
        return True, "speed_stability"
    return False, ""


def _check_current_anomaly(
    df: pd.DataFrame,
    period: OperationPeriod,
    baseline: PumpBaseline,
    cfg: SystemConfig,
) -> tuple[bool, str]:
    current_col = cfg.columns.current_col
    steady = _steady_df(df, period, cfg)
    if steady.empty or current_col not in steady.columns:
        return False, ""

    current_vals = steady[current_col].to_numpy(dtype=float)
    if np.isfinite(current_vals).sum() < 2:
        return False, ""

    current_mean = float(np.nanmean(current_vals))
    current_std = float(np.nanstd(current_vals))

    baseline_std = _safe_sigma_threshold(baseline.current_std)
    mean_anomaly = abs(current_mean - baseline.current_median) > cfg.thresholds.anomaly_sigma * baseline_std
    variability_anomaly = current_std > cfg.thresholds.anomaly_sigma * baseline_std

    if mean_anomaly or variability_anomaly:
        return True, "current_anomaly"
    return False, ""


def _check_current_speed_ratio(
    df: pd.DataFrame,
    period: OperationPeriod,
    baseline: PumpBaseline,
    cfg: SystemConfig,
) -> tuple[bool, str]:
    current_col = cfg.columns.current_col
    speed_col = cfg.columns.speed_col
    steady = _steady_df(df, period, cfg)
    if steady.empty or current_col not in steady.columns or speed_col not in steady.columns:
        return False, ""

    speed = steady[speed_col].to_numpy(dtype=float)
    current = steady[current_col].to_numpy(dtype=float)

    valid = np.isfinite(speed) & np.isfinite(current) & (np.abs(speed) > 1e-6)
    if valid.sum() < 2:
        return False, ""

    ratio_mean = float(np.nanmean(current[valid] / speed[valid]))
    baseline_std = _safe_sigma_threshold(baseline.current_speed_ratio_std)

    if abs(ratio_mean - baseline.current_speed_ratio_median) > 3.0 * baseline_std:
        return True, "current_speed_ratio"
    return False, ""


def _check_pressure_on(
    df: pd.DataFrame,
    period: OperationPeriod,
    baseline: PumpBaseline,
    cfg: SystemConfig,
) -> tuple[bool, str]:
    pressure_col = cfg.columns.pressure_col
    if pressure_col is None:
        return False, ""

    steady = _steady_df(df, period, cfg)
    if steady.empty or pressure_col not in steady.columns:
        return False, ""

    pressure_vals = steady[pressure_col].to_numpy(dtype=float)
    if np.isfinite(pressure_vals).sum() < 2:
        return False, ""

    pressure_mean = float(np.nanmean(pressure_vals))
    baseline_std = _safe_sigma_threshold(baseline.pressure_on_std)

    if abs(pressure_mean - baseline.pressure_on_median) > cfg.thresholds.anomaly_sigma * baseline_std:
        return True, "pressure_on"
    return False, ""


def _check_pressure_off(
    df: pd.DataFrame,
    period: OperationPeriod,
    baseline: PumpBaseline,
    cfg: SystemConfig,
) -> tuple[bool, str]:
    pressure_col = cfg.columns.pressure_off_col
    speed_col = cfg.columns.speed_col
    timestamp_col = cfg.columns.timestamp_col

    if pressure_col is None:
        return False, ""

    if df.empty or timestamp_col not in df.columns or pressure_col not in df.columns or speed_col not in df.columns:
        return False, ""

    ts = df[timestamp_col]
    before_start = period.start - pd.Timedelta(minutes=30)
    after_end = period.end + pd.Timedelta(minutes=30)

    before_mask = (ts >= before_start) & (ts < period.start)
    after_mask = (ts > period.end) & (ts <= after_end)
    window_mask = before_mask | after_mask

    before_off_mask = before_mask & (df[speed_col].fillna(np.nan) == 0)
    after_off_mask = after_mask & (df[speed_col].fillna(np.nan) == 0)
    off_mask = window_mask & (df[speed_col].fillna(np.nan) == 0)

    if int(before_off_mask.sum()) < 2:
        return False, ""

    before_mean_pressure_off = float(np.nanmean(df.loc[before_off_mask, pressure_col]))
    if not np.isfinite(before_mean_pressure_off):
        return False, ""

    baseline_std = _safe_sigma_threshold(baseline.pressure_off_std)
    before_deviation = abs(before_mean_pressure_off - baseline.pressure_off_median)
    mean_shift_anomaly = before_deviation > cfg.thresholds.anomaly_sigma * baseline_std
    absolute_drop_anomaly = before_deviation > cfg.thresholds.pressure_off_absolute_threshold

    if int(off_mask.sum()) >= 2:
        off_window_std = float(np.nanstd(df.loc[off_mask, pressure_col]))
    else:
        off_window_std = float("nan")

    variability_threshold = max(
        baseline.pressure_off_var_median + _safe_sigma_threshold(baseline.pressure_off_var_std),
        3.0,
    )
    variability_anomaly = np.isfinite(off_window_std) and off_window_std > variability_threshold

    if int(after_off_mask.sum()) >= 2:
        after_mean_pressure_off = float(np.nanmean(df.loc[after_off_mask, pressure_col]))
    else:
        after_mean_pressure_off = float("nan")

    rebound_anomaly = False
    if np.isfinite(after_mean_pressure_off):
        rebound_delta = after_mean_pressure_off - before_mean_pressure_off
        after_near_baseline = abs(after_mean_pressure_off - baseline.pressure_off_median) <= 1.5 * baseline_std
        rebound_anomaly = rebound_delta > cfg.thresholds.pressure_off_rebound_threshold and after_near_baseline

    # High-confidence detection: deviation is so large that rebound/variability
    # evidence is unnecessary (e.g., multi-pump systems where header stays pressurized)
    high_confidence = before_deviation > 2 * cfg.thresholds.anomaly_sigma * baseline_std

    if mean_shift_anomaly and absolute_drop_anomaly and (variability_anomaly or rebound_anomaly or high_confidence):
        return True, "pressure_off"
    return False, ""


def _check_pressure_off_variability(
    df: pd.DataFrame,
    period: OperationPeriod,
    baseline: PumpBaseline,
    cfg: SystemConfig,
) -> tuple[bool, str]:
    pressure_col = cfg.columns.pressure_off_col
    speed_col = cfg.columns.speed_col
    timestamp_col = cfg.columns.timestamp_col

    if pressure_col is None:
        return False, ""

    if df.empty or timestamp_col not in df.columns or pressure_col not in df.columns or speed_col not in df.columns:
        return False, ""

    ts = df[timestamp_col]
    before_start = period.start - pd.Timedelta(minutes=30)
    after_end = period.end + pd.Timedelta(minutes=30)

    off_window_mask = (
        (((ts >= before_start) & (ts < period.start)) | ((ts > period.end) & (ts <= after_end)))
        & (df[speed_col].fillna(np.nan) == 0)
    )
    if int(off_window_mask.sum()) < 2:
        return False, ""

    off_variability = float(np.nanstd(df.loc[off_window_mask, pressure_col]))
    if not np.isfinite(off_variability):
        return False, ""

    baseline_median = baseline.pressure_off_var_median
    baseline_std = _safe_sigma_threshold(baseline.pressure_off_var_std)

    if np.isfinite(baseline_median) and off_variability > baseline_median + 3.0 * baseline_std:
        return True, "pressure_off_variability"
    return False, ""


def _check_pressure_off_extended(
    df: pd.DataFrame,
    period: OperationPeriod,
    baseline: PumpBaseline,
    cfg: SystemConfig,
) -> tuple[bool, str]:
    """Detect day-wide OFF-state pressure anomalies using an extended pre-start window.

    The standard pressure_off detector uses a ±30-minute window, which can miss
    slow pressure drifts that develop over hours. This detector uses all OFF-state
    data from the start of the day (or previous period end) to the current period
    start, then requires rebound evidence after the pump operates.
    """
    pressure_col = cfg.columns.pressure_off_col
    speed_col = cfg.columns.speed_col
    timestamp_col = cfg.columns.timestamp_col

    if pressure_col is None:
        return False, ""

    if (
        df.empty
        or timestamp_col not in df.columns
        or pressure_col not in df.columns
        or speed_col not in df.columns
    ):
        return False, ""

    ts = df[timestamp_col]
    speed = df[speed_col]

    # --- Extended pre-start OFF window ---
    # Use all OFF-state data from the start of the file (or 5 min after
    # a preceding detected period) up to the current period start.
    # We detect preceding periods by scanning for speed transitions to
    # avoid calling detect_periods inside a detector.
    on_mask = speed.fillna(0) > cfg.thresholds.speed_on_threshold
    pre_period_mask = ts < period.start

    # Find the last ON->OFF transition before this period
    pre_on = on_mask & pre_period_mask
    if pre_on.any():
        last_on_idx = pre_on[::-1].idxmax()
        last_on_ts = ts.iloc[last_on_idx]
        ext_start = last_on_ts + pd.Timedelta(minutes=5)
    else:
        ext_start = ts.iloc[0]

    ext_before_mask = (ts >= ext_start) & (ts < period.start) & (speed.fillna(0) == 0)
    n_ext = int(ext_before_mask.sum())

    # Require at least 1 hour of OFF data (12 points at 5-min, 720 at 5-sec)
    sampling_interval = float(df.attrs.get("sampling_interval_sec", 5.0))
    if not np.isfinite(sampling_interval) or sampling_interval <= 0:
        sampling_interval = 5.0
    min_points = max(int(3600 / sampling_interval), 6)

    if n_ext < min_points:
        return False, ""

    ext_pressure = df.loc[ext_before_mask, pressure_col]
    ext_mean = float(np.nanmean(ext_pressure))
    if not np.isfinite(ext_mean):
        return False, ""

    # --- Check 1: Extended mean deviation ---
    baseline_std = _safe_sigma_threshold(baseline.pressure_off_std)
    ext_deviation = abs(ext_mean - baseline.pressure_off_median)

    sigma_k = cfg.thresholds.pressure_off_extended_sigma
    stat_shift = ext_deviation > sigma_k * baseline_std
    abs_floor = ext_deviation > 4.5  # slightly below the standard 5.0 bar

    if not (stat_shift and abs_floor):
        return False, ""

    # --- Check 2: Rebound evidence ---
    # After the pump operates, does pressure recover toward baseline?
    after_end = period.end + pd.Timedelta(minutes=30)
    after_off_mask = (
        (ts > period.end) & (ts <= after_end) & (speed.fillna(0) == 0)
    )
    if int(after_off_mask.sum()) < 2:
        return False, ""

    after_mean = float(np.nanmean(df.loc[after_off_mask, pressure_col]))
    if not np.isfinite(after_mean):
        return False, ""

    rebound_delta = after_mean - ext_mean
    after_near_baseline = (
        abs(after_mean - baseline.pressure_off_median) <= 2.0 * baseline_std
    )

    if rebound_delta > 3.5 and after_near_baseline:
        return True, "pressure_off_extended"

    # High-confidence: when the extended deviation is overwhelming,
    # fire without rebound (handles multi-pump header back-pressure)
    if ext_deviation > 2 * sigma_k * baseline_std:
        return True, "pressure_off_extended"

    return False, ""


def _check_temperatures(
    df: pd.DataFrame,
    period: OperationPeriod,
    baseline: PumpBaseline,
    cfg: SystemConfig,
) -> tuple[bool, str]:
    steady = _steady_df(df, period, cfg)
    if steady.empty:
        return False, ""

    elevated: list[str] = []
    for col in cfg.columns.temp_cols:
        if col not in steady.columns:
            continue
        temp_mean = float(np.nanmean(steady[col]))
        if not np.isfinite(temp_mean):
            continue
        median = baseline.temp_medians.get(col, float("nan"))
        std = _safe_sigma_threshold(baseline.temp_stds.get(col, float("nan")), min_std=0.01)
        if np.isfinite(median) and temp_mean > median + cfg.thresholds.anomaly_sigma * std:
            elevated.append(col)

    reasons: list[str] = []
    if elevated:
        reasons.append("temp_elevated:" + ",".join(elevated))

    if reasons:
        return True, ";".join(reasons)
    return False, ""


def _check_vibration(
    df: pd.DataFrame,
    period: OperationPeriod,
    baseline: PumpBaseline,
    cfg: SystemConfig,
) -> tuple[bool, str]:
    steady = _steady_df(df, period, cfg)
    if steady.empty:
        return False, ""

    triggered: list[str] = []
    for col in cfg.columns.prox_cols:
        if col not in steady.columns:
            continue

        vals = steady[col].to_numpy(dtype=float)
        if np.isfinite(vals).sum() < 2:
            continue

        mean_val = float(np.nanmean(vals))
        max_val = float(np.nanmax(vals))

        median = baseline.prox_medians.get(col, float("nan"))
        std = _safe_sigma_threshold(baseline.prox_stds.get(col, float("nan")), min_std=0.005)
        baseline_max = baseline.prox_maxes.get(col, float("nan"))

        spike = np.isfinite(baseline_max) and max_val > cfg.thresholds.vibration_spike_factor * baseline_max
        elevated_mean = np.isfinite(median) and mean_val > median + cfg.thresholds.anomaly_sigma * std

        if spike:
            triggered.append(f"{col}_spike")
        if elevated_mean:
            triggered.append(f"{col}_mean")

    if triggered:
        return True, "vibration:" + ",".join(triggered)
    return False, ""


def _check_off_state_current(
    df: pd.DataFrame,
    period: OperationPeriod,
    baseline: PumpBaseline,
    cfg: SystemConfig,
) -> tuple[bool, str]:
    timestamp_col = cfg.columns.timestamp_col
    current_col = cfg.columns.current_col
    speed_col = cfg.columns.speed_col

    if df.empty or timestamp_col not in df.columns or current_col not in df.columns or speed_col not in df.columns:
        return False, ""

    ts = df[timestamp_col]
    before_start = period.start - pd.Timedelta(minutes=30)
    after_end = period.end + pd.Timedelta(minutes=30)

    before_mask = (ts >= before_start) & (ts < period.start)
    after_mask = (ts > period.end) & (ts <= after_end)
    window_mask = before_mask | after_mask

    off_mask = window_mask & (df[speed_col].fillna(np.nan) == 0)
    if int(off_mask.sum()) < 2:
        return False, ""

    # Use median to be robust against brief motor-energisation spikes
    # (inrush current while speed is still reported as 0).
    idle_current = float(np.nanmedian(df.loc[off_mask, current_col]))
    baseline_std = _safe_sigma_threshold(baseline.current_off_std)

    if abs(idle_current - baseline.current_off_median) > cfg.thresholds.off_state_sigma * baseline_std:
        return True, "off_state_current"
    return False, ""


def _check_pressure_response(
    df: pd.DataFrame,
    period: OperationPeriod,
    baseline: PumpBaseline,
    cfg: SystemConfig,
) -> tuple[bool, str]:
    """Flag periods where pressure doesn't respond to pump operation as expected.

    Compares the observed pressure transition (OFF->ON) against the historical
    baseline. A healthy pump produces a predictable pressure delta; deviation
    indicates suction-side issues (blocked filters, valve faults, etc.).
    """
    pressure_col = cfg.columns.pressure_off_col
    speed_col = cfg.columns.speed_col
    timestamp_col = cfg.columns.timestamp_col

    if pressure_col is None:
        return False, ""

    if (
        df.empty
        or timestamp_col not in df.columns
        or pressure_col not in df.columns
        or speed_col not in df.columns
    ):
        return False, ""

    # Off-state pressure in the 30 min before the period
    ts = df[timestamp_col]
    before_start = period.start - pd.Timedelta(minutes=30)
    before_off_mask = (
        (ts >= before_start)
        & (ts < period.start)
        & (df[speed_col].fillna(np.nan) == 0)
    )
    if int(before_off_mask.sum()) < 2:
        return False, ""

    p_off_before = float(np.nanmean(df.loc[before_off_mask, pressure_col]))
    if not np.isfinite(p_off_before):
        return False, ""

    # Steady-state pressure during operation
    steady = _steady_df(df, period, cfg)
    if steady.empty or pressure_col not in steady.columns:
        return False, ""

    p_on = float(np.nanmean(steady[pressure_col]))
    if not np.isfinite(p_on):
        return False, ""

    observed_response = p_on - p_off_before

    # Select baseline by sampling rate
    sampling_interval = float(df.attrs.get("sampling_interval_sec", 5.0))
    if sampling_interval <= 30:
        resp_median = baseline.pressure_response_hires_median
        resp_std = baseline.pressure_response_hires_std
    else:
        resp_median = baseline.pressure_response_lores_median
        resp_std = baseline.pressure_response_lores_std

    if not np.isfinite(resp_median) or not np.isfinite(resp_std):
        return False, ""

    response_std = _safe_sigma_threshold(resp_std, min_std=0.5)
    deviation = abs(observed_response - resp_median)

    if deviation > cfg.thresholds.anomaly_sigma * response_std:
        return True, "pressure_response"

    # Physical check: pump operation should not cause significant pressure drop.
    # A negative response exceeding 2× the off-state noise indicates suction failure.
    off_std = _safe_sigma_threshold(baseline.pressure_off_std, min_std=0.5)
    if observed_response < -2.0 * off_std:
        return True, "pressure_response"

    return False, ""


def _check_temp_absolute_limit(
    df: pd.DataFrame,
    period: OperationPeriod,
    baseline: PumpBaseline,
    cfg: SystemConfig,
) -> tuple[bool, str]:
    """Flag periods where any temperature exceeds absolute safety limits."""
    limits = cfg.thresholds.absolute_temp_limits
    if not limits:
        return False, ""

    steady = _steady_df(df, period, cfg)
    if steady.empty:
        return False, ""

    exceeded: list[str] = []
    for col, limit in limits.items():
        if col not in steady.columns:
            continue
        max_val = float(np.nanmax(steady[col]))
        if np.isfinite(max_val) and max_val > limit:
            exceeded.append(col)

    if exceeded:
        return True, "temp_absolute_limit:" + ",".join(exceeded)
    return False, ""


BUILTIN_DETECTORS: dict[str, DetectorFn] = {
    "speed_stability": _check_speed_stability,
    "current_anomaly": _check_current_anomaly,
    "current_speed_ratio": _check_current_speed_ratio,
    "pressure_on": _check_pressure_on,
    "pressure_off": _check_pressure_off,
    "pressure_off_variability": _check_pressure_off_variability,
    "pressure_off_extended": _check_pressure_off_extended,
    "temperatures": _check_temperatures,
    "temp_absolute_limit": _check_temp_absolute_limit,
    "vibration": _check_vibration,
    "off_state_current": _check_off_state_current,
    "pressure_response": _check_pressure_response,
}


def load_custom_detectors(cfg: SystemConfig) -> dict[str, DetectorFn]:
    custom: dict[str, DetectorFn] = {}
    detectors_dir = cfg.custom_detectors_dir

    if detectors_dir is None:
        return custom

    if not detectors_dir.exists() or not detectors_dir.is_dir():
        return custom

    for path in sorted(detectors_dir.glob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue

        module_name = f"custom_detector_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            warnings.warn(f"Could not load detector module spec from {path}", RuntimeWarning)
            continue

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            warnings.warn(f"Failed to import detector module {path}: {exc}", RuntimeWarning)
            continue

        module_detectors = getattr(module, "DETECTORS", None)
        if not isinstance(module_detectors, dict):
            warnings.warn(f"Module {path} does not define a DETECTORS dict", RuntimeWarning)
            continue

        for name, fn in module_detectors.items():
            if not isinstance(name, str):
                warnings.warn(f"Ignoring non-string detector name in {path}", RuntimeWarning)
                continue
            if not callable(fn):
                warnings.warn(f"Ignoring non-callable detector '{name}' in {path}", RuntimeWarning)
                continue
            if name in BUILTIN_DETECTORS:
                warnings.warn(
                    f"Custom detector '{name}' in {path} overrides a built-in name; built-in will take precedence",
                    RuntimeWarning,
                )
            if name in custom:
                warnings.warn(
                    f"Custom detector '{name}' from {path} overrides a previously loaded custom detector",
                    RuntimeWarning,
                )
            custom[name] = fn

    return custom


def classify_period(
    df: pd.DataFrame,
    period: OperationPeriod,
    baseline: PumpBaseline,
    cfg: SystemConfig,
    custom_detectors: dict[str, DetectorFn] | None = None,
) -> ClassificationResult:
    reasons: list[str] = []

    sampling_interval = float(df.attrs.get("sampling_interval_sec", 5.0))
    if not np.isfinite(sampling_interval) or sampling_interval <= 0:
        sampling_interval = 5.0

    steady = _steady_df(df, period, cfg)
    min_rows = 2 if sampling_interval >= 240.0 else 10
    if len(steady) < min_rows:
        reasons.append("insufficient_data")

    plugins = custom_detectors or {}

    for detector_name in cfg.detectors:
        detector = BUILTIN_DETECTORS.get(detector_name)
        if detector is None:
            detector = plugins.get(detector_name)

        if detector is None:
            warnings.warn(f"Detector '{detector_name}' not found; skipping", RuntimeWarning)
            continue

        fired, reason = detector(df, period, baseline, cfg)
        if fired and reason:
            reasons.append(reason)

    if reasons:
        # Count unique detector categories using the base token before ':' or ';'.
        unique_categories = set()
        for r in reasons:
            unique_categories.add(r.split(":")[0].split(";")[0])
        # Absolute temperature limits always force abnormal classification
        has_critical = "temp_absolute_limit" in unique_categories
        min_required = cfg.thresholds.min_detectors_for_abnormal
        classification = "abnormal" if has_critical or len(unique_categories) >= min_required else "normal"
    else:
        classification = "normal"
    return ClassificationResult(classification=classification, reasons=reasons)
