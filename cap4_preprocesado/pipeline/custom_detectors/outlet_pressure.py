"""Outlet pressure anomaly detector."""

import numpy as np

from pipeline.period_detector import get_steady_state_mask

COL = "outlet_pressure"


def _check_outlet_pressure(df, period, baseline, cfg):
    """Flag abnormal outlet pressure: mean shift, excessive variability, or frozen sensor."""
    steady_mask = get_steady_state_mask(df, period, cfg)
    if not steady_mask.any() or COL not in df.columns:
        return False, ""

    vals = df.loc[steady_mask, COL].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size < 2:
        return False, ""

    mean_val = float(np.mean(finite))
    period_std = float(np.std(finite))

    median = baseline.temp_medians.get(COL, float("nan"))
    std = baseline.temp_stds.get(COL, float("nan"))
    var_median = baseline.temp_var_medians.get(COL, float("nan"))
    var_std = baseline.temp_var_stds.get(COL, float("nan"))
    sigma = cfg.thresholds.anomaly_sigma

    reasons = []

    # Check 1: Speed-normalized mean shift (affinity law: P ∝ N²)
    speed_col = cfg.columns.speed_col
    if np.isfinite(median) and np.isfinite(std) and speed_col in df.columns:
        speed_vals = df.loc[steady_mask, speed_col].to_numpy(dtype=float)
        speed_mean = float(np.nanmean(speed_vals))
        baseline_speed = baseline.speed_median
        if (
            np.isfinite(speed_mean)
            and speed_mean > 100
            and np.isfinite(baseline_speed)
            and baseline_speed > 100
        ):
            # Compare pressure/speed² ratios
            observed_ratio = mean_val / (speed_mean**2)
            expected_ratio = median / (baseline_speed**2)
            ratio_std = max(std / (baseline_speed**2), 1e-9)
            if abs(observed_ratio - expected_ratio) > sigma * ratio_std:
                reasons.append("outlet_pressure:mean_shift")
        else:
            # Fallback: absolute comparison when speed data unavailable
            safe_std = max(std, 0.01)
            if abs(mean_val - median) > sigma * safe_std:
                reasons.append("outlet_pressure:mean_shift")

    # Check 2: Excessive variability
    variability_flagged = False
    if np.isfinite(var_median) and var_median > 0:
        # Additive check: period_std > var_median + sigma * var_std
        if np.isfinite(var_std):
            safe_var_std = max(var_std, 0.01)
            if period_std > var_median + sigma * safe_var_std:
                variability_flagged = True
        # Multiplicative check using var_median
        if period_std > 2.5 * var_median:
            variability_flagged = True
    # Hard cap: domain-specific maximum expected variability
    # Resistant to baseline contamination — uses physical knowledge
    variability_cap = getattr(cfg.thresholds, "outlet_pressure_variability_cap", 0.0)
    if variability_cap > 0 and period_std > variability_cap:
        variability_flagged = True
    if variability_flagged:
        reasons.append("outlet_pressure:variability")

    # Check 3: Frozen sensor (new)
    frozen_threshold = cfg.thresholds.frozen_sensor_std_threshold
    frozen_min_samples = cfg.thresholds.frozen_sensor_min_samples
    if finite.size >= frozen_min_samples and period_std < frozen_threshold:
        reasons.append("outlet_pressure:frozen")

    if reasons:
        return True, ";".join(reasons)
    return False, ""


DETECTORS = {
    "outlet_pressure": _check_outlet_pressure,
}
