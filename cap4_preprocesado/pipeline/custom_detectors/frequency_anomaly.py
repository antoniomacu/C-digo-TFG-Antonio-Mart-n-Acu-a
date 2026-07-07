"""Frequency (VFD) anomaly detector.

Flags periods where the reported electrical frequency is zero or abnormally
deviated from baseline while the motor is running at operational speed.
"""

import numpy as np

from pipeline.period_detector import get_steady_state_mask


def _check_frequency_anomaly(df, period, baseline, cfg):
    col = "frequency"
    if col not in df.columns:
        return False, ""

    steady_mask = get_steady_state_mask(df, period, cfg)
    if not steady_mask.any():
        return False, ""

    vals = df.loc[steady_mask, col].to_numpy(dtype=float)
    finite_vals = vals[np.isfinite(vals)]
    if len(finite_vals) < 2:
        return False, ""

    freq_mean = float(np.mean(finite_vals))

    # Absolute check: frequency near zero while pump is operating.
    if freq_mean < 1.0:
        return True, "frequency_anomaly"

    # Baseline deviation check.
    median = baseline.temp_medians.get(col, float("nan"))
    std = baseline.temp_stds.get(col, float("nan"))
    if not np.isfinite(median) or not np.isfinite(std):
        return False, ""

    std = max(std, 0.01)
    sigma = cfg.thresholds.anomaly_sigma
    if abs(freq_mean - median) > sigma * std:
        return True, "frequency_anomaly"

    return False, ""


DETECTORS = {
    "frequency_anomaly": _check_frequency_anomaly,
}