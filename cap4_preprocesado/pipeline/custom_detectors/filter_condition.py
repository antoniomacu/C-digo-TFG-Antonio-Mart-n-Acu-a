"""Filter condition detector - monitors strainer/filter differential pressure."""

import numpy as np

from pipeline.period_detector import get_steady_state_mask

COL = "filter_diff_pressure"


def _check_filter_condition(df, period, baseline, cfg):
    """Flag elevated filter differential pressure (indicates clogging)."""
    steady_mask = get_steady_state_mask(df, period, cfg)
    if not steady_mask.any() or COL not in df.columns:
        return False, ""

    vals = df.loc[steady_mask, COL].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size < 2:
        return False, ""

    mean_val = float(np.mean(finite))
    median = baseline.temp_medians.get(COL, float("nan"))
    std = baseline.temp_stds.get(COL, float("nan"))

    if not np.isfinite(median) or not np.isfinite(std):
        return False, ""

    std = max(std, 0.01)
    if mean_val > median + cfg.thresholds.anomaly_sigma * std:
        return True, "filter_condition"
    return False, ""


DETECTORS = {
    "filter_condition": _check_filter_condition,
}
