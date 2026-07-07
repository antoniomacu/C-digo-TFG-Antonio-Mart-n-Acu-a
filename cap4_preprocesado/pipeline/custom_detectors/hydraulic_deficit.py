"""Hydraulic deficit detector.

Flags periods where flow, NPSH pressure, and filter differential pressure
are all simultaneously depressed - indicating a dead-heading, blocked
suction, or similar hydraulic restriction condition.
"""

import numpy as np

from pipeline.period_detector import get_steady_state_mask

SIGNALS = [
    ("total_fw_flow", 1.0),
    ("npsh_pressure", 1.0),
    ("filter_diff_pressure", 1.5),
]


def _check_hydraulic_deficit(df, period, baseline, cfg):
    steady_mask = get_steady_state_mask(df, period, cfg)
    if not steady_mask.any():
        return False, ""

    depressed_count = 0
    for col, k in SIGNALS:
        if col not in df.columns:
            return False, ""

        vals = df.loc[steady_mask, col].to_numpy(dtype=float)
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals) < 2:
            return False, ""

        col_mean = float(np.mean(finite_vals))

        # Get baseline from the appropriate source.
        if col == "npsh_pressure":
            median = baseline.pressure_on_median
            std = baseline.pressure_on_std
        else:
            median = baseline.temp_medians.get(col, float("nan"))
            std = baseline.temp_stds.get(col, float("nan"))

        if not np.isfinite(median) or not np.isfinite(std):
            return False, ""

        std = max(std, 0.001)
        if col_mean < median - k * std:
            depressed_count += 1

    if depressed_count == len(SIGNALS):
        return True, "hydraulic_deficit"

    return False, ""


DETECTORS = {
    "hydraulic_deficit": _check_hydraulic_deficit,
}