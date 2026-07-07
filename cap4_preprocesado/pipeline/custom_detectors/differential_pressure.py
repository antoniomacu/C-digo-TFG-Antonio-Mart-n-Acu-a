"""Pump differential pressure (head) detector."""

import numpy as np

from pipeline.period_detector import get_steady_state_mask

OUTLET_COL = "outlet_pressure"
INLET_COL = "inlet_pressure"


def _check_differential_pressure(df, period, baseline, cfg):
    """Flag abnormal pump differential pressure (head = outlet - inlet)."""
    steady_mask = get_steady_state_mask(df, period, cfg)
    if not steady_mask.any():
        return False, ""
    if OUTLET_COL not in df.columns or INLET_COL not in df.columns:
        return False, ""

    outlet = df.loc[steady_mask, OUTLET_COL].to_numpy(dtype=float)
    inlet = df.loc[steady_mask, INLET_COL].to_numpy(dtype=float)
    valid = np.isfinite(outlet) & np.isfinite(inlet)
    if valid.sum() < 2:
        return False, ""

    head = outlet[valid] - inlet[valid]
    head_std = float(np.std(head))

    outlet_std = baseline.temp_stds.get(OUTLET_COL, float("nan"))
    if not np.isfinite(outlet_std):
        return False, ""

    sigma = cfg.thresholds.anomaly_sigma
    baseline_std = max(float(outlet_std), 0.5)

    if head_std > sigma * baseline_std:
        return True, "differential_pressure"
    return False, ""


DETECTORS = {
    "differential_pressure": _check_differential_pressure,
}
