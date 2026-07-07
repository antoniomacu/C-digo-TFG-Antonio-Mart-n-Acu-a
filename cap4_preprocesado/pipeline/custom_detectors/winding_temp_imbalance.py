"""Motor winding temperature balance detector.

Detects per-winding deviation from baseline rather than fixed spread threshold.
Flags imbalance when one winding deviates significantly from its own baseline
while at least one other winding does not — indicating a phase-specific issue.
"""

import numpy as np

from pipeline.period_detector import get_steady_state_mask

WINDING_COLS = [
    "motor_u_winding_temp", "motor_v_winding_temp", "motor_w_winding_temp",
    "motor_u_winding_temp_1", "motor_u_winding_temp_2", "motor_u_winding_temp_3",
]
WINDING_PHYSICAL_MAX_C = 200.0  # above this, treat as sensor fault → NaN
STD_FLOOR = 0.01


def _check_winding_temp_imbalance(df, period, baseline, cfg):
    """Flag winding temperature imbalance via per-winding baseline deviation."""
    steady_mask = get_steady_state_mask(df, period, cfg)
    if not steady_mask.any():
        return False, ""

    available = [col for col in WINDING_COLS if col in df.columns]
    if len(available) < 2:
        return False, ""

    sigma = cfg.thresholds.anomaly_sigma

    deviations = {}
    for col in available:
        median = baseline.temp_medians.get(col, float("nan"))
        std = max(baseline.temp_stds.get(col, float("nan")), STD_FLOOR)
        if not np.isfinite(median) or not np.isfinite(std):
            continue

        vals = df.loc[steady_mask, col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals <= WINDING_PHYSICAL_MAX_C)]
        if len(vals) < 2:
            continue

        col_mean = float(np.mean(vals))
        deviations[col] = (col_mean - median) / std

    if len(deviations) < 2:
        return False, ""

    elevated = [col for col, d in deviations.items() if abs(d) > sigma]
    normal = [col for col, d in deviations.items() if abs(d) <= sigma]

    if elevated and normal:
        labels = [c.replace("motor_", "").replace("_winding_temp", "") for c in elevated]
        return True, f"winding_temp_imbalance:{','.join(labels)}"
    return False, ""


DETECTORS = {
    "winding_temp_imbalance": _check_winding_temp_imbalance,
}
