"""Flow rate anomaly detector."""

import numpy as np

from pipeline.period_detector import get_steady_state_mask

FLOW_COL_CANDIDATES = ["flow", "total_fw_flow"]


def _check_flow_anomaly(df, period, baseline, cfg):
    """Flag abnormal flow rate during operation."""
    steady_mask = get_steady_state_mask(df, period, cfg)
    flow_col = next((c for c in FLOW_COL_CANDIDATES if c in df.columns), None)
    if not steady_mask.any() or flow_col is None:
        return False, ""

    vals = df.loc[steady_mask, flow_col].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size < 2:
        return False, ""

    mean_val = float(np.mean(finite))
    median = baseline.temp_medians.get(flow_col, float("nan"))
    std = baseline.temp_stds.get(flow_col, float("nan"))

    if not np.isfinite(median) or not np.isfinite(std):
        return False, ""

    std = max(std, 0.01)
    sigma = cfg.thresholds.anomaly_sigma

    reasons = []

    speed_col = cfg.columns.speed_col
    if speed_col in df.columns:
        speed_vals = df.loc[steady_mask, speed_col].to_numpy(dtype=float)
        valid = np.isfinite(vals) & np.isfinite(speed_vals) & (speed_vals > 100)
        if valid.sum() >= 2:
            ratio = vals[valid] / speed_vals[valid]
            ratio_mean = float(np.mean(ratio))
            if np.isfinite(median) and baseline.speed_median > 100:
                expected_ratio = median / baseline.speed_median
                ratio_std = max(std / baseline.speed_median, 1e-6)
                if abs(ratio_mean - expected_ratio) > sigma * ratio_std:
                    reasons.append("flow_speed_ratio")

    if reasons:
        return True, "flow_anomaly:" + ",".join(reasons)
    return False, ""


DETECTORS = {
    "flow_anomaly": _check_flow_anomaly,
}
