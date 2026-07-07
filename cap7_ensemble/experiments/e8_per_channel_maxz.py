"""E8 — Per-Channel Max-z Scoring for L1.

Evaluates three alternative L1 scoring policies that use per-channel absolute
z-scores (standardised residuals) instead of or alongside the aggregate
Mahalanobis distance to improve Pump 3 detection.

Background
----------
INV-1 (inv1_pump3_investigation.py) showed that the 61 missed Pump 3 failure
days have a hydraulic / single-channel fault signature: Current Consumption,
Outlet Pressure, and DE Side Bearing vibration show mild but sustained
elevation that the aggregate Mahalanobis distance averages away because the
other 10 channels remain normal. Per-channel max-z scoring surfaces this
weak signal directly.

Policies evaluated
------------------
1. Supplemental P3-channel escalation
   Keep Mahalanobis but also flag a timestep for Pump 3 if the max |z| across
   the three P3-fault channels (Current, Pressure, DE vibration) exceeds a
   calibrated threshold. Other pumps unchanged.

2. Combined OR rule (all pumps)
   Flag a timestep if Mahalanobis >= warning threshold OR the global max |z|
   across all 13 channels exceeds a per-pump calibrated max-z threshold.

3. Pure max-z replacement (all pumps)
   Flag a timestep if the global max |z| across all 13 channels exceeds the
   per-pump calibrated threshold, ignoring Mahalanobis entirely.

All policies apply the same K=3 sustained-alert filter at day level as the
production batch evaluation.

Threshold calibration
---------------------
Primary: Run on normal training CSVs (P95 of per-timestep global max |z|,
per pump). If training data is unavailable, falls back to analytical
Gaussian model thresholds.

Analytical fallback: Solve for t such that P(max of n_channels |z| > t) = 0.05,
using z ~ N(0,1) approximation and independence between channels. This gives a
conservative upper bound on the empirical threshold.

False-alert estimation
----------------------
When raw CSVs are unavailable, false-alert rates are estimated analytically:
for each day in batch_results.json, use n_timesteps and the per-timestep
false-alarm rate (from the Gaussian model) to compute P(Binomial(N, p) >= K).
This ignores inter-channel and inter-timestep correlations, so it overstates
the FA rate for correlated channels. Values are flagged as "analytical".

Acceptance criterion (PLAN.md §4.2)
-------------------------------------
  ≥10 Pump 3 missed days recovered at K=3
  ≤3 pp overall false-alert rate increase
  ≤5 pp Pump 3 false-alert rate increase

Usage
-----
    cd /path/to/unsupervised_learning/ensemble
    uv run python ../experiments/e8_per_channel_maxz.py
    uv run python ../experiments/e8_per_channel_maxz.py --skip-full-batch
    uv run python ../experiments/e8_per_channel_maxz.py --maxz-threshold 4.0

Outputs
-------
    experiments/E8_max_z_results.json   — full results for all policies
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import special as sp

EXPERIMENTS_DIR = Path(__file__).resolve().parent
ENSEMBLE_DIR = EXPERIMENTS_DIR.parent / "ensemble"
WORKSPACE_ROOT = ENSEMBLE_DIR.parent

sys.path.insert(0, str(ENSEMBLE_DIR))
sys.path.insert(0, str(WORKSPACE_ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path("<PATH_TO_DATA_DIR>")
PROD_THRESHOLDS_PATH = (
    ENSEMBLE_DIR / "cond_reg_v2" / "model" / "weights" / "production_thresholds.json"
)
CAL_DIR = ENSEMBLE_DIR / "demos" / "batch_results" / "calibrated_thresholds_full"
BATCH_RESULTS_PATH = ENSEMBLE_DIR / "demos" / "batch_results" / "batch_results.json"
INV1_RESULTS_PATH = EXPERIMENTS_DIR / "INV1_results.json"
OUTPUT_PATH = EXPERIMENTS_DIR / "E8_max_z_results.json"

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

K = 3  # Must match batch_results.json min_alert_steps

# Three channels identified by INV-1 as the Pump 3 hydraulic fault signature
P3_FAULT_CHANNELS = [
    "Main HTF Pump Current Consumption",
    "Main HTF Pump Outlet Pressure",
    "Main HTF Pump DE Side Bearing vibration",
]

N_CHANNELS_GLOBAL = 13  # Total L1 output channels
P_WARN = 95.0  # Calibration percentile for warning threshold


# ---------------------------------------------------------------------------
# Analytical threshold and FA estimation
# ---------------------------------------------------------------------------

def _gaussian_maxz_threshold(n_channels: int, target_percentile: float = 95.0) -> float:
    """Compute the analytical global max-z threshold under a Gaussian model.

    Solves for t such that P(max of n_channels independent |z_i| > t) = 1 - target/100,
    where z_i ~ N(0, 1). Since channels are NOT independent in practice, this is
    a conservative (over-estimated) threshold — it will trigger less often than
    the empirical threshold from correlated residuals.

    Returns t such that P(at least one of n_channels |z_i| > t) = 1 - target/100.
    Equivalently, P(each |z_i| <= t) = (target/100)^(1/n_channels).
    """
    per_channel_quantile = (target_percentile / 100.0) ** (1.0 / n_channels)
    # Per-channel: P(|z| <= t) = per_channel_quantile → P(z <= t) = (1 + per_channel_quantile) / 2
    normal_q = (1.0 + per_channel_quantile) / 2.0
    # Inverse normal CDF
    t = math.sqrt(2) * sp.erfinv(2 * normal_q - 1)
    return t


def _binom_fa_prob(n_timesteps: int, per_ts_fa_rate: float, k: int = K) -> float:
    """P(Binomial(n_timesteps, per_ts_fa_rate) >= k)."""
    if n_timesteps < k:
        return 0.0
    if per_ts_fa_rate <= 0.0:
        return 0.0
    # Use log-space binomial CDF for numerical stability
    cdf_k_minus_1 = sp.bdtr(k - 1, n_timesteps, per_ts_fa_rate)
    return max(0.0, 1.0 - float(cdf_k_minus_1))


def _per_ts_fa_rate(threshold: float, n_channels: int) -> float:
    """P(max of n_channels independent |z_i| > threshold) where z_i ~ N(0,1)."""
    per_ch = 2.0 * float(sp.ndtr(threshold)) - 1.0  # P(|z| <= t)
    return 1.0 - per_ch ** n_channels


# ---------------------------------------------------------------------------
# INV1 fast validation (uses existing per-day channel_max_z data)
# ---------------------------------------------------------------------------

def inv1_analysis(
    inv1_results: list[dict],
    prod_thresholds: dict,
) -> dict:
    """Analysis using existing INV1_results.json data.

    INV1 has per-day channel_max_z: the maximum absolute z-score across all
    timesteps for each channel. This gives the day's peak signal strength but
    cannot apply a K=3 per-timestep filter. It is a conservative (upper-bound)
    estimate of detection — if day-level max_z < threshold, the day is
    definitely not recovered. If day-level max_z >= threshold, the day MAY be
    recovered depending on how many timesteps exceeded the threshold.
    """
    missed = [r for r in inv1_results if r["group"] == "missed"]
    detected = [r for r in inv1_results if r["group"] == "detected"]

    def day_p3_maxz(r: dict) -> float:
        cz = r.get("channel_max_z", {})
        vals = [cz[ch] for ch in P3_FAULT_CHANNELS
                if ch in cz and cz[ch] is not None and not math.isnan(cz[ch])]
        return max(vals) if vals else 0.0

    def day_global_maxz(r: dict) -> float:
        cz = r.get("channel_max_z", {})
        vals = [v for v in cz.values() if v is not None and not math.isnan(v)]
        return max(vals) if vals else 0.0

    p3_maxz_missed = [day_p3_maxz(r) for r in missed]
    p3_maxz_detected = [day_p3_maxz(r) for r in detected]
    global_maxz_missed = [day_global_maxz(r) for r in missed]
    global_maxz_detected = [day_global_maxz(r) for r in detected]

    print("\n" + "=" * 70)
    print("INV1 Fast Validation (per-day max-z proxy, no K=3 timestep filter)")
    print("=" * 70)
    print(
        f"Missed days: {len(missed)}  |  Detected sample: {len(detected)}\n"
        f"Note: 'recovered' here means the day's peak z-score exceeded the threshold.\n"
        f"Full K=3 filter requires per-timestep data (raw CSVs).\n"
    )

    # Threshold sweep for Policy 1 (P3-channel supplemental)
    print("Policy 1 — P3-channel max-z sweep (Pump 3 only):")
    print(f"  {'Threshold':>9} | {'Missed recovered':>18} | {'Det days above thr':>20}")
    sweep_p1: list[dict] = []
    for thr in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
        n_rec = sum(1 for v in p3_maxz_missed if v >= thr)
        n_det_above = sum(1 for v in p3_maxz_detected if v >= thr)
        # Analytical per-ts FA rate at this threshold (3 channels)
        p_ts = _per_ts_fa_rate(thr, n_channels=len(P3_FAULT_CHANNELS))
        print(
            f"  {thr:>9.1f} | {n_rec:>10d}/{len(missed)} ({100*n_rec/max(len(missed),1):>5.1f}%) "
            f"| {n_det_above:>10d}/{len(detected)} "
            f"(per-ts FA~{p_ts*100:.4f}%)"
        )
        sweep_p1.append({
            "threshold": thr,
            "missed_recovered_day_level": n_rec,
            "pct_recovered": round(100 * n_rec / max(len(missed), 1), 1),
            "detected_above_thr": n_det_above,
            "analytical_per_ts_fa_pct": round(p_ts * 100, 5),
        })

    # Threshold sweep for Policies 2 & 3 (global max-z, all channels)
    print("\nPolicies 2/3 — Global max-z sweep (all pumps):")
    print(f"  {'Threshold':>9} | {'Missed recovered':>18} | {'Det days above thr':>20}")
    sweep_global: list[dict] = []
    for thr in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
        n_rec = sum(1 for v in global_maxz_missed if v >= thr)
        n_det_above = sum(1 for v in global_maxz_detected if v >= thr)
        p_ts = _per_ts_fa_rate(thr, n_channels=N_CHANNELS_GLOBAL)
        print(
            f"  {thr:>9.1f} | {n_rec:>10d}/{len(missed)} ({100*n_rec/max(len(missed),1):>5.1f}%) "
            f"| {n_det_above:>10d}/{len(detected)} "
            f"(per-ts FA~{p_ts*100:.4f}%)"
        )
        sweep_global.append({
            "threshold": thr,
            "missed_recovered_day_level": n_rec,
            "pct_recovered": round(100 * n_rec / max(len(missed), 1), 1),
            "detected_above_thr": n_det_above,
            "analytical_per_ts_fa_pct": round(p_ts * 100, 5),
        })

    # Distribution statistics
    print("\nP3-channel max-z distribution (per-day values):")
    print(
        f"  MISSED  — mean={np.mean(p3_maxz_missed):.2f}, median={np.median(p3_maxz_missed):.2f}, "
        f"p90={np.percentile(p3_maxz_missed, 90):.2f}, max={np.max(p3_maxz_missed):.2f}"
    )
    print(
        f"  DETECTED — mean={np.mean(p3_maxz_detected):.2f}, median={np.median(p3_maxz_detected):.2f}, "
        f"p90={np.percentile(p3_maxz_detected, 90):.2f}, max={np.max(p3_maxz_detected):.2f}"
    )

    print("\nGlobal max-z distribution (per-day values):")
    print(
        f"  MISSED  — mean={np.mean(global_maxz_missed):.2f}, median={np.median(global_maxz_missed):.2f}, "
        f"p90={np.percentile(global_maxz_missed, 90):.2f}, max={np.max(global_maxz_missed):.2f}"
    )
    print(
        f"  DETECTED — mean={np.mean(global_maxz_detected):.2f}, median={np.median(global_maxz_detected):.2f}, "
        f"p90={np.percentile(global_maxz_detected, 90):.2f}, max={np.max(global_maxz_detected):.2f}"
    )

    # Recommended threshold for Policy 1 based on acceptance criterion (>=10 recovered)
    # Find lowest threshold where day-level recovery >= acceptance floor
    ACCEPTANCE_FLOOR = 10
    best_p1_thr = None
    for entry in reversed(sweep_p1):  # start from high threshold (more conservative)
        if entry["missed_recovered_day_level"] >= ACCEPTANCE_FLOOR:
            best_p1_thr = entry["threshold"]
            break

    best_global_thr = None
    for entry in reversed(sweep_global):
        if entry["missed_recovered_day_level"] >= ACCEPTANCE_FLOOR:
            best_global_thr = entry["threshold"]
            break

    print(
        f"\nDay-level recommended threshold (≥{ACCEPTANCE_FLOOR} P3 missed days recoverable):"
    )
    print(f"  Policy 1 (P3-channel):  thr={best_p1_thr}")
    print(f"  Policies 2/3 (global):  thr={best_global_thr}")
    print(
        "\nNote: day-level threshold is an UPPER BOUND on the K=3 per-timestep result.\n"
        "The actual K=3 threshold must be LOWER (more permissive) to compensate for\n"
        "the filter reducing flagged-day counts. Full-batch evaluation with raw CSVs\n"
        "is required for precise K=3 policy selection."
    )

    return {
        "n_missed": len(missed),
        "n_detected_sample": len(detected),
        "p3_fault_channels": P3_FAULT_CHANNELS,
        "sweep_policy1_p3_channel": sweep_p1,
        "sweep_global": sweep_global,
        "p3_maxz_missed_stats": {
            "mean": float(np.mean(p3_maxz_missed)),
            "median": float(np.median(p3_maxz_missed)),
            "p90": float(np.percentile(p3_maxz_missed, 90)),
            "max": float(np.max(p3_maxz_missed)),
        },
        "p3_maxz_detected_stats": {
            "mean": float(np.mean(p3_maxz_detected)),
            "median": float(np.median(p3_maxz_detected)),
            "p90": float(np.percentile(p3_maxz_detected, 90)),
            "max": float(np.max(p3_maxz_detected)),
        },
        "global_maxz_missed_stats": {
            "mean": float(np.mean(global_maxz_missed)),
            "median": float(np.median(global_maxz_missed)),
            "p90": float(np.percentile(global_maxz_missed, 90)),
            "max": float(np.max(global_maxz_missed)),
        },
        "global_maxz_detected_stats": {
            "mean": float(np.mean(global_maxz_detected)),
            "median": float(np.median(global_maxz_detected)),
            "p90": float(np.percentile(global_maxz_detected, 90)),
            "max": float(np.max(global_maxz_detected)),
        },
        "recommended_threshold_p3_channel_day_level": best_p1_thr,
        "recommended_threshold_global_day_level": best_global_thr,
    }


# ---------------------------------------------------------------------------
# Analytical batch FA estimation (used when raw CSVs unavailable)
# ---------------------------------------------------------------------------

def analytical_batch_fa_estimate(
    batch_days: list[dict],
    threshold_p3: float,
    threshold_global: float,
    k: int = K,
) -> dict:
    """Estimate false-alert rate increase under each policy using a Gaussian model.

    For each normal day in batch_results.json, computes:
      P(at least K timesteps flagged by max-z policy)
    using a Binomial model with the Gaussian per-timestep FA rate.

    This is a conservative overestimate because:
    - Channels are treated as independent (actual correlations reduce FA rate)
    - Residuals are treated as i.i.d. across timesteps (actual correlations
      reduce the effective number of independent trials)

    Results labelled "analytical_estimate" in the output JSON.
    """
    normal_days = [d for d in batch_days if d["label"] == "normal"]
    p3_normal_days = [d for d in normal_days if d["pump_id"] == 3]

    # Per-timestep FA rates under the Gaussian model
    p_ts_p3 = _per_ts_fa_rate(threshold_p3, n_channels=len(P3_FAULT_CHANNELS))
    p_ts_global = _per_ts_fa_rate(threshold_global, n_channels=N_CHANNELS_GLOBAL)

    def expected_fa_days(days: list[dict], p_ts: float) -> float:
        """Expected number of false-alarm days under the Binomial model."""
        return sum(
            _binom_fa_prob(d["n_timesteps"], p_ts, k=k)
            for d in days
        )

    # Policy 1: P3-channel escalation only affects Pump 3 days
    pol1_expected_p3_fa = expected_fa_days(p3_normal_days, p_ts_p3)
    pol1_total_fa = sum(
        _binom_fa_prob(d["n_timesteps"], p_ts_p3 if d["pump_id"] == 3 else 0.0, k=k)
        for d in normal_days
    )
    # Policy 1 adds to baseline FA: total FA = baseline + new P3-FA days
    # (days already flagged by Mahalanobis are unaffected)
    # We report the INCREMENTAL FA from the max-z component only
    pol1_new_fa_p3 = pol1_expected_p3_fa
    pol1_new_fa_total = pol1_total_fa

    # Policy 2/3: global max-z applies to all pumps
    pol23_expected_fa = expected_fa_days(normal_days, p_ts_global)
    pol23_expected_p3_fa = expected_fa_days(p3_normal_days, p_ts_global)

    n_normal = len(normal_days)
    n_p3_normal = len(p3_normal_days)

    print("\n--- Analytical FA Rate Estimate (Gaussian model, upper bound) ---")
    print(f"Thresholds used: P3-channel={threshold_p3:.2f}, global={threshold_global:.2f}")
    print(
        f"Per-timestep FA rate: P3-channel={p_ts_p3*100:.5f}%, "
        f"global={p_ts_global*100:.5f}%"
    )
    print(
        f"Policy 1 (P3-channel): expected {pol1_new_fa_p3:.1f} new P3 FA days "
        f"/ {n_p3_normal} P3 normal = "
        f"{100*pol1_new_fa_p3/max(n_p3_normal,1):.2f}% estimated P3 FA increase"
    )
    print(
        f"Policy 2/3 (global): expected {pol23_expected_fa:.1f} / {n_normal} normal days "
        f"= {100*pol23_expected_fa/max(n_normal,1):.2f}% estimated overall FA increase"
    )
    print("(These are UPPER BOUNDS — actual FA is lower due to inter-channel correlations)")

    return {
        "method": "analytical_gaussian_upper_bound",
        "threshold_p3_channel": threshold_p3,
        "threshold_global": threshold_global,
        "per_ts_fa_rate_p3_pct": round(p_ts_p3 * 100, 6),
        "per_ts_fa_rate_global_pct": round(p_ts_global * 100, 6),
        "n_normal_days": n_normal,
        "n_p3_normal_days": n_p3_normal,
        "policy1": {
            "expected_new_p3_fa_days": round(pol1_new_fa_p3, 2),
            "estimated_p3_fa_increase_pp": round(
                100 * pol1_new_fa_p3 / max(n_p3_normal, 1), 2
            ),
            "estimated_overall_fa_increase_pp": round(
                100 * pol1_new_fa_total / max(n_normal, 1), 2
            ),
        },
        "policy23": {
            "expected_new_fa_days": round(pol23_expected_fa, 2),
            "expected_new_p3_fa_days": round(pol23_expected_p3_fa, 2),
            "estimated_overall_fa_increase_pp": round(
                100 * pol23_expected_fa / max(n_normal, 1), 2
            ),
            "estimated_p3_fa_increase_pp": round(
                100 * pol23_expected_p3_fa / max(n_p3_normal, 1), 2
            ),
        },
    }


# ---------------------------------------------------------------------------
# Full batch evaluation (requires raw CSVs)
# ---------------------------------------------------------------------------

def _try_import_streaming():
    try:
        from model.streaming import create_streaming_detector, StreamingTimestepResult
        from demos.streaming_demo import _load_day, _parse_pump_from_filename
        return create_streaming_detector, StreamingTimestepResult, _load_day, _parse_pump_from_filename
    except ImportError as exc:
        return None


def run_day_l1(detector, csv_path: Path, pump_id: int):
    from demos.streaming_demo import _load_day
    from model.streaming import StreamingTimestepResult
    try:
        df = _load_day(csv_path)
    except Exception as exc:
        logger.debug("Failed to load %s: %s", csv_path.name, exc)
        return None
    if len(df) < 5:
        return None
    detector.reset_pump(pump_id)
    z_per_ts: list[dict[str, float]] = []
    mahal_per_ts: list[float] = []
    for ts, row in df.iterrows():
        try:
            result: StreamingTimestepResult = detector.process_timestep(pump_id, ts, row)
        except Exception as exc:
            logger.debug("Error on %s: %s", csv_path.name, exc)
            continue
        abs_z = {ch: abs(z) for ch, z in result.l1_z_scores.items()}
        z_per_ts.append(abs_z)
        mahal_per_ts.append(float(result.l1_mahalanobis))
    return (z_per_ts, mahal_per_ts) if z_per_ts else None


def classify_day_policy(
    z_per_ts: list[dict[str, float]],
    mahal_per_ts: list[float],
    pump_id: int,
    mahal_warning: float,
    threshold_global: float,
    threshold_p3: float,
    k: int = K,
) -> tuple[bool, bool, bool, bool]:
    """Return (baseline, policy1, policy2, policy3) day flags."""
    base, p1, p2, p3 = 0, 0, 0, 0
    for i, mahal in enumerate(mahal_per_ts):
        z_dict = z_per_ts[i]
        mahal_flag = mahal >= mahal_warning
        g_vals = [v for v in z_dict.values() if not math.isnan(v)]
        g_maxz = max(g_vals) if g_vals else 0.0
        p3_vals = [z_dict[ch] for ch in P3_FAULT_CHANNELS
                   if ch in z_dict and not math.isnan(z_dict[ch])]
        p3_maxz = max(p3_vals) if p3_vals else 0.0

        base += int(mahal_flag)
        p1 += int(mahal_flag or (pump_id == 3 and p3_maxz >= threshold_p3))
        p2 += int(mahal_flag or g_maxz >= threshold_global)
        p3 += int(g_maxz >= threshold_global)

    return base >= k, p1 >= k, p2 >= k, p3 >= k


def calibrate_maxz_thresholds(detector, data_dir: Path) -> tuple[dict, dict]:
    """Calibrate max-z thresholds from normal training CSVs.

    Returns (global_warn_by_pump, p3_channel_warn) where global_warn_by_pump is
    {pump_id: P95 threshold} and p3_channel_warn is a float for Pump 3.
    """
    from demos.streaming_demo import _parse_pump_from_filename
    normal_dir = data_dir / "train"
    csv_files = sorted(normal_dir.glob("*.csv"))
    print(f"\n[Calibration] Running on {len(csv_files)} normal training CSVs...")

    global_maxz: dict[int, list[float]] = {}
    p3_ch_maxz: list[float] = []

    for i, csv_path in enumerate(csv_files, 1):
        if i % 100 == 0:
            print(f"  {i}/{len(csv_files)}...")
        pump_id = _parse_pump_from_filename(csv_path)
        if pump_id is None:
            continue
        day = run_day_l1(detector, csv_path, pump_id)
        if day is None:
            continue
        z_per_ts, _ = day
        for z_dict in z_per_ts:
            g_vals = [v for v in z_dict.values() if not math.isnan(v)]
            if g_vals:
                global_maxz.setdefault(pump_id, []).append(max(g_vals))
            if pump_id == 3:
                p3_vals = [z_dict[ch] for ch in P3_FAULT_CHANNELS
                           if ch in z_dict and not math.isnan(z_dict[ch])]
                if p3_vals:
                    p3_ch_maxz.append(max(p3_vals))

    global_warn = {}
    for pid, scores in global_maxz.items():
        arr = np.asarray(scores, dtype=float)
        global_warn[pid] = float(np.percentile(arr, P_WARN))
        print(
            f"  Pump {pid}: global P95={global_warn[pid]:.3f}  (n={len(scores)})"
        )
    p3_warn = float(np.percentile(p3_ch_maxz, P_WARN)) if p3_ch_maxz else 0.0
    print(f"  Pump 3 P3-channel P95={p3_warn:.3f}")
    return global_warn, p3_warn


def evaluate_full_batch(
    detector,
    data_dir: Path,
    batch_days: list[dict],
    calibrated_l1: dict,
    threshold_global: dict[int, float],
    threshold_p3: float,
) -> dict:
    """Full-dataset policy evaluation. Returns metrics dict."""
    from demos.streaming_demo import _parse_pump_from_filename
    baseline_by_name = {d["csv_name"]: d for d in batch_days}
    all_files = (
        [(f, "normal") for f in sorted((data_dir / "train").glob("*.csv"))] +
        [(f, "abnormal") for f in sorted((data_dir / "test").glob("*.csv"))]
    )

    results: list[dict] = []
    skipped = 0
    print(f"\n[Full Batch] Evaluating {len(all_files)} CSVs...")

    for i, (csv_path, label) in enumerate(all_files, 1):
        if i % 200 == 0:
            print(f"  {i}/{len(all_files)}...")
        pump_id = _parse_pump_from_filename(csv_path)
        if pump_id is None:
            skipped += 1
            continue
        baseline_row = baseline_by_name.get(csv_path.name)
        if baseline_row is None:
            skipped += 1
            continue
        day = run_day_l1(detector, csv_path, pump_id)
        if day is None:
            skipped += 1
            continue
        z_per_ts, mahal_per_ts = day

        pump_key = str(pump_id)
        pp = calibrated_l1.get("per_pump", {}).get(pump_key, {})
        if not pp.get("fallback_to_global", True):
            mw = pp["mahalanobis"]["warning"]
        else:
            mw = calibrated_l1["global"]["mahalanobis"]["warning"]

        t_global = threshold_global.get(pump_id, float("inf"))
        base, p1, p2, p3 = classify_day_policy(
            z_per_ts, mahal_per_ts, pump_id, mw, t_global, threshold_p3
        )
        n_nan = sum(
            1 for ch in (z_per_ts[0].keys() if z_per_ts else [])
            if all(math.isnan(ts.get(ch, float("nan"))) for ts in z_per_ts)
        )
        results.append({
            "csv_name": csv_path.name,
            "pump_id": pump_id,
            "label": label,
            "n_ts": len(mahal_per_ts),
            "baseline": base,
            "policy1": p1,
            "policy2": p2,
            "policy3": p3,
            "n_nan_channels": n_nan,
        })

    print(f"[Full Batch] Done: {len(results)} evaluated, {skipped} skipped.")
    return _compute_batch_metrics(results, threshold_global, threshold_p3)


def _compute_batch_metrics(results: list[dict], threshold_global: dict, threshold_p3: float) -> dict:
    """Compute per-policy FA/detection metrics from full-batch day results."""
    normal = [r for r in results if r["label"] == "normal"]
    abnormal = [r for r in results if r["label"] == "abnormal"]
    p3_normal = [r for r in normal if r["pump_id"] == 3]
    p3_abnormal = [r for r in abnormal if r["pump_id"] == 3]

    def metrics(col: str) -> dict:
        n_fa = sum(1 for r in normal if r[col])
        n_det = sum(1 for r in abnormal if r[col])
        p3_fa = sum(1 for r in p3_normal if r[col])
        p3_det = sum(1 for r in p3_abnormal if r[col])
        # P3 missed days recovered vs baseline
        p3_rec = sum(1 for r in p3_abnormal if r[col] and not r["baseline"])
        return {
            "n_normal": len(normal),
            "n_abnormal": len(abnormal),
            "n_fa": n_fa,
            "n_det": n_det,
            "fa_rate": round(n_fa / max(len(normal), 1), 4),
            "det_rate": round(n_det / max(len(abnormal), 1), 4),
            "p3_n_normal": len(p3_normal),
            "p3_n_abnormal": len(p3_abnormal),
            "p3_fa": p3_fa,
            "p3_det": p3_det,
            "p3_fa_rate": round(p3_fa / max(len(p3_normal), 1), 4),
            "p3_det_rate": round(p3_det / max(len(p3_abnormal), 1), 4),
            "p3_missed_recovered": p3_rec,
        }

    base = metrics("baseline")
    pol = {k: metrics(k) for k in ("policy1", "policy2", "policy3")}

    # Acceptance check
    def check(m: dict, bm: dict) -> tuple[bool, list[str]]:
        notes = []
        ok = True
        rec = m["p3_missed_recovered"]
        fa_d = (m["fa_rate"] - bm["fa_rate"]) * 100
        p3_d = (m["p3_fa_rate"] - bm["p3_fa_rate"]) * 100
        if rec < 10:
            notes.append(f"FAIL: {rec} P3 days recovered (need ≥10)")
            ok = False
        else:
            notes.append(f"PASS: {rec} P3 days recovered")
        if fa_d > 3.0:
            notes.append(f"FAIL: overall FA +{fa_d:.1f}pp (max 3pp)")
            ok = False
        else:
            notes.append(f"PASS: overall FA delta {fa_d:+.1f}pp")
        if p3_d > 5.0:
            notes.append(f"FAIL: Pump 3 FA +{p3_d:.1f}pp (max 5pp)")
            ok = False
        else:
            notes.append(f"PASS: Pump 3 FA delta {p3_d:+.1f}pp")
        return ok, notes

    sep = "=" * 80
    print(f"\n{sep}")
    print("E8 POLICY COMPARISON — Full Dataset")
    print(sep)
    print(f"{'Policy':<30} {'FA%':>6} {'Det%':>6} {'P3 FA%':>8} {'P3 Det%':>9} {'P3 Rec':>7}")
    print("-" * 80)
    base_fa, base_det = base["fa_rate"] * 100, base["det_rate"] * 100
    base_p3fa, base_p3det = base["p3_fa_rate"] * 100, base["p3_det_rate"] * 100
    print(f"  {'Baseline (Mahalanobis K=3)':<28} {base_fa:>5.1f}% {base_det:>5.1f}% {base_p3fa:>7.1f}% {base_p3det:>8.1f}%  {'—':>7}")
    for name, m in [("Policy 1: P3-ch suppl.", "policy1"), ("Policy 2: Combined OR", "policy2"), ("Policy 3: Pure max-z", "policy3")]:
        mm = pol[m]
        fa_d = (mm["fa_rate"] - base["fa_rate"]) * 100
        det_d = (mm["det_rate"] - base["det_rate"]) * 100
        ok, notes = check(mm, base)
        verdict = "✓" if ok else "✗"
        print(
            f"  {name:<28} {mm['fa_rate']*100:>5.1f}% {mm['det_rate']*100:>5.1f}% "
            f"{mm['p3_fa_rate']*100:>7.1f}% {mm['p3_det_rate']*100:>8.1f}% "
            f"{mm['p3_missed_recovered']:>6d} {verdict}"
        )
        print(
            f"  {'delta':<28} {fa_d:>+5.1f}pp {det_d:>+5.1f}pp "
            f"{(mm['p3_fa_rate']-base['p3_fa_rate'])*100:>+7.1f}pp "
            f"{(mm['p3_det_rate']-base['p3_det_rate'])*100:>+7.1f}pp"
        )
    print(sep)

    return {
        "source": "full_batch_empirical",
        "thresholds": {
            "global_warn_by_pump": {str(k): v for k, v in threshold_global.items()},
            "p3_channel_warn": threshold_p3,
        },
        "baseline": base,
        "policy1": {**pol["policy1"], **dict(zip(["accepts", "notes"], check(pol["policy1"], base)))},
        "policy2": {**pol["policy2"], **dict(zip(["accepts", "notes"], check(pol["policy2"], base)))},
        "policy3": {**pol["policy3"], **dict(zip(["accepts", "notes"], check(pol["policy3"], base)))},
        "day_results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E8: Per-Channel Max-z Scoring for L1")
    p.add_argument("--skip-full-batch", action="store_true",
                   help="Skip the full-dataset batch evaluation (INV1 analysis only).")
    p.add_argument("--data-dir", default=str(DATA_DIR),
                   help=f"Dataset root path (must have train/ and test/). Default: {DATA_DIR}")
    p.add_argument("--maxz-threshold", type=float, default=None,
                   help="Override global max-z warning threshold (used for all pumps).")
    p.add_argument("--p3-threshold", type=float, default=None,
                   help="Override P3-channel max-z warning threshold.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    normal_dir = data_dir / "train"
    data_available = normal_dir.exists() and any(normal_dir.glob("*.csv"))

    print("E8 — Per-Channel Max-z Scoring for L1")
    print("=" * 50)
    if not data_available:
        print(f"[Warning] Raw CSV data not found at {data_dir}/train/")
        print("  INV1 analysis will run. Full batch requires raw data.")
        print("  Analytical FA estimates will be reported instead.")

    # Load supporting data
    with open(PROD_THRESHOLDS_PATH) as f:
        prod_thresholds: dict = json.load(f)
    with open(CAL_DIR / "l1_thresholds.json") as f:
        calibrated_l1: dict = json.load(f)
    with open(CAL_DIR / "l2_thresholds.json") as f:
        calibrated_l2: dict = json.load(f)
    with open(BATCH_RESULTS_PATH) as f:
        batch_data: dict = json.load(f)
    batch_days: list[dict] = batch_data["days"]
    with open(INV1_RESULTS_PATH) as f:
        inv1_results: list[dict] = json.load(f)

    print(f"Loaded {len(batch_days)} batch days, {len(inv1_results)} INV1 days.")

    # -----------------------------------------------------------------------
    # INV1 fast validation
    # -----------------------------------------------------------------------
    inv1_summary = inv1_analysis(inv1_results, prod_thresholds)

    # -----------------------------------------------------------------------
    # Determine thresholds to use for batch evaluation / analytical estimates
    # -----------------------------------------------------------------------
    # Analytical fallback thresholds
    analytical_thr_global = _gaussian_maxz_threshold(N_CHANNELS_GLOBAL, P_WARN)
    analytical_thr_p3 = _gaussian_maxz_threshold(len(P3_FAULT_CHANNELS), P_WARN)

    print(
        f"\nAnalytical Gaussian thresholds (P{P_WARN:.0f}, independent-channel upper bound):"
        f"\n  Global ({N_CHANNELS_GLOBAL} channels): {analytical_thr_global:.3f}"
        f"\n  P3-channel ({len(P3_FAULT_CHANNELS)} channels): {analytical_thr_p3:.3f}"
    )

    # -----------------------------------------------------------------------
    # Full batch evaluation (if data available and not skipped)
    # -----------------------------------------------------------------------
    full_batch_result: dict | None = None

    if not args.skip_full_batch:
        if data_available:
            print("\nLoading streaming detector for full-batch evaluation...")
            from model.streaming import create_streaming_detector
            detector = create_streaming_detector(
                l1_thresholds=calibrated_l1,
                l2_thresholds=calibrated_l2,
            )
            # Calibrate or use override
            t_global_map, t_p3 = calibrate_maxz_thresholds(detector, data_dir)
            if args.maxz_threshold is not None:
                t_global_map = {p: args.maxz_threshold for p in t_global_map}
                print(f"  Override: global threshold={args.maxz_threshold}")
            if args.p3_threshold is not None:
                t_p3 = args.p3_threshold
                print(f"  Override: P3-channel threshold={args.p3_threshold}")
            full_batch_result = evaluate_full_batch(
                detector, data_dir, batch_days, calibrated_l1, t_global_map, t_p3
            )
        else:
            # No raw data — use analytical estimates
            # CLI args take precedence; else fall back to INV1-recommended; else analytical
            inv1_best_p3_thr = inv1_summary.get("recommended_threshold_p3_channel_day_level")
            inv1_best_global_thr = inv1_summary.get("recommended_threshold_global_day_level")

            if args.p3_threshold is not None:
                t_p3 = args.p3_threshold
            elif inv1_best_p3_thr is not None:
                t_p3 = float(inv1_best_p3_thr)
            else:
                t_p3 = analytical_thr_p3

            if args.maxz_threshold is not None:
                t_global = args.maxz_threshold
            elif inv1_best_global_thr is not None:
                t_global = float(inv1_best_global_thr)
            else:
                t_global = analytical_thr_global

            print(
                f"\nUsing thresholds for analytical FA estimate: "
                f"global={t_global:.2f}, P3-channel={t_p3:.2f}"
            )
            full_batch_result = analytical_batch_fa_estimate(batch_days, t_p3, t_global)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    analytical_thresholds = {
        "global_independent_channel_p95": analytical_thr_global,
        "p3_channel_independent_p95": analytical_thr_p3,
        "note": (
            "Analytical thresholds assume independent Gaussian channels; "
            "actual correlated thresholds from training data will be lower."
        ),
    }

    output = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "k_filter": K,
        "n_channels_global": N_CHANNELS_GLOBAL,
        "p3_fault_channels": P3_FAULT_CHANNELS,
        "data_available_for_batch": data_available,
        "analytical_gaussian_thresholds": analytical_thresholds,
        "inv1_analysis": inv1_summary,
        "full_batch_analysis": full_batch_result,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
