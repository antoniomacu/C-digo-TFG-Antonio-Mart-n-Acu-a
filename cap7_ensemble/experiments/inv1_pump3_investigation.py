"""INV-1 — Pump 3 missed failure day investigation.

Runs the ensemble detector on the 61 true-miss Pump 3 abnormal days (0 warning + 0 alarm
steps in batch_results.json) and on a matched sample of detected Pump 3 days for comparison.
Computes per-timestep L1 Mahalanobis scores, L2 smoothed MSE, and per-channel z-scores to
identify why the ensemble classifies these failure days as normal.

Usage:
    cd <PATH_TO_PROJECT>/ensemble
    uv run python ../experiments/inv1_pump3_investigation.py
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ENSEMBLE_DIR = Path(__file__).resolve().parent.parent / "ensemble"
WORKSPACE_ROOT = ENSEMBLE_DIR.parent

sys.path.insert(0, str(ENSEMBLE_DIR))
sys.path.insert(0, str(WORKSPACE_ROOT))

from model.streaming import create_streaming_detector, StreamingTimestepResult
from demos.streaming_demo import _load_day, _parse_pump_from_filename

logging.basicConfig(level=logging.WARNING)

DATA_DIR = Path("<PATH_TO_DATA_DIR>")
BATCH_RESULTS = ENSEMBLE_DIR / "demos" / "batch_results" / "batch_results.json"
CAL_DIR = ENSEMBLE_DIR / "demos" / "batch_results" / "calibrated_thresholds_full"
RESULTS_OUT = Path(__file__).parent / "INV1_results.json"


def load_targets() -> tuple[list[str], list[str]]:
    """Return (missed_csv_names, detected_csv_names) for Pump 3 from batch_results."""
    with open(BATCH_RESULTS) as f:
        data = json.load(f)

    p3_abn = [d for d in data["days"] if d["pump_id"] == 3 and d["label"] == "abnormal"]
    missed = [d["csv_name"] for d in p3_abn
              if d["v2_n_warning_steps"] == 0 and d["v2_n_alarm_steps"] == 0]
    detected = [d["csv_name"] for d in p3_abn
                if d["v2_n_warning_steps"] > 0 or d["v2_n_alarm_steps"] > 0]

    # Sample up to 60 detected for comparison
    rng = random.Random(42)
    detected_sample = rng.sample(detected, min(60, len(detected)))

    return missed, detected_sample


def load_calibrated_thresholds() -> tuple[dict, dict]:
    with open(CAL_DIR / "l1_thresholds.json") as f:
        l1 = json.load(f)
    with open(CAL_DIR / "l2_thresholds.json") as f:
        l2 = json.load(f)
    return l1, l2


def run_day(
    detector,
    csv_path: Path,
    pump_id: int,
) -> Optional[dict]:
    """Run detector on one CSV. Returns per-day summary or None if too short."""
    try:
        df = _load_day(csv_path)
    except Exception as e:
        print(f"  Failed to load {csv_path.name}: {e}")
        return None

    if len(df) < 5:
        return None

    detector.reset_pump(pump_id)

    l1_scores: list[float] = []
    l2_scores: list[float] = []
    z_scores_by_channel: dict[str, list[float]] = {}

    for ts, row in df.iterrows():
        result: StreamingTimestepResult = detector.process_timestep(
            pump_id=pump_id,
            timestamp=ts,
            row=row,
        )
        l1_scores.append(result.l1_mahalanobis)

        if result.l2_smoothed_mse is not None:
            l2_scores.append(result.l2_smoothed_mse)

        # Accumulate per-channel absolute z-scores
        for channel, z in result.l1_z_scores.items():
            z_scores_by_channel.setdefault(channel, []).append(abs(z))

    if not l1_scores:
        return None

    # Per-channel max and mean z-score across the day
    channel_max_z = {ch: float(np.max(zs)) for ch, zs in z_scores_by_channel.items()}
    channel_mean_z = {ch: float(np.mean(zs)) for ch, zs in z_scores_by_channel.items()}

    return {
        "csv_name": csv_path.name,
        "n_timesteps": len(l1_scores),
        "l1_mean": float(np.mean(l1_scores)),
        "l1_max": float(np.max(l1_scores)),
        "l1_p95": float(np.percentile(l1_scores, 95)),
        "l1_p50": float(np.percentile(l1_scores, 50)),
        "l2_mean": float(np.mean(l2_scores)) if l2_scores else None,
        "l2_max": float(np.max(l2_scores)) if l2_scores else None,
        "l2_p95": float(np.percentile(l2_scores, 95)) if l2_scores else None,
        "channel_max_z": channel_max_z,
        "channel_mean_z": channel_mean_z,
    }


def process_group(
    detector,
    csv_names: list[str],
    data_dir: Path,
    label: str,
) -> list[dict]:
    results = []
    for i, name in enumerate(csv_names, 1):
        csv_path = data_dir / "test" / name
        if not csv_path.exists():
            csv_path = data_dir / "train" / name
        if not csv_path.exists():
            print(f"  [{i}/{len(csv_names)}] NOT FOUND: {name}")
            continue

        print(f"  [{i}/{len(csv_names)}] {name}", end="", flush=True)
        result = run_day(detector, csv_path, pump_id=3)
        if result:
            result["group"] = label
            results.append(result)
            l2_str = f"{result['l2_max']:.4f}" if result['l2_max'] is not None else "N/A"
            print(f"  L1_max={result['l1_max']:.2f}  L2_max={l2_str}")
        else:
            print("  SKIPPED")
    return results


def print_summary(results: list[dict], l1_thresholds: dict, l2_thresholds: dict) -> None:
    missed = [r for r in results if r["group"] == "missed"]
    detected = [r for r in results if r["group"] == "detected"]

    p3_l1_warn = l1_thresholds["per_pump"]["3"]["mahalanobis"]["warning"]
    p3_l1_alarm = l1_thresholds["per_pump"]["3"]["mahalanobis"]["alarm"]
    p3_l1_mean = l1_thresholds["per_pump"]["3"]["mahalanobis"]["mean"]
    p3_l2_warn = l2_thresholds["per_pump"]["3"]["window_warning_smoothed"]
    p3_l2_alarm = l2_thresholds["per_pump"]["3"]["window_alarm_smoothed"]
    p3_l2_mean = l2_thresholds["per_pump"]["3"]["mean"]

    print("\n" + "="*70)
    print("INV-1 SUMMARY — Pump 3 Missed vs Detected Failure Days")
    print("="*70)

    print(f"\nPump 3 calibrated thresholds:")
    print(f"  L1 — mean={p3_l1_mean:.3f}, warning={p3_l1_warn:.3f}, alarm={p3_l1_alarm:.3f}")
    print(f"  L2 — mean={p3_l2_mean:.5f}, warning={p3_l2_warn:.5f}, alarm={p3_l2_alarm:.5f}")

    for grp_name, grp in [("MISSED (61 days)", missed), ("DETECTED (sample)", detected)]:
        l1_maxes = [r["l1_max"] for r in grp]
        l1_means = [r["l1_mean"] for r in grp]
        l2_maxes = [r["l2_max"] for r in grp if r["l2_max"] is not None]

        print(f"\n--- {grp_name} (n={len(grp)}) ---")
        print(f"  L1 max/day: mean={np.mean(l1_maxes):.3f}, median={np.median(l1_maxes):.3f}, "
              f"p90={np.percentile(l1_maxes, 90):.3f}, max={np.max(l1_maxes):.3f}")
        print(f"  L1 mean/day: mean={np.mean(l1_means):.3f}, median={np.median(l1_means):.3f}")
        if l2_maxes:
            print(f"  L2 max/day: mean={np.mean(l2_maxes):.5f}, median={np.median(l2_maxes):.5f}, "
                  f"p90={np.percentile(l2_maxes, 90):.5f}, max={np.max(l2_maxes):.5f}")

        # Fraction of missed days where L1 max reached warning
        frac_warn = sum(1 for v in l1_maxes if v >= p3_l1_warn) / len(l1_maxes) if l1_maxes else 0
        frac_alarm = sum(1 for v in l1_maxes if v >= p3_l1_alarm) / len(l1_maxes) if l1_maxes else 0
        print(f"  Days where L1 max >= warning ({p3_l1_warn:.2f}): {frac_warn:.1%}")
        print(f"  Days where L1 max >= alarm   ({p3_l1_alarm:.2f}): {frac_alarm:.1%}")

    # Channel analysis: which channels have highest mean z-score on MISSED days?
    if missed:
        print("\n--- Top anomalous channels on MISSED days (mean |z-score| across all timesteps+days) ---")
        channel_sums: dict[str, list[float]] = {}
        for r in missed:
            for ch, z in r["channel_mean_z"].items():
                channel_sums.setdefault(ch, []).append(z)
        channel_grand_mean = {ch: np.mean(vals) for ch, vals in channel_sums.items()}
        ranked = sorted(channel_grand_mean.items(), key=lambda x: x[1], reverse=True)
        for ch, z in ranked:
            det_vals = [r["channel_mean_z"].get(ch, 0) for r in detected]
            det_mean = np.mean(det_vals) if det_vals else 0
            marker = " **" if z > det_mean * 1.5 else ""
            print(f"  {ch[:50]:<50}: missed={z:.3f}, detected={det_mean:.3f}{marker}")

    # Score gap analysis for missed days
    print("\n--- L1 score gap on MISSED days ---")
    l1_maxes = [r["l1_max"] for r in missed]
    gaps_to_warning = [p3_l1_warn - v for v in l1_maxes]
    print(f"  Mean gap to warning: {np.mean(gaps_to_warning):.3f}")
    print(f"  Median gap to warning: {np.median(gaps_to_warning):.3f}")
    print(f"  Min gap (closest to warning): {np.min(gaps_to_warning):.3f}")
    close_to_warn = sum(1 for g in gaps_to_warning if g < 1.0)
    print(f"  Days within 1.0 of warning: {close_to_warn}")


def main() -> None:
    print("INV-1: Pump 3 Missed Failure Day Investigation")
    print("=" * 50)

    l1_thresholds, l2_thresholds = load_calibrated_thresholds()
    missed_names, detected_names = load_targets()

    print(f"Target: {len(missed_names)} missed, {len(detected_names)} detected (sample)")

    print("\nLoading detector with calibrated thresholds...")
    detector = create_streaming_detector(
        l1_thresholds=l1_thresholds,
        l2_thresholds=l2_thresholds,
    )

    print(f"\n--- Processing MISSED days ({len(missed_names)}) ---")
    missed_results = process_group(detector, missed_names, DATA_DIR, "missed")

    print(f"\n--- Processing DETECTED days ({len(detected_names)}) ---")
    detected_results = process_group(detector, detected_names, DATA_DIR, "detected")

    all_results = missed_results + detected_results
    print_summary(all_results, l1_thresholds, l2_thresholds)

    with open(RESULTS_OUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to: {RESULTS_OUT}")


if __name__ == "__main__":
    main()
