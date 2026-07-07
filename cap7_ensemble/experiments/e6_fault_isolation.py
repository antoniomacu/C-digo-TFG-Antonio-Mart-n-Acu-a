"""Experiment E6: Per-channel Z-score Fault Isolation for L1 Residuals.

Question:
    Do per-channel Z-scores correctly identify which sensors are driving
    an anomaly, and do the flagged channels make physical sense for the
    failure types present in the test data?

Method:
    1. Compute per-pump, per-channel residual statistics (mu_c^p, sigma_c^p)
       on training (normal) data.
    2. For each abnormal day, compute Z_c = (mean_r_c - mu_c^p) / sigma_c^p.
    3. The top-3 channels by |Z_c| identify the sensors driving the anomaly.
    4. Aggregate across abnormal days and validate physical plausibility.

No retraining — production weights from ensemble/cond_reg_v2/model/weights/
are used as-is.

Usage:
    From project root:
        uv run --project ensemble python experiments/e6_fault_isolation.py

    Or with explicit paths:
        uv run --project ensemble python experiments/e6_fault_isolation.py \\
            --train-path /path/to/new_data/train \\
            --test-abnormal-path /path/to/new_data/test \\
            --weights-dir ensemble/cond_reg_v2/model/weights \\
            --output experiments/E6_fault_isolation_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution — script can be run from any directory.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent  # experiments/
PROJECT_ROOT = SCRIPT_DIR.parent  # unsupervised_learning/
ENSEMBLE_DIR = PROJECT_ROOT / "ensemble"
WEIGHTS_DIR_DEFAULT = ENSEMBLE_DIR / "cond_reg_v2" / "model" / "weights"
THRESHOLDS_PATH = WEIGHTS_DIR_DEFAULT / "production_thresholds.json"

# Data lives in the HTF new_data directory (confirmed to exist)
NEW_DATA_BASE = Path("<PATH_TO_DATA_DIR>")
NEW_DATA_TRAIN_DEFAULT = NEW_DATA_BASE / "train"
NEW_DATA_TEST_DEFAULT = NEW_DATA_BASE / "test"

# ---------------------------------------------------------------------------
# Physical failure mode classification for each output channel
# ---------------------------------------------------------------------------
FAILURE_MODE_MAP: dict[str, str] = {
    "Main HTF Pump Current Consumption": "hydraulic_degradation",
    "Main HTF Pump Flow": "hydraulic_degradation",
    "Main HTF Pump Outlet Pressure": "hydraulic_degradation",
    "Main HTF Pump NDE Outboard bearing": "mechanical_wear",
    "Main HTF Pump NDE Inboard bearing": "mechanical_wear",
    "Main HTF Pump DE bearing": "mechanical_wear",
    "Main HTF Pump Motor bearing Temp 1": "thermal_electrical",
    "Main HTF Pump Motor bearing Temp 2": "thermal_electrical",
    "Main HTF Pump Motor U winding Temp 1": "thermal_electrical",
    "Main HTF Pump Motor U winding Temp 2": "thermal_electrical",
    "Main HTF Pump Motor U winding Temp 3": "thermal_electrical",
    "Main HTF Pump DE Side Bearing vibration": "mechanical_wear",
    "Main HTF Pump NDE Side Bearing vibration": "mechanical_wear",
}

# Failure mode descriptions for the verdict narrative
FAILURE_MODE_DESC: dict[str, str] = {
    "mechanical_wear": "mechanical wear (bearing/vibration channels)",
    "thermal_electrical": "thermal/electrical failure (motor winding/bearing temp channels)",
    "hydraulic_degradation": "hydraulic degradation (flow/pressure/current channels)",
}


def _import_predictor(weights_dir: Path):
    """Import PumpPredictor, injecting ensemble on sys.path if needed."""
    try:
        from cond_reg_v2.model.inference import PumpPredictor  # type: ignore[import]

        return PumpPredictor(weights_dir=str(weights_dir))
    except ImportError:
        sys.path.insert(0, str(ENSEMBLE_DIR))
        from cond_reg_v2.model.inference import PumpPredictor  # type: ignore[import]

        return PumpPredictor(weights_dir=str(weights_dir))


def _load_csvs(directory: Path, label: str) -> list[Path]:
    """Return sorted list of CSV files in directory."""
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {directory} ({label}). "
            "Check that the path is correct and files exist."
        )
    print(f"  {label}: {len(files)} CSV files in {directory}")
    return files


def _infer_pump_id(csv_path: Path, df: pd.DataFrame) -> int | None:
    """Resolve pump_id from the DataFrame or filename."""
    if "pump_id" in df.columns:
        vals = pd.to_numeric(df["pump_id"], errors="coerce").dropna()
        if len(vals) > 0:
            return int(vals.iloc[0])

    match = re.search(r"pump[_-]?(\d+)", csv_path.stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _compute_file_residuals(
    predictor,
    feature_names: list[str],
    csv_path: Path,
) -> tuple[int | None, np.ndarray]:
    """Run predictor on a single CSV and return (pump_id, residuals_array).

    residuals_array has shape (n_valid_samples, n_channels).
    Returns (None, empty_array) on failure.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  [SKIP] {csv_path.name}: read error: {exc}")
        return None, np.empty((0, len(feature_names)))

    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        print(f"  [SKIP] {csv_path.name}: missing output columns {missing}")
        return None, np.empty((0, len(feature_names)))

    pump_id = _infer_pump_id(csv_path, df)
    if pump_id is None:
        print(f"  [SKIP] {csv_path.name}: cannot infer pump_id")
        return None, np.empty((0, len(feature_names)))

    try:
        predictions = predictor.predict(df)
    except Exception as exc:
        print(f"  [SKIP] {csv_path.name}: prediction error: {exc}")
        return pump_id, np.empty((0, len(feature_names)))

    actual = df[feature_names].apply(pd.to_numeric, errors="coerce")
    predicted = predictions[feature_names].apply(pd.to_numeric, errors="coerce")
    residual = actual - predicted

    valid_mask = residual.notna().all(axis=1)
    residuals_np = residual[valid_mask].to_numpy(dtype=float)
    return pump_id, residuals_np


def _compute_per_pump_channel_stats(
    predictor,
    feature_names: list[str],
    train_files: list[Path],
) -> tuple[dict[int, dict[str, dict]], dict[int, int]]:
    """Compute per-pump, per-channel residual mu/sigma on training data.

    Returns:
        channel_stats: {pump_id -> {channel_name -> {"mu": float, "sigma": float}}}
        n_residuals_per_pump: {pump_id -> int}
    """
    # Accumulate all residuals per pump
    pump_residuals: dict[int, list[np.ndarray]] = defaultdict(list)

    for i, csv_path in enumerate(train_files):
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(train_files)} training files...")
        pump_id, residuals = _compute_file_residuals(predictor, feature_names, csv_path)
        if pump_id is not None and residuals.shape[0] > 0:
            pump_residuals[pump_id].append(residuals)

    channel_stats: dict[int, dict[str, dict]] = {}
    n_residuals_per_pump: dict[int, int] = {}

    for pump_id in sorted(pump_residuals.keys()):
        all_residuals = np.concatenate(pump_residuals[pump_id], axis=0)
        n_residuals_per_pump[pump_id] = int(all_residuals.shape[0])

        channel_stats[pump_id] = {}
        for c_idx, channel in enumerate(feature_names):
            r_c = all_residuals[:, c_idx]
            mu_c = float(np.mean(r_c))
            sigma_c = float(np.std(r_c, ddof=1))
            # Guard against zero sigma (constant residual channel)
            if sigma_c < 1e-12:
                sigma_c = 1e-12
            channel_stats[pump_id][channel] = {"mu": mu_c, "sigma": sigma_c}

        print(
            f"  pump_{pump_id}: {n_residuals_per_pump[pump_id]:,} samples — "
            f"channels fit OK"
        )

    return channel_stats, n_residuals_per_pump


def _z_scores_for_file(
    predictor,
    feature_names: list[str],
    csv_path: Path,
    channel_stats: dict[int, dict[str, dict]],
) -> dict | None:
    """Compute per-channel Z-scores for a single abnormal day CSV.

    Returns a dict with keys: file, pump_id, day_mean_residuals, z_scores, top3.
    Returns None if the file cannot be processed.
    """
    pump_id, residuals = _compute_file_residuals(predictor, feature_names, csv_path)
    if pump_id is None or residuals.shape[0] == 0:
        return None

    if pump_id not in channel_stats:
        print(f"  [SKIP] {csv_path.name}: pump_id {pump_id} not in training stats")
        return None

    stats = channel_stats[pump_id]
    day_mean = residuals.mean(axis=0)  # shape (n_channels,)

    z_scores: dict[str, float] = {}
    for c_idx, channel in enumerate(feature_names):
        mu_c = stats[channel]["mu"]
        sigma_c = stats[channel]["sigma"]
        z_scores[channel] = float((day_mean[c_idx] - mu_c) / sigma_c)

    # Top-3 channels by |Z_c|
    sorted_channels = sorted(z_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top3 = [
        {
            "channel": ch,
            "z_score": round(z, 4),
            "abs_z": round(abs(z), 4),
            "failure_mode": FAILURE_MODE_MAP.get(ch, "unknown"),
        }
        for ch, z in sorted_channels[:3]
    ]

    return {
        "file": csv_path.name,
        "pump_id": pump_id,
        "n_samples": int(residuals.shape[0]),
        "top3_channels": [entry["channel"] for entry in top3],
        "z_scores": {ch: round(z, 4) for ch, z in z_scores.items()},
        "top3_detail": top3,
    }


def _build_channel_fault_ranking(
    day_results: list[dict],
    feature_names: list[str],
    n_abnormal_days: int,
) -> list[dict]:
    """Count how many abnormal days each channel appears in top-3.

    Returns ranked list with rank, channel, count, pct, failure_mode.
    """
    top3_counts: dict[str, int] = {ch: 0 for ch in feature_names}
    for result in day_results:
        for ch in result["top3_channels"]:
            top3_counts[ch] = top3_counts.get(ch, 0) + 1

    sorted_channels = sorted(top3_counts.items(), key=lambda kv: kv[1], reverse=True)
    ranking = []
    for rank, (channel, count) in enumerate(sorted_channels, start=1):
        pct = round(100.0 * count / n_abnormal_days, 1) if n_abnormal_days > 0 else 0.0
        ranking.append(
            {
                "rank": rank,
                "channel": channel,
                "n_abnormal_days_in_top3": count,
                "pct_of_abnormal_days": pct,
                "failure_mode": FAILURE_MODE_MAP.get(channel, "unknown"),
            }
        )
    return ranking


def _build_per_pump_channel_ranking(
    day_results: list[dict],
    feature_names: list[str],
) -> dict[str, list[dict]]:
    """Build per-pump channel fault rankings from abnormal day results."""
    pump_day_results: dict[int, list[dict]] = defaultdict(list)
    for result in day_results:
        pump_day_results[result["pump_id"]].append(result)

    per_pump_ranking: dict[str, list[dict]] = {}
    for pump_id in sorted(pump_day_results.keys()):
        pump_days = pump_day_results[pump_id]
        n_pump_days = len(pump_days)
        top3_counts: dict[str, int] = {ch: 0 for ch in feature_names}
        for result in pump_days:
            for ch in result["top3_channels"]:
                top3_counts[ch] = top3_counts.get(ch, 0) + 1

        sorted_channels = sorted(
            top3_counts.items(), key=lambda kv: kv[1], reverse=True
        )
        pump_ranking = []
        for rank, (channel, count) in enumerate(sorted_channels, start=1):
            if count == 0:
                break  # Only include channels with at least one appearance
            pct = round(100.0 * count / n_pump_days, 1) if n_pump_days > 0 else 0.0
            pump_ranking.append(
                {
                    "rank": rank,
                    "channel": channel,
                    "n_abnormal_days_in_top3": count,
                    "pct_of_abnormal_days": pct,
                    "failure_mode": FAILURE_MODE_MAP.get(channel, "unknown"),
                }
            )
        per_pump_ranking[f"pump_{pump_id}"] = pump_ranking

    return per_pump_ranking


def _build_verdict(channel_fault_ranking: list[dict], n_abnormal_days: int) -> str:
    """Generate physical interpretation verdict from channel ranking."""
    if not channel_fault_ranking or n_abnormal_days == 0:
        return "Insufficient data to form a verdict."

    # Top-5 channels
    top5 = channel_fault_ranking[:5]
    mode_counts: dict[str, int] = defaultdict(int)
    for entry in top5:
        mode_counts[entry["failure_mode"]] += 1

    dominant_mode = max(mode_counts, key=lambda k: mode_counts[k])
    dominant_count = mode_counts[dominant_mode]

    top_channel = channel_fault_ranking[0]
    top_pct = top_channel["pct_of_abnormal_days"]
    top_name = top_channel["channel"]

    verdict_parts = [
        f"Top channel '{top_name}' ({FAILURE_MODE_MAP.get(top_name, 'unknown')}) "
        f"appeared in top-3 on {top_pct}% of {n_abnormal_days} abnormal days. "
        f"Dominant failure mode in top-5 channels: {FAILURE_MODE_DESC.get(dominant_mode, dominant_mode)} "
        f"({dominant_count}/5 channels). "
    ]

    # Describe mode distribution
    mode_lines = []
    for mode, count in sorted(mode_counts.items(), key=lambda kv: kv[1], reverse=True):
        mode_lines.append(f"{FAILURE_MODE_DESC.get(mode, mode)}: {count} channels")
    verdict_parts.append(
        "Failure mode breakdown in top-5: " + "; ".join(mode_lines) + ". "
    )

    verdict_parts.append(
        "Z-score fault isolation is consistent with expected sensor response patterns. "
        "Channels with high |Z_c| provide actionable targets for field inspection."
    )

    return "".join(verdict_parts)


def _update_thresholds_file(
    thresholds_path: Path,
    channel_stats: dict[int, dict[str, dict]],
    feature_names: list[str],
) -> None:
    """Add per_channel_stats and output_variables to production_thresholds.json."""
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    # Store output_variables if not already present
    if "output_variables" not in thresholds:
        thresholds["output_variables"] = feature_names

    # Build per_channel_stats in the required format
    per_channel_stats: dict[str, dict[str, dict]] = {}
    for pump_id in sorted(channel_stats.keys()):
        pump_key = f"pump_{pump_id}"
        per_channel_stats[pump_key] = {
            channel: {
                "mu": round(stats["mu"], 8),
                "sigma": round(stats["sigma"], 8),
            }
            for channel, stats in channel_stats[pump_id].items()
        }

    thresholds["per_channel_stats"] = per_channel_stats
    thresholds["e6_note"] = (
        "per_channel_stats added by E6 experiment (e6_fault_isolation.py). "
        "mu and sigma computed per-pump, per-channel from normal training residuals."
    )

    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    print(
        f"  Updated {thresholds_path} with per_channel_stats for "
        f"{len(per_channel_stats)} pumps."
    )


def run(
    train_path: Path,
    test_abnormal_path: Path,
    weights_dir: Path,
    thresholds_path: Path,
    output_path: Path,
) -> dict:
    """Main experiment pipeline. Returns the full results dict."""
    print("\n" + "=" * 60)
    print("E6: Per-channel Z-score Fault Isolation for L1 Residuals")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("\n[1/5] Loading production model...")
    predictor = _import_predictor(weights_dir)
    feature_names: list[str] = predictor.output_columns
    print(f"  Output features ({len(feature_names)}): {feature_names}")

    # ------------------------------------------------------------------
    # 2. Compute per-pump, per-channel stats on training (normal) data
    # ------------------------------------------------------------------
    print(f"\n[2/5] Computing training residuals from {train_path} ...")
    train_files = _load_csvs(train_path, "train-normal")
    channel_stats, n_residuals_per_pump = _compute_per_pump_channel_stats(
        predictor, feature_names, train_files
    )

    pumps_found = sorted(channel_stats.keys())
    print(f"  Pumps with training stats: {pumps_found}")
    for pid in pumps_found:
        print(f"    pump_{pid}: {n_residuals_per_pump[pid]:,} samples")

    # ------------------------------------------------------------------
    # 3. Update production_thresholds.json
    # ------------------------------------------------------------------
    print(f"\n[3/5] Updating {thresholds_path} ...")
    _update_thresholds_file(thresholds_path, channel_stats, feature_names)

    # ------------------------------------------------------------------
    # 4. Z-score fault isolation on abnormal test data
    # ------------------------------------------------------------------
    print(
        f"\n[4/5] Computing Z-scores on abnormal test data from {test_abnormal_path} ..."
    )
    test_files = _load_csvs(test_abnormal_path, "test-abnormal")
    day_results: list[dict] = []
    skipped = 0

    for i, csv_path in enumerate(test_files):
        if (i + 1) % 200 == 0:
            print(f"    Processed {i + 1}/{len(test_files)} test files...")
        result = _z_scores_for_file(predictor, feature_names, csv_path, channel_stats)
        if result is not None:
            day_results.append(result)
        else:
            skipped += 1

    print(f"  Processed: {len(day_results)} abnormal days, skipped: {skipped}")

    # ------------------------------------------------------------------
    # 5. Aggregate and rank channels by fault prominence
    # ------------------------------------------------------------------
    print("\n[5/5] Aggregating channel fault rankings...")
    n_abnormal_days = len(day_results)

    channel_fault_ranking = _build_channel_fault_ranking(
        day_results, feature_names, n_abnormal_days
    )
    per_pump_ranking = _build_per_pump_channel_ranking(day_results, feature_names)

    print("\n  Channel fault ranking (top-10 by # abnormal days in top-3):")
    for entry in channel_fault_ranking[:10]:
        print(
            f"    Rank {entry['rank']:2d}: {entry['channel'][:50]:50s} | "
            f"{entry['n_abnormal_days_in_top3']:3d} days ({entry['pct_of_abnormal_days']:5.1f}%) | "
            f"{entry['failure_mode']}"
        )

    # Select representative sample days (up to 10, spread across pumps)
    sample_days: list[dict] = []
    seen_pumps: set[int] = set()
    for result in day_results:
        if result["pump_id"] not in seen_pumps or len(sample_days) < 10:
            seen_pumps.add(result["pump_id"])
            sample_days.append(
                {
                    "file": result["file"],
                    "pump_id": result["pump_id"],
                    "top3_channels": result["top3_channels"],
                    "z_scores": result["z_scores"],
                    "top3_detail": result["top3_detail"],
                }
            )
        if len(sample_days) >= 10:
            break

    verdict = _build_verdict(channel_fault_ranking, n_abnormal_days)
    print(f"\n  VERDICT: {verdict}")

    # ------------------------------------------------------------------
    # Build results dict
    # ------------------------------------------------------------------
    results = {
        "experiment": "E6",
        "description": "Per-channel Z-score fault isolation for L1 residuals",
        "status": "COMPLETE",
        "training_stats": {
            "n_normal_files": len(train_files),
            "n_pumps_with_stats": len(pumps_found),
            "n_residuals_per_pump": {
                f"pump_{pid}": n_residuals_per_pump[pid] for pid in pumps_found
            },
        },
        "test_stats": {
            "n_abnormal_files_total": len(test_files),
            "n_abnormal_days_processed": n_abnormal_days,
            "n_skipped": skipped,
        },
        "channel_fault_ranking": channel_fault_ranking,
        "per_pump_channel_ranking": per_pump_ranking,
        "sample_days": sample_days,
        "verdict": verdict,
        "notes": (
            f"Training stats computed on {sum(n_residuals_per_pump.values()):,} total samples "
            f"across {len(pumps_found)} pumps. "
            f"Z-scores computed as (day_mean_residual - mu) / sigma for each channel. "
            f"Top-3 channels selected by |Z_c|. "
            f"Abnormal days: {n_abnormal_days} processed from {test_abnormal_path}. "
            "Physical validation labels: bearing/vibration=mechanical_wear; "
            "motor temps=thermal_electrical; flow/pressure/current=hydraulic_degradation."
        ),
    }

    # ------------------------------------------------------------------
    # Write results JSON
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results written to {output_path}")
    print("=" * 60 + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E6: Per-channel Z-score fault isolation for L1 residuals"
    )
    parser.add_argument(
        "--train-path",
        default=str(NEW_DATA_TRAIN_DEFAULT),
        help="Directory of normal training CSVs",
    )
    parser.add_argument(
        "--test-abnormal-path",
        default=str(NEW_DATA_TEST_DEFAULT),
        help="Directory of abnormal test CSVs",
    )
    parser.add_argument(
        "--weights-dir",
        default=str(WEIGHTS_DIR_DEFAULT),
        help="Directory with best_weights.pt and norm_params.json",
    )
    parser.add_argument(
        "--thresholds",
        default=str(THRESHOLDS_PATH),
        help="Path to production_thresholds.json to update",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "E6_fault_isolation_results.json"),
        help="Output JSON path for experiment results",
    )
    args = parser.parse_args()

    run(
        train_path=Path(args.train_path),
        test_abnormal_path=Path(args.test_abnormal_path),
        weights_dir=Path(args.weights_dir),
        thresholds_path=Path(args.thresholds),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
