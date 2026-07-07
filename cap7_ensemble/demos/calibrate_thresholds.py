"""Calibrate demo thresholds from normal inference data.

Runs all normal operating days through the ensemble detector, collects
per-timestep L1 Mahalanobis and L2 smoothed MSE scores, and computes
P95/P99 thresholds adapted to the inference distribution.

Training thresholds are never modified. Calibrated thresholds are saved
to demos/calibrated_thresholds/ and auto-detected by demo scripts.

Usage:
    cd ensemble
    uv run python demos/calibrate_thresholds.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ENSEMBLE_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = ENSEMBLE_DIR.parent

sys.path.insert(0, str(ENSEMBLE_DIR))
sys.path.insert(0, str(WORKSPACE_ROOT))

from model.streaming import create_streaming_detector

_SEASON_MAP: dict[str, list[int]] = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}
_SEASONS: list[str] = ["winter", "spring", "summer", "autumn"]
_MIN_SEASONAL_SAMPLES = 50


def _month_to_season(month: int) -> str:
    for season, months in _SEASON_MAP.items():
        if month in months:
            return season
    return "winter"  # fallback for invalid month


# ---------------------------------------------------------------------------
# Data loading (adapted from streaming_demo.py)
# ---------------------------------------------------------------------------

def _parse_pump_from_filename(path: Path) -> int | None:
    match = re.match(r"pump_(\d+)_", path.name)
    return int(match.group(1)) if match else None


def _resample_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if len(df.index) < 3:
        return df.sort_index()
    diffs = pd.Series(df.index).sort_values().diff().dropna().dt.total_seconds() / 60.0
    if diffs.empty or float(diffs.median()) >= 3.0:
        return df.sort_index()
    return df.resample("5min").first().dropna(how="all").sort_index()


def _apply_speed_filter(df: pd.DataFrame) -> pd.DataFrame:
    speed_col = "Main HTF Pump Speed"
    if speed_col not in df.columns:
        return df
    speed_series = pd.to_numeric(df[speed_col], errors="coerce")
    peak_speed = float(speed_series.max()) if not speed_series.empty else float("nan")
    if not pd.notna(peak_speed) or peak_speed <= 0.0:
        return df
    stable_threshold = peak_speed * 0.90
    stable_mask = (speed_series >= stable_threshold).fillna(False)
    stable_indices = df.index[stable_mask]
    if len(stable_indices) < 10:
        return df
    return df.loc[stable_indices[0] : stable_indices[-1]]


def _load_day(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {csv_path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if df.empty:
        return df
    df = _resample_if_needed(df)
    df = _apply_speed_filter(df)
    if df.empty:
        return df
    return df.dropna(how="all")


def _resolve_data_dir(data_dir: str) -> Path:
    candidate = Path(data_dir).expanduser()
    if candidate.exists():
        return candidate.resolve()
    alt = (SCRIPT_DIR / data_dir).resolve()
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Data directory not found: {data_dir}")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _parse_date_from_filename(path: Path) -> date | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if not match:
        return None
    try:
        y, m, d = match.group(1).split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def _collect_scores(
    data_dir: Path,
    detector,
) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    """Run normal data through the detector and collect raw scores.

    Returns (l1_scores_by_pump, l2_scores_by_pump).
    """
    normal_dir = data_dir / "train"
    if not normal_dir.exists():
        raise FileNotFoundError(f"Normal data directory not found: {normal_dir}")

    csv_files = sorted(normal_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {normal_dir}")

    l1_scores: dict[int, list[float]] = {}
    l2_scores: dict[int, list[float]] = {}

    for csv_path in csv_files:
        pump_id = _parse_pump_from_filename(csv_path)
        if pump_id is None:
            print(f"  Skipping {csv_path.name} (cannot parse pump ID)")
            continue

        df = _load_day(csv_path)
        if len(df) < 5:
            print(f"  Skipping {csv_path.name} (too few rows: {len(df)})")
            continue

        detector.reset_pump(pump_id)
        l1_scores.setdefault(pump_id, [])
        l2_scores.setdefault(pump_id, [])

        day_l1 = 0
        day_l2 = 0
        for ts, row in df.iterrows():
            result = detector.process_timestep(pump_id, ts, row)
            l1_scores[pump_id].append(result.l1_mahalanobis)
            day_l1 += 1
            if result.l2_smoothed_mse is not None:
                l2_scores[pump_id].append(result.l2_smoothed_mse)
                day_l2 += 1

        print(f"  {csv_path.name}: pump {pump_id}, {day_l1} L1 scores, {day_l2} L2 scores")

    return l1_scores, l2_scores


def _collect_scores_seasonal(
    data_dir: Path,
    detector,
) -> tuple[dict[int, dict[str, list[float]]], dict[int, dict[str, list[float]]]]:
    """Run normal data through the detector and collect scores bucketed by season.

    Returns (l1_seasonal[pump_id][season], l2_seasonal[pump_id][season]).
    """
    normal_dir = data_dir / "train"
    if not normal_dir.exists():
        raise FileNotFoundError(f"Normal data directory not found: {normal_dir}")

    csv_files = sorted(normal_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {normal_dir}")

    l1_seasonal: dict[int, dict[str, list[float]]] = {}
    l2_seasonal: dict[int, dict[str, list[float]]] = {}

    for csv_path in csv_files:
        pump_id = _parse_pump_from_filename(csv_path)
        if pump_id is None:
            continue

        file_date = _parse_date_from_filename(csv_path)
        if file_date is None:
            print(f"  Skipping {csv_path.name} (cannot parse date for season assignment)")
            continue
        season = _month_to_season(file_date.month)

        df = _load_day(csv_path)
        if len(df) < 5:
            continue

        detector.reset_pump(pump_id)
        l1_seasonal.setdefault(pump_id, {s: [] for s in _SEASONS})
        l2_seasonal.setdefault(pump_id, {s: [] for s in _SEASONS})

        for ts, row in df.iterrows():
            result = detector.process_timestep(pump_id, ts, row)
            l1_seasonal[pump_id][season].append(result.l1_mahalanobis)
            if result.l2_smoothed_mse is not None:
                l2_seasonal[pump_id][season].append(result.l2_smoothed_mse)

    return l1_seasonal, l2_seasonal


def _compute_seasonal_l2_thresholds(
    l2_seasonal: dict[int, dict[str, list[float]]],
    global_per_pump: dict[int, dict],
    p_warning: float,
    p_alarm: float,
) -> dict:
    """Build the seasonal L2 threshold dict from per-pump per-season score lists."""
    seasonal_out: dict[str, dict] = {}

    for season in _SEASONS:
        per_pump_season: dict[str, dict] = {}
        for pump_id, season_scores in sorted(l2_seasonal.items()):
            pump_id_str = str(pump_id)
            scores = season_scores.get(season, [])
            n = len(scores)

            if n < _MIN_SEASONAL_SAMPLES:
                # Fallback to global per-pump thresholds for this season.
                gp = global_per_pump.get(pump_id, {})
                per_pump_season[pump_id_str] = {
                    "warning": float(gp.get("warning", 0.0)),
                    "alarm": float(gp.get("alarm", 0.0)),
                    "n_samples": n,
                    "fallback_to_global": True,
                }
            else:
                arr = np.asarray(scores, dtype=float)
                per_pump_season[pump_id_str] = {
                    "warning": float(np.percentile(arr, p_warning)),
                    "alarm": float(np.percentile(arr, p_alarm)),
                    "n_samples": n,
                    "fallback_to_global": False,
                }

        seasonal_out[season] = {"per_pump": per_pump_season}

    return {
        "description": (
            "Per-season L2 thresholds calibrated from normal training window EMA-MSE values. "
            "Season mapping uses northern-hemisphere calendar months."
        ),
        "method": "seasonal-bucketed P95/P99 on EMA-smoothed per-timestep MSE",
        "seasons": _SEASONS,
        **{season: seasonal_out[season] for season in _SEASONS},
    }


def _compute_thresholds(
    scores_by_pump: dict[int, list[float]],
    p_warning: float,
    p_alarm: float,
    min_samples: int = 50,
) -> tuple[dict, dict[int, dict]]:
    """Compute global and per-pump threshold dicts from score distributions."""
    all_scores = []
    for pump_scores in scores_by_pump.values():
        all_scores.extend(pump_scores)

    if not all_scores:
        raise ValueError("No scores collected — cannot compute thresholds")

    arr = np.asarray(all_scores, dtype=float)
    global_block = {
        "warning": float(np.percentile(arr, p_warning)),
        "alarm": float(np.percentile(arr, p_alarm)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }

    per_pump: dict[int, dict] = {}
    for pump_id, pump_scores in sorted(scores_by_pump.items()):
        if len(pump_scores) < min_samples:
            per_pump[pump_id] = {"fallback_to_global": True}
            print(f"  Pump {pump_id}: only {len(pump_scores)} samples, falling back to global")
            continue
        parr = np.asarray(pump_scores, dtype=float)
        per_pump[pump_id] = {
            "fallback_to_global": False,
            "warning": float(np.percentile(parr, p_warning)),
            "alarm": float(np.percentile(parr, p_alarm)),
            "mean": float(np.mean(parr)),
            "std": float(np.std(parr)),
            "p95": float(np.percentile(parr, 95)),
            "p99": float(np.percentile(parr, 99)),
            "n_samples": len(pump_scores),
        }

    return global_block, per_pump


def _build_l1_output(global_block: dict, per_pump: dict[int, dict], n_samples: int) -> dict:
    return {
        "description": "Calibrated L1 thresholds from inference normal data distribution",
        "calibration_date": date.today().isoformat(),
        "n_calibration_samples": n_samples,
        "global": {
            "mahalanobis": global_block,
        },
        "per_pump": {
            str(pid): (
                {"fallback_to_global": True}
                if block.get("fallback_to_global")
                else {
                    "fallback_to_global": False,
                    "n_samples": block.get("n_samples", 0),
                    "mahalanobis": {
                        k: v for k, v in block.items()
                        if k not in ("fallback_to_global", "n_samples")
                    },
                }
            )
            for pid, block in per_pump.items()
        },
    }


def _build_l2_output(global_block: dict, per_pump: dict[int, dict], n_samples: int) -> dict:
    return {
        "description": "Calibrated L2 thresholds from inference normal data distribution",
        "calibration_date": date.today().isoformat(),
        "n_calibration_samples": n_samples,
        "global": {
            "window_warning_smoothed": global_block["warning"],
            "window_alarm_smoothed": global_block["alarm"],
            "mean": global_block["mean"],
            "std": global_block["std"],
            "smoothing_alpha": 0.3,
        },
        "per_pump": {
            str(pid): (
                {"fallback_to_global": True}
                if block.get("fallback_to_global")
                else {
                    "window_warning_smoothed": block["warning"],
                    "window_alarm_smoothed": block["alarm"],
                    "mean": block["mean"],
                    "n_samples": block.get("n_samples", 0),
                }
            )
            for pid, block in per_pump.items()
        },
    }


def _print_comparison(label: str, training_thresholds: dict, calibrated: dict) -> None:
    """Print old vs new thresholds side by side."""
    print(f"\n  {label} Threshold Comparison (training → calibrated):")
    print(f"  {'':>10s}  {'Training':>10s}  {'Calibrated':>10s}  {'Change':>10s}")

    if "mahalanobis" in calibrated.get("global", {}):
        cal_g = calibrated["global"]["mahalanobis"]
        trn_g = training_thresholds.get("global", {}).get("mahalanobis", {})
    else:
        cal_g = calibrated.get("global", {})
        trn_g = training_thresholds.get("global", {})
        key_map = {"warning": "window_warning_smoothed", "alarm": "window_alarm_smoothed"}
        cal_g = {
            "warning": cal_g.get("window_warning_smoothed", 0),
            "alarm": cal_g.get("window_alarm_smoothed", 0),
            "mean": cal_g.get("mean", 0),
        }
        trn_g = {
            "warning": trn_g.get("window_warning_smoothed", 0),
            "alarm": trn_g.get("window_alarm_smoothed", 0),
            "mean": trn_g.get("mean", 0),
        }

    for key in ("mean", "warning", "alarm"):
        old_val = float(trn_g.get(key, 0))
        new_val = float(cal_g.get(key, 0))
        diff = new_val - old_val
        sign = "+" if diff >= 0 else ""
        print(f"  {key:>10s}  {old_val:>10.4f}  {new_val:>10.4f}  {sign}{diff:>9.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate demo thresholds from inference normal data")
    parser.add_argument("--data-dir", default="../../data/inference_data", help="Path to inference data directory")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "calibrated_thresholds"), help="Where to save calibrated thresholds")
    parser.add_argument("--percentiles", default="95,99", help="Warning,alarm percentiles (default: 95,99)")
    parser.add_argument("--seasonal", action="store_true", help="Also compute and save per-season L2 thresholds")
    args = parser.parse_args()

    p_warning, p_alarm = (float(x) for x in args.percentiles.split(","))
    data_dir = _resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"Calibrating thresholds from normal data in {data_dir}/train/")
    print(f"Percentiles: P{p_warning:.0f} (warning), P{p_alarm:.0f} (alarm)")
    if args.seasonal:
        print("Seasonal mode: will also compute per-season L2 thresholds")
    print()

    print("Loading ensemble detector...")
    detector = create_streaming_detector()

    # Load training thresholds for comparison
    l1_train_path = WORKSPACE_ROOT / "ensemble" / "cond_reg_v2" / "model" / "weights" / "production_thresholds.json"
    l2_train_path = WORKSPACE_ROOT / "bin" / "final_metrics" / "production_thresholds.json"

    with open(l1_train_path, "r", encoding="utf-8") as f:
        l1_training = json.load(f)
    with open(l2_train_path, "r", encoding="utf-8") as f:
        l2_training = json.load(f)

    if args.seasonal:
        print("\nCollecting seasonal scores from normal inference data...")
        t0 = time.perf_counter()
        l1_seasonal, l2_seasonal = _collect_scores_seasonal(data_dir, detector)
        elapsed = time.perf_counter() - t0
        total_l1 = sum(len(v) for pump_s in l1_seasonal.values() for v in pump_s.values())
        total_l2 = sum(len(v) for pump_s in l2_seasonal.values() for v in pump_s.values())
        print(f"\nDone in {elapsed:.1f}s: {total_l1} L1 scores, {total_l2} L2 scores across all seasons")

        # Global per-pump L2 thresholds used as fallback for under-sampled (pump, season) combos.
        l2_scores_flat = {
            pump_id: [score for season_scores in season_map.values() for score in season_scores]
            for pump_id, season_map in l2_seasonal.items()
        }
        _, l2_per_pump_global = _compute_thresholds(l2_scores_flat, p_warning, p_alarm)
        l2_global_lookup = {
            pid: {"warning": block.get("warning", 0.0), "alarm": block.get("alarm", 0.0)}
            for pid, block in l2_per_pump_global.items()
        }

        print("\nComputing seasonal L2 thresholds...")
        l2_seasonal_output = _compute_seasonal_l2_thresholds(l2_seasonal, l2_global_lookup, p_warning, p_alarm)

        output_dir.mkdir(parents=True, exist_ok=True)
        l2_seasonal_path = output_dir / "l2_thresholds_seasonal.json"
        with open(l2_seasonal_path, "w", encoding="utf-8") as f:
            json.dump(l2_seasonal_output, f, indent=2)
        print(f"Seasonal L2 thresholds saved to {l2_seasonal_path}")

        # Also write the canonical location expected by Level2Detector.
        l2_canonical_seasonal = WORKSPACE_ROOT / "bin" / "final_metrics" / "production_thresholds_seasonal_l2.json"
        with open(l2_canonical_seasonal, "w", encoding="utf-8") as f:
            json.dump(l2_seasonal_output, f, indent=2)
        print(f"Canonical seasonal L2 thresholds written to {l2_canonical_seasonal}")
        return

    print("\nCollecting scores from normal inference data...")
    t0 = time.perf_counter()
    l1_scores, l2_scores = _collect_scores(data_dir, detector)
    elapsed = time.perf_counter() - t0

    total_l1 = sum(len(v) for v in l1_scores.values())
    total_l2 = sum(len(v) for v in l2_scores.values())
    print(f"\nDone in {elapsed:.1f}s: {total_l1} L1 scores, {total_l2} L2 scores")

    print("\nComputing calibrated thresholds...")
    l1_global, l1_per_pump = _compute_thresholds(l1_scores, p_warning, p_alarm)
    l1_output = _build_l1_output(l1_global, l1_per_pump, total_l1)

    l2_global, l2_per_pump = _compute_thresholds(l2_scores, p_warning, p_alarm)
    l2_output = _build_l2_output(l2_global, l2_per_pump, total_l2)

    output_dir.mkdir(parents=True, exist_ok=True)
    l1_out_path = output_dir / "l1_thresholds.json"
    l2_out_path = output_dir / "l2_thresholds.json"

    with open(l1_out_path, "w", encoding="utf-8") as f:
        json.dump(l1_output, f, indent=2)
    with open(l2_out_path, "w", encoding="utf-8") as f:
        json.dump(l2_output, f, indent=2)

    print(f"\nSaved calibrated thresholds to {output_dir}/")

    _print_comparison("L1 (Mahalanobis)", l1_training, l1_output)
    _print_comparison("L2 (Smoothed MSE)", l2_training, l2_output)

    print(f"\nDemos will auto-detect these thresholds on next run.")
    print(f"To reset to training thresholds, delete {output_dir}/")


if __name__ == "__main__":
    main()
