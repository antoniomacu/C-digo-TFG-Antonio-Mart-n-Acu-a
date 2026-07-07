"""
P8 — Joint L1+L2 Threshold Optimisation.

Finds jointly optimised warning/alarm thresholds for Level 1 (Mahalanobis) and
Level 2 (smoothed MSE) that minimise ensemble false-alarm rate at K>=3 without
degrading per-pump detection rates.

Usage:
    uv run python experiments/p8_joint_threshold_optimization.py [OPTIONS]

Options:
    --skip-extraction       Use cached scores.parquet (must exist)
    --force-extraction      Re-extract even if cache exists
    --promote               Auto-promote thresholds if acceptance criteria met
    --grid-resolution N     Points per threshold dimension (default: 20)
    --opt-fraction F        Fraction for optimisation set (default: 0.7)
    --min-alert-steps K     Min alarm timesteps for day-level flag (default: 3)
    --data-dir PATH         Override data directory
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENSEMBLE_DIR = PROJECT_ROOT / "ensemble"
SCORES_DIR = SCRIPT_DIR / "p8_scores"

DEFAULT_DATA_DIR = Path("<PATH_TO_DATA_DIR>")
L1_THRESHOLDS_PATH = ENSEMBLE_DIR / "cond_reg_v2" / "model" / "weights" / "production_thresholds.json"
L2_THRESHOLDS_PATH = PROJECT_ROOT / "bin" / "final_metrics" / "production_thresholds.json"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
console = Console()

# Fusion matrix: (l1_status_int, l2_status_int) -> ensemble_status_int
# 0=NORMAL, 1=WARNING, 2=ALARM
FUSION_MATRIX = np.array([
    [0, 1, 2],  # L1=NORMAL:  L2=NORMAL->NORMAL, L2=WARNING->WARNING, L2=ALARM->ALARM
    [1, 2, 2],  # L1=WARNING: L2=NORMAL->WARNING, L2=WARNING->ALARM,  L2=ALARM->ALARM
    [2, 2, 2],  # L1=ALARM:   L2=NORMAL->ALARM,   L2=WARNING->ALARM,  L2=ALARM->ALARM
], dtype=np.int8)

STATUS_ALARM = 2


# ---------------------------------------------------------------------------
# Phase 1: Score Extraction
# ---------------------------------------------------------------------------

def _parse_pump_from_filename(path: Path) -> int | None:
    match = re.match(r"pump_(\d+)_", path.name)
    return int(match.group(1)) if match else None


def _load_day(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess a pump-day CSV (same logic as streaming_demo)."""
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {csv_path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    if df.empty:
        return df

    # Resample if needed (sub-5min data)
    if len(df.index) >= 3:
        diffs = pd.Series(df.index).sort_values().diff().dropna().dt.total_seconds() / 60.0
        if not diffs.empty and float(diffs.median()) < 3.0:
            df = df.resample("5min").first().dropna(how="all").sort_index()

    # Speed filter
    speed_col = "Main HTF Pump Speed"
    if speed_col in df.columns:
        speed_series = pd.to_numeric(df[speed_col], errors="coerce")
        peak_speed = float(speed_series.max()) if not speed_series.empty else float("nan")
        if pd.notna(peak_speed) and peak_speed > 0.0:
            stable_threshold = peak_speed * 0.90
            stable_mask = (speed_series >= stable_threshold).fillna(False)
            stable_indices = df.index[stable_mask]
            if len(stable_indices) >= 10:
                df = df.loc[stable_indices[0]:stable_indices[-1]]

    df = df.dropna(how="all")
    return df


def extract_scores(data_dir: Path) -> pd.DataFrame:
    """Run L1+L2 inference on all days, return per-timestep scores DataFrame."""
    sys.path.insert(0, str(ENSEMBLE_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))

    from model.streaming import create_streaming_detector

    detector = create_streaming_detector(device="cpu")

    records: list[dict] = []
    sources = [
        (data_dir / "train", "normal"),
        (data_dir / "test", "abnormal"),
    ]

    all_csvs: list[tuple[Path, str, int]] = []
    for source_dir, label in sources:
        if not source_dir.exists():
            logger.warning("Source directory not found: %s", source_dir)
            continue
        for csv_path in sorted(source_dir.glob("pump_*.csv")):
            pump_id = _parse_pump_from_filename(csv_path)
            if pump_id is not None:
                all_csvs.append((csv_path, label, pump_id))

    console.print(f"[bold]Phase 1:[/bold] Extracting scores from {len(all_csvs)} CSVs...")

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(all_csvs))

        for csv_path, label, pump_id in all_csvs:
            try:
                df = _load_day(csv_path)
            except Exception as exc:
                logger.debug("Failed to load %s: %s", csv_path.name, exc)
                progress.advance(task)
                continue

            if len(df) < 5:
                progress.advance(task)
                continue

            detector.reset_pump(pump_id)

            for idx, (ts, row) in enumerate(df.iterrows()):
                try:
                    result = detector.process_timestep(pump_id, ts, row)
                    records.append({
                        "csv_name": csv_path.name,
                        "pump_id": pump_id,
                        "label": label,
                        "timestep_idx": idx,
                        "l1_mahalanobis": float(result.l1_mahalanobis),
                        "l2_mse": float(result.l2_smoothed_mse) if result.l2_smoothed_mse is not None else float("nan"),
                    })
                except Exception:
                    pass

            progress.advance(task)

    scores_df = pd.DataFrame(records)
    scores_df["pump_id"] = scores_df["pump_id"].astype(np.int8)
    scores_df["timestep_idx"] = scores_df["timestep_idx"].astype(np.int16)
    scores_df["l1_mahalanobis"] = scores_df["l1_mahalanobis"].astype(np.float32)
    scores_df["l2_mse"] = scores_df["l2_mse"].astype(np.float32)

    console.print(f"  Extracted {len(scores_df):,} timestep scores from {scores_df['csv_name'].nunique()} days.")
    return scores_df


# ---------------------------------------------------------------------------
# Phase 2: Data Split
# ---------------------------------------------------------------------------

def split_days(scores_df: pd.DataFrame, opt_fraction: float = 0.7, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into optimisation and held-out sets, stratified by (pump_id, label)."""
    day_info = scores_df.groupby("csv_name").agg(
        pump_id=("pump_id", "first"),
        label=("label", "first"),
    ).reset_index()

    strat_key = day_info["pump_id"].astype(str) + "_" + day_info["label"]

    opt_days, holdout_days = train_test_split(
        day_info["csv_name"],
        train_size=opt_fraction,
        random_state=seed,
        stratify=strat_key,
    )

    opt_set = scores_df[scores_df["csv_name"].isin(set(opt_days))].copy()
    holdout_set = scores_df[scores_df["csv_name"].isin(set(holdout_days))].copy()

    console.print(
        f"  Split: opt={opt_set['csv_name'].nunique()} days ({len(opt_set):,} timesteps), "
        f"holdout={holdout_set['csv_name'].nunique()} days ({len(holdout_set):,} timesteps)"
    )
    return opt_set, holdout_set


# ---------------------------------------------------------------------------
# Phase 3: Grid Search
# ---------------------------------------------------------------------------

def _precompute_day_structure(scores_df: pd.DataFrame) -> dict:
    """Pre-compute day boundary indices for fast vectorised evaluation.

    Returns a dict with numpy arrays for efficient per-day aggregation without
    pandas groupby in the inner loop.
    """
    # Assign integer day IDs (preserving order of first appearance)
    csv_names = scores_df["csv_name"].values
    unique_days, day_ids = np.unique(csv_names, return_inverse=True)

    n_days = len(unique_days)

    # Per-day metadata (pump_id, label) — take first occurrence
    day_pump = np.empty(n_days, dtype=np.int8)
    day_label_is_normal = np.empty(n_days, dtype=bool)

    for i, day_name in enumerate(unique_days):
        first_idx = np.where(day_ids == i)[0][0]
        day_pump[i] = scores_df.iloc[first_idx]["pump_id"]
        day_label_is_normal[i] = scores_df.iloc[first_idx]["label"] == "normal"

    # Per-timestep pump_id for per-pump threshold application
    pump_ids = scores_df["pump_id"].values.astype(np.int8)

    return {
        "l1_scores": scores_df["l1_mahalanobis"].values.astype(np.float32),
        "l2_scores": scores_df["l2_mse"].values.astype(np.float32),
        "pump_ids": pump_ids,
        "day_ids": day_ids,
        "n_days": n_days,
        "unique_days": unique_days,
        "day_pump": day_pump,
        "day_label_is_normal": day_label_is_normal,
    }


def _evaluate_thresholds_fast(
    precomp: dict,
    l1_warn: float,
    l1_alarm: float,
    l2_warn: float,
    l2_alarm: float,
    min_alert_steps: int,
) -> dict:
    """Evaluate a threshold combo using pre-computed structure (pure numpy).

    Uses uniform thresholds for all pumps (global mode).
    """
    l1 = precomp["l1_scores"]
    l2 = precomp["l2_scores"]
    day_ids = precomp["day_ids"]
    n_days = precomp["n_days"]

    # Per-timestep classification
    l1_status = np.zeros(len(l1), dtype=np.int8)
    l1_status[l1 >= l1_warn] = 1
    l1_status[l1 >= l1_alarm] = 2

    l2_status = np.zeros(len(l2), dtype=np.int8)
    l2_valid = ~np.isnan(l2)
    l2_status[l2_valid & (l2 >= l2_warn)] = 1
    l2_status[l2_valid & (l2 >= l2_alarm)] = 2

    # Fusion
    ensemble = FUSION_MATRIX[l1_status, l2_status]
    # Count WARNING (1) + ALARM (2) steps per day — matches batch_evaluate logic
    is_alert = (ensemble >= 1)  # WARNING or ALARM

    # Count alert steps per day using bincount
    alert_per_day = np.bincount(day_ids, weights=is_alert.astype(np.float32), minlength=n_days)
    day_flagged = alert_per_day >= min_alert_steps

    # Metrics
    is_normal = precomp["day_label_is_normal"]
    n_normal = int(is_normal.sum())
    n_false_alarms = int((day_flagged & is_normal).sum())
    fa_rate = n_false_alarms / n_normal if n_normal > 0 else 1.0

    # Per-pump detection
    is_abnormal = ~is_normal
    day_pump = precomp["day_pump"]
    per_pump_detection = {}
    for pump in np.unique(day_pump):
        pump_abn_mask = is_abnormal & (day_pump == pump)
        n_abn = int(pump_abn_mask.sum())
        if n_abn > 0:
            n_det = int((day_flagged & pump_abn_mask).sum())
            per_pump_detection[int(pump)] = n_det / n_abn
        else:
            per_pump_detection[int(pump)] = 0.0

    n_abnormal = int(is_abnormal.sum())
    overall_detection = int((day_flagged & is_abnormal).sum()) / n_abnormal if n_abnormal > 0 else 0.0

    return {
        "fa_rate": fa_rate,
        "n_false_alarms": n_false_alarms,
        "n_normal_days": n_normal,
        "overall_detection": overall_detection,
        "per_pump_detection": per_pump_detection,
    }


def _evaluate_per_pump_thresholds_fast(
    precomp: dict,
    l1_warn_per_pump: dict[int, float],
    l1_alarm_per_pump: dict[int, float],
    l2_warn_per_pump: dict[int, float],
    l2_alarm_per_pump: dict[int, float],
    min_alert_steps: int,
) -> dict:
    """Evaluate per-pump thresholds using pre-computed structure (pure numpy)."""
    l1 = precomp["l1_scores"]
    l2 = precomp["l2_scores"]
    pump_ids = precomp["pump_ids"]
    day_ids = precomp["day_ids"]
    n_days = precomp["n_days"]

    # Build per-timestep threshold arrays based on pump_id
    unique_pumps = sorted(set(l1_warn_per_pump.keys()))
    l1_warn_arr = np.zeros(len(l1), dtype=np.float32)
    l1_alarm_arr = np.zeros(len(l1), dtype=np.float32)
    l2_warn_arr = np.zeros(len(l2), dtype=np.float32)
    l2_alarm_arr = np.zeros(len(l2), dtype=np.float32)

    for pump in unique_pumps:
        mask = pump_ids == pump
        l1_warn_arr[mask] = l1_warn_per_pump[pump]
        l1_alarm_arr[mask] = l1_alarm_per_pump[pump]
        l2_warn_arr[mask] = l2_warn_per_pump[pump]
        l2_alarm_arr[mask] = l2_alarm_per_pump[pump]

    # Per-timestep classification
    l1_status = np.zeros(len(l1), dtype=np.int8)
    l1_status[l1 >= l1_warn_arr] = 1
    l1_status[l1 >= l1_alarm_arr] = 2

    l2_status = np.zeros(len(l2), dtype=np.int8)
    l2_valid = ~np.isnan(l2)
    l2_status[l2_valid & (l2 >= l2_warn_arr)] = 1
    l2_status[l2_valid & (l2 >= l2_alarm_arr)] = 2

    # Fusion + alert counting
    ensemble = FUSION_MATRIX[l1_status, l2_status]
    is_alert = (ensemble >= 1)

    alert_per_day = np.bincount(day_ids, weights=is_alert.astype(np.float32), minlength=n_days)
    day_flagged = alert_per_day >= min_alert_steps

    # Metrics
    is_normal = precomp["day_label_is_normal"]
    n_normal = int(is_normal.sum())
    n_false_alarms = int((day_flagged & is_normal).sum())
    fa_rate = n_false_alarms / n_normal if n_normal > 0 else 1.0

    is_abnormal = ~is_normal
    day_pump = precomp["day_pump"]
    per_pump_detection = {}
    for pump in unique_pumps:
        pump_abn_mask = is_abnormal & (day_pump == pump)
        n_abn = int(pump_abn_mask.sum())
        if n_abn > 0:
            n_det = int((day_flagged & pump_abn_mask).sum())
            per_pump_detection[int(pump)] = n_det / n_abn
        else:
            per_pump_detection[int(pump)] = 0.0

    n_abnormal = int(is_abnormal.sum())
    overall_detection = int((day_flagged & is_abnormal).sum()) / n_abnormal if n_abnormal > 0 else 0.0

    return {
        "fa_rate": fa_rate,
        "n_false_alarms": n_false_alarms,
        "n_normal_days": n_normal,
        "overall_detection": overall_detection,
        "per_pump_detection": per_pump_detection,
    }


def _evaluate_thresholds(
    scores_df: pd.DataFrame,
    l1_warn: float,
    l1_alarm: float,
    l2_warn: float,
    l2_alarm: float,
    min_alert_steps: int,
) -> dict:
    """Evaluate a single threshold combo (convenience wrapper using precompute)."""
    precomp = _precompute_day_structure(scores_df)
    return _evaluate_thresholds_fast(precomp, l1_warn, l1_alarm, l2_warn, l2_alarm, min_alert_steps)


def grid_search(
    opt_set: pd.DataFrame,
    resolution: int,
    min_alert_steps: int,
    baseline_detection: dict[int, float],
    baseline_per_pump: dict,
) -> pd.DataFrame:
    """Exhaustive grid search over 4 threshold multipliers applied to per-pump thresholds.

    Searches multipliers for (l1_warn, l1_alarm, l2_warn, l2_alarm) applied uniformly
    to all per-pump thresholds. This preserves relative per-pump differences while
    jointly optimizing the overall sensitivity.
    """
    # Multiplier ranges: search from slightly below to well above current
    l1_warn_mults = np.linspace(0.70, 1.30, resolution)
    l1_alarm_mults = np.linspace(0.70, 1.30, resolution)
    l2_warn_mults = np.linspace(0.70, 1.30, resolution)
    l2_alarm_mults = np.linspace(0.70, 1.30, resolution)

    console.print(
        f"[bold]Phase 3:[/bold] Grid search over threshold multipliers "
        f"({resolution}^4 = {resolution**4:,} combos)..."
    )
    console.print(f"  Multiplier range: [0.70, 1.30] for each of 4 threshold types")
    console.print(f"  Applied to per-pump thresholds (preserves relative differences)")

    precomp = _precompute_day_structure(opt_set)
    pumps = sorted(baseline_per_pump["l1_warn_per_pump"].keys())

    results: list[dict] = []
    n_evaluated = 0
    n_pruned_order = 0
    n_pruned_constraint = 0

    t_start = time.perf_counter()

    for l1_wm in l1_warn_mults:
        for l1_am in l1_alarm_mults:
            # Check: for all pumps, warn*mult < alarm*mult
            # Since we apply same mult ratio, check if warn_mult < alarm_mult
            # would guarantee it, but per-pump thresholds may differ
            # Just check: the max(warn*mult) must be < min(alarm*mult) per pump
            any_invalid = False
            for p in pumps:
                if baseline_per_pump["l1_warn_per_pump"][p] * l1_wm >= baseline_per_pump["l1_alarm_per_pump"][p] * l1_am:
                    any_invalid = True
                    break
            if any_invalid:
                n_pruned_order += len(l2_warn_mults) * len(l2_alarm_mults)
                continue

            for l2_wm in l2_warn_mults:
                for l2_am in l2_alarm_mults:
                    any_invalid_l2 = False
                    for p in pumps:
                        if baseline_per_pump["l2_warn_per_pump"][p] * l2_wm >= baseline_per_pump["l2_alarm_per_pump"][p] * l2_am:
                            any_invalid_l2 = True
                            break
                    if any_invalid_l2:
                        n_pruned_order += 1
                        continue

                    # Build per-pump thresholds
                    l1_wp = {p: baseline_per_pump["l1_warn_per_pump"][p] * l1_wm for p in pumps}
                    l1_ap = {p: baseline_per_pump["l1_alarm_per_pump"][p] * l1_am for p in pumps}
                    l2_wp = {p: baseline_per_pump["l2_warn_per_pump"][p] * l2_wm for p in pumps}
                    l2_ap = {p: baseline_per_pump["l2_alarm_per_pump"][p] * l2_am for p in pumps}

                    metrics = _evaluate_per_pump_thresholds_fast(
                        precomp, l1_wp, l1_ap, l2_wp, l2_ap, min_alert_steps
                    )
                    n_evaluated += 1

                    meets_constraints = all(
                        metrics["per_pump_detection"].get(p, 0.0) >= baseline_detection[p]
                        for p in baseline_detection
                    )

                    if not meets_constraints:
                        n_pruned_constraint += 1
                        continue

                    results.append({
                        "l1_warn_mult": l1_wm,
                        "l1_alarm_mult": l1_am,
                        "l2_warn_mult": l2_wm,
                        "l2_alarm_mult": l2_am,
                        "fa_rate": metrics["fa_rate"],
                        "n_false_alarms": metrics["n_false_alarms"],
                        "overall_detection": metrics["overall_detection"],
                        **{f"det_pump_{p}": v for p, v in metrics["per_pump_detection"].items()},
                    })

    elapsed = time.perf_counter() - t_start
    console.print(
        f"  Evaluated {n_evaluated:,} combos in {elapsed:.1f}s "
        f"(pruned {n_pruned_order:,} order violations, "
        f"{n_pruned_constraint:,} constraint violations)"
    )
    console.print(f"  [green]{len(results):,} feasible combos found.[/green]")

    if not results:
        console.print("[red]  No feasible threshold combos found! Constraints may be too tight.[/red]")
        return pd.DataFrame()

    results_df = pd.DataFrame(results).sort_values("fa_rate")
    return results_df


# ---------------------------------------------------------------------------
# Phase 4: Validation
# ---------------------------------------------------------------------------

def validate_on_holdout(
    holdout_set: pd.DataFrame,
    best: dict,
    min_alert_steps: int,
    baseline_metrics: dict,
) -> dict:
    """Evaluate the best multipliers on the held-out set."""
    console.print("[bold]Phase 4:[/bold] Validating on held-out set...")

    precomp = _precompute_day_structure(holdout_set)
    pumps = sorted(baseline_metrics["l1_warn_per_pump"].keys())

    # Apply best multipliers to per-pump thresholds
    l1_wp = {p: baseline_metrics["l1_warn_per_pump"][p] * best["l1_warn_mult"] for p in pumps}
    l1_ap = {p: baseline_metrics["l1_alarm_per_pump"][p] * best["l1_alarm_mult"] for p in pumps}
    l2_wp = {p: baseline_metrics["l2_warn_per_pump"][p] * best["l2_warn_mult"] for p in pumps}
    l2_ap = {p: baseline_metrics["l2_alarm_per_pump"][p] * best["l2_alarm_mult"] for p in pumps}

    metrics = _evaluate_per_pump_thresholds_fast(
        precomp, l1_wp, l1_ap, l2_wp, l2_ap, min_alert_steps
    )

    # Baseline on holdout (multiplier=1.0)
    holdout_baseline = _evaluate_per_pump_thresholds_fast(
        precomp,
        baseline_metrics["l1_warn_per_pump"],
        baseline_metrics["l1_alarm_per_pump"],
        baseline_metrics["l2_warn_per_pump"],
        baseline_metrics["l2_alarm_per_pump"],
        min_alert_steps,
    )

    fa_delta = holdout_baseline["fa_rate"] - metrics["fa_rate"]

    return {
        "holdout_fa_rate": metrics["fa_rate"],
        "holdout_baseline_fa_rate": holdout_baseline["fa_rate"],
        "holdout_fa_delta_pp": fa_delta * 100,
        "holdout_detection": metrics["overall_detection"],
        "holdout_per_pump_detection": metrics["per_pump_detection"],
        "holdout_baseline_per_pump_detection": holdout_baseline["per_pump_detection"],
        "baseline_on_holdout": holdout_baseline,
    }


# ---------------------------------------------------------------------------
# Phase 5: Promotion
# ---------------------------------------------------------------------------

def promote_thresholds(best: dict, baseline: dict) -> None:
    """Update production threshold JSONs with optimised multipliers."""
    l1_wm = best["l1_warn_mult"]
    l1_am = best["l1_alarm_mult"]
    l2_wm = best["l2_warn_mult"]
    l2_am = best["l2_alarm_mult"]

    # L1: update global + per-pump mahalanobis warning/alarm
    with open(L1_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        l1_data = json.load(f)

    old_l1_gw = l1_data["global"]["mahalanobis"]["warning"]
    old_l1_ga = l1_data["global"]["mahalanobis"]["alarm"]
    l1_data["global"]["mahalanobis"]["warning"] = old_l1_gw * l1_wm
    l1_data["global"]["mahalanobis"]["alarm"] = old_l1_ga * l1_am

    for pid_str, block in l1_data.get("per_pump", {}).items():
        maha = block.get("mahalanobis", {})
        if "warning" in maha:
            maha["warning"] = maha["warning"] * l1_wm
        if "alarm" in maha:
            maha["alarm"] = maha["alarm"] * l1_am

    with open(L1_THRESHOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(l1_data, f, indent=4)

    # L2: update global + per-pump warning/alarm
    with open(L2_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        l2_data = json.load(f)

    old_l2_gw = l2_data["global"]["warning"]
    old_l2_ga = l2_data["global"]["alarm"]
    l2_data["global"]["warning"] = old_l2_gw * l2_wm
    l2_data["global"]["alarm"] = old_l2_ga * l2_am

    for pid_str, block in l2_data.get("per_pump", {}).items():
        if "warning" in block:
            block["warning"] = block["warning"] * l2_wm
        if "alarm" in block:
            block["alarm"] = block["alarm"] * l2_am

    with open(L2_THRESHOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(l2_data, f, indent=4)

    console.print("[green bold]  Thresholds promoted![/green bold]")
    console.print(f"    L1 warn multiplier: ×{l1_wm:.3f}")
    console.print(f"    L1 alarm multiplier: ×{l1_am:.3f}")
    console.print(f"    L2 warn multiplier: ×{l2_wm:.3f}")
    console.print(f"    L2 alarm multiplier: ×{l2_am:.3f}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(
    baseline_metrics: dict,
    opt_metrics: dict,
    holdout_results: dict,
    best: dict,
) -> None:
    """Print a rich comparison table."""
    table = Table(title="P8 Joint Threshold Optimisation Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Optimised (opt-set)", justify="right")
    table.add_column("Optimised (holdout)", justify="right")
    table.add_column("Δ (holdout)", justify="right")

    # FA rate
    base_fa = baseline_metrics["full_fa_rate"]
    opt_fa = opt_metrics["fa_rate"]
    hold_fa = holdout_results["holdout_fa_rate"]
    delta_fa = holdout_results["holdout_fa_delta_pp"]
    delta_style = "green" if delta_fa >= 5.0 else "yellow" if delta_fa > 0 else "red"

    table.add_row(
        "False Alarm Rate",
        f"{base_fa*100:.1f}%",
        f"{opt_fa*100:.1f}%",
        f"{hold_fa*100:.1f}%",
        f"[{delta_style}]-{delta_fa:.1f}pp[/{delta_style}]",
    )

    # Per-pump detection
    for pump in sorted(baseline_metrics["per_pump_detection"].keys()):
        base_det = baseline_metrics["per_pump_detection"][pump]
        hold_det = holdout_results["holdout_per_pump_detection"].get(pump, 0.0)
        opt_det = opt_metrics.get(f"det_pump_{pump}", opt_metrics.get("per_pump_detection", {}).get(pump, 0.0))
        delta_det = (hold_det - base_det) * 100
        det_style = "green" if delta_det >= 0 else "red"

        table.add_row(
            f"Detection Pump {pump}",
            f"{base_det*100:.1f}%",
            f"{opt_det*100:.1f}%",
            f"{hold_det*100:.1f}%",
            f"[{det_style}]{delta_det:+.1f}pp[/{det_style}]",
        )

    console.print(table)
    console.print()

    # Best thresholds
    thresh_table = Table(title="Optimised Thresholds")
    thresh_table.add_column("Threshold")
    thresh_table.add_column("Baseline", justify="right")
    thresh_table.add_column("Optimised", justify="right")

    thresh_table.add_row("L1 Warning", "×1.000", f"×{best['l1_warn_mult']:.3f}")
    thresh_table.add_row("L1 Alarm", "×1.000", f"×{best['l1_alarm_mult']:.3f}")
    thresh_table.add_row("L2 Warning", "×1.000", f"×{best['l2_warn_mult']:.3f}")
    thresh_table.add_row("L2 Alarm", "×1.000", f"×{best['l2_alarm_mult']:.3f}")

    console.print(thresh_table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_baseline_thresholds() -> dict:
    """Load current production thresholds (global + per-pump) as baseline."""
    with open(L1_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        l1_data = json.load(f)
    with open(L2_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        l2_data = json.load(f)

    # Global thresholds
    baseline = {
        "l1_warn": l1_data["global"]["mahalanobis"]["warning"],
        "l1_alarm": l1_data["global"]["mahalanobis"]["alarm"],
        "l2_warn": l2_data["global"]["warning"],
        "l2_alarm": l2_data["global"]["alarm"],
    }

    # Per-pump thresholds
    l1_warn_per_pump = {}
    l1_alarm_per_pump = {}
    for pid_str, block in l1_data.get("per_pump", {}).items():
        pid = int(pid_str)
        maha = block.get("mahalanobis", {})
        # Use per-pump if available, fallback to global
        l1_warn_per_pump[pid] = maha.get("warning", baseline["l1_warn"])
        l1_alarm_per_pump[pid] = maha.get("alarm", baseline["l1_alarm"])

    l2_warn_per_pump = {}
    l2_alarm_per_pump = {}
    for pid_str, block in l2_data.get("per_pump", {}).items():
        pid = int(pid_str)
        l2_warn_per_pump[pid] = block.get("warning", baseline["l2_warn"])
        l2_alarm_per_pump[pid] = block.get("alarm", baseline["l2_alarm"])

    baseline["l1_warn_per_pump"] = l1_warn_per_pump
    baseline["l1_alarm_per_pump"] = l1_alarm_per_pump
    baseline["l2_warn_per_pump"] = l2_warn_per_pump
    baseline["l2_alarm_per_pump"] = l2_alarm_per_pump

    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="P8: Joint L1+L2 Threshold Optimisation")
    parser.add_argument("--skip-extraction", action="store_true", help="Use cached scores")
    parser.add_argument("--force-extraction", action="store_true", help="Force re-extraction")
    parser.add_argument("--promote", action="store_true", help="Auto-promote if criteria met")
    parser.add_argument("--grid-resolution", type=int, default=20, help="Points per dimension")
    parser.add_argument("--opt-fraction", type=float, default=0.7, help="Optimisation set fraction")
    parser.add_argument("--min-alert-steps", type=int, default=3, help="K for day-level alarm")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Data directory")
    args = parser.parse_args()

    console.rule("[bold blue]P8 — Joint L1+L2 Threshold Optimisation[/bold blue]")
    data_dir = Path(args.data_dir)

    # --- Phase 1: Score Extraction ---
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = SCORES_DIR / "scores.parquet"

    if args.skip_extraction:
        if not scores_path.exists():
            console.print("[red]ERROR: --skip-extraction but scores.parquet not found![/red]")
            sys.exit(1)
        console.print(f"[dim]Skipping extraction, loading {scores_path}[/dim]")
        scores_df = pd.read_parquet(scores_path)
    elif scores_path.exists() and not args.force_extraction:
        console.print(f"[dim]Cache hit: loading {scores_path}[/dim]")
        scores_df = pd.read_parquet(scores_path)
    else:
        scores_df = extract_scores(data_dir)
        scores_df.to_parquet(scores_path, index=False)
        console.print(f"  Saved to {scores_path}")

    console.print(
        f"  Scores: {len(scores_df):,} timesteps, "
        f"{scores_df['csv_name'].nunique()} days, "
        f"{scores_df['pump_id'].nunique()} pumps"
    )

    # --- Phase 2: Split ---
    console.print()
    console.print("[bold]Phase 2:[/bold] Splitting into opt/holdout...")
    opt_set, holdout_set = split_days(scores_df, opt_fraction=args.opt_fraction)

    # --- Baseline metrics (using per-pump thresholds) ---
    baseline_thresholds = _load_baseline_thresholds()
    console.print()
    console.print("[bold]Baseline evaluation[/bold] (current per-pump production thresholds)...")

    precomp_full = _precompute_day_structure(scores_df)
    full_baseline = _evaluate_per_pump_thresholds_fast(
        precomp_full,
        baseline_thresholds["l1_warn_per_pump"],
        baseline_thresholds["l1_alarm_per_pump"],
        baseline_thresholds["l2_warn_per_pump"],
        baseline_thresholds["l2_alarm_per_pump"],
        args.min_alert_steps,
    )
    baseline_thresholds["full_fa_rate"] = full_baseline["fa_rate"]
    baseline_thresholds["per_pump_detection"] = full_baseline["per_pump_detection"]
    baseline_thresholds["overall_detection"] = full_baseline["overall_detection"]

    console.print(f"  FA rate: {full_baseline['fa_rate']*100:.1f}% ({full_baseline['n_false_alarms']}/{full_baseline['n_normal_days']})")
    for p, d in sorted(full_baseline["per_pump_detection"].items()):
        console.print(f"  Pump {p} detection: {d*100:.1f}%")

    # --- Phase 3: Grid Search ---
    console.print()
    results_df = grid_search(
        opt_set,
        resolution=args.grid_resolution,
        min_alert_steps=args.min_alert_steps,
        baseline_detection=full_baseline["per_pump_detection"],
        baseline_per_pump=baseline_thresholds,
    )

    if results_df.empty:
        console.print("[red]Optimisation failed — no feasible combos. Try relaxing constraints.[/red]")
        sys.exit(1)

    # Save grid results
    grid_path = SCORES_DIR / "grid_results.parquet"
    results_df.to_parquet(grid_path, index=False)

    best = results_df.iloc[0].to_dict()
    console.print(f"\n  [green bold]Best combo (multipliers):[/green bold]")
    console.print(f"    L1 warn ×{best['l1_warn_mult']:.3f}, alarm ×{best['l1_alarm_mult']:.3f}")
    console.print(f"    L2 warn ×{best['l2_warn_mult']:.3f}, alarm ×{best['l2_alarm_mult']:.3f}")
    console.print(f"    FA rate={best['fa_rate']*100:.1f}% (vs baseline {full_baseline['fa_rate']*100:.1f}%)")

    # --- Phase 4: Holdout Validation ---
    console.print()
    holdout_results = validate_on_holdout(
        holdout_set, best, args.min_alert_steps, baseline_thresholds
    )

    # Overfit check
    opt_delta = (full_baseline["fa_rate"] - best["fa_rate"]) * 100
    holdout_delta = holdout_results["holdout_fa_delta_pp"]
    overfit_gap = opt_delta - holdout_delta

    console.print(f"  Holdout FA rate: {holdout_results['holdout_fa_rate']*100:.1f}%")
    console.print(f"  Holdout baseline FA rate: {holdout_results['holdout_baseline_fa_rate']*100:.1f}%")
    console.print(f"  Holdout improvement: {holdout_delta:.1f}pp")
    console.print(f"  Opt-to-holdout gap: {overfit_gap:.1f}pp", style="yellow" if overfit_gap > 3.0 else "green")

    # --- Results ---
    console.print()
    print_results_table(baseline_thresholds, best, holdout_results, best)

    # --- Acceptance check ---
    acceptance_met = (
        holdout_delta >= 5.0
        and all(
            holdout_results["holdout_per_pump_detection"].get(p, 0.0) >= baseline_thresholds["per_pump_detection"][p]
            for p in baseline_thresholds["per_pump_detection"]
        )
        and overfit_gap < 3.0
    )

    console.print()
    if acceptance_met:
        console.print("[green bold]✓ Acceptance criteria MET on held-out set.[/green bold]")
    else:
        reasons = []
        if holdout_delta < 5.0:
            reasons.append(f"FA reduction only {holdout_delta:.1f}pp (need ≥5pp)")
        for p in baseline_thresholds["per_pump_detection"]:
            hold_det = holdout_results["holdout_per_pump_detection"].get(p, 0.0)
            base_det = baseline_thresholds["per_pump_detection"][p]
            if hold_det < base_det:
                reasons.append(f"Pump {p} detection dropped: {hold_det*100:.1f}% < {base_det*100:.1f}%")
        if overfit_gap >= 3.0:
            reasons.append(f"Overfit gap {overfit_gap:.1f}pp ≥ 3pp")
        console.print("[yellow bold]✗ Acceptance criteria NOT met.[/yellow bold]")
        for r in reasons:
            console.print(f"  - {r}")

    # --- Save report ---
    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "baseline_thresholds_global": {k: v for k, v in baseline_thresholds.items() if k in ("l1_warn", "l1_alarm", "l2_warn", "l2_alarm")},
        "optimised_multipliers": {k: best[k] for k in ("l1_warn_mult", "l1_alarm_mult", "l2_warn_mult", "l2_alarm_mult")},
        "grid_resolution": args.grid_resolution,
        "opt_fraction": args.opt_fraction,
        "min_alert_steps": args.min_alert_steps,
        "baseline_fa_rate": full_baseline["fa_rate"],
        "baseline_per_pump_detection": {str(k): v for k, v in full_baseline["per_pump_detection"].items()},
        "opt_set_fa_rate": best["fa_rate"],
        "holdout_fa_rate": holdout_results["holdout_fa_rate"],
        "holdout_fa_delta_pp": holdout_delta,
        "holdout_per_pump_detection": {str(k): v for k, v in holdout_results["holdout_per_pump_detection"].items()},
        "overfit_gap_pp": overfit_gap,
        "acceptance_met": acceptance_met,
        "n_feasible_combos": len(results_df),
        "top_10_combos": results_df.head(10).to_dict(orient="records"),
    }

    report_path = SCRIPT_DIR / "P8_joint_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    console.print(f"\n  Report saved to {report_path}")

    # --- Phase 5: Promotion ---
    if acceptance_met and args.promote:
        console.print()
        promote_thresholds(best, baseline_thresholds)
    elif acceptance_met and not args.promote:
        console.print("\n[dim]  Pass --promote to auto-update production thresholds.[/dim]")


if __name__ == "__main__":
    main()
