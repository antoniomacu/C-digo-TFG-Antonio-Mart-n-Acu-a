"""Batch evaluation of Old Model (v1) vs Ensemble (v2) across the full dataset.

Calibrates v2 thresholds on all normal training data (110 CSVs), then evaluates
both models on 210 CSVs (110 normal + 100 abnormal). Writes a JSON results file
consumed by report_generator.py.

Usage:
    cd ensemble
    uv run python demos/batch_evaluate.py [--data-dir ../../data] [--output-dir demos/batch_results]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

SCRIPT_DIR = Path(__file__).resolve().parent
ENSEMBLE_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = ENSEMBLE_DIR.parent

sys.path.insert(0, str(ENSEMBLE_DIR))
sys.path.insert(0, str(WORKSPACE_ROOT))

from model.streaming import create_streaming_detector
from demos.cond_reg_v1_bridge import CondRegV1Bridge
from demos.streaming_demo import _load_day, _parse_pump_from_filename

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

HARDCODED_DATA_FALLBACK = Path("<PATH_TO_DATA_DIR>")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DayResult:
    csv_name: str
    pump_id: int
    label: str                       # "normal" | "abnormal"
    n_timesteps: int
    # V1 (old deployed model)
    v1_n_alarm_steps: int
    v1_n_warning_steps: int
    v1_had_false_alarm: bool         # label=="normal" and v1_n_alarm_steps > 0
    v1_detected_failure: bool        # label=="abnormal" and v1_n_alarm_steps > 0
    v1_first_alarm_timestep: int | None
    v1_avg_health: float
    v1_health_timeline: list[float]  # full timeline for showcase day, else []
    # V2 (ensemble)
    v2_n_alarm_steps: int
    v2_n_warning_steps: int
    v2_had_false_alarm: bool
    v2_detected_failure: bool
    v2_first_alarm_timestep: int | None
    v2_avg_health: float
    v2_health_timeline: list[float]
    # Metadata
    is_showcase_day: bool


@dataclass
class BatchResults:
    generated_at: str
    data_dir: str
    calibration_n_samples_l1: int
    calibration_n_samples_l2: int
    calibration_n_normal_files: int
    showcase_pump_id: int
    showcase_csv_name: str
    min_alert_steps: int
    days: list[DayResult]


# ---------------------------------------------------------------------------
# Data directory resolution
# ---------------------------------------------------------------------------

def _resolve_data_dir(data_dir_arg: str) -> Path:
    p = Path(data_dir_arg).expanduser()
    if p.is_absolute() and p.exists():
        return p.resolve()
    alt = (ENSEMBLE_DIR / data_dir_arg).resolve()
    if alt.exists():
        return alt
    alt2 = (WORKSPACE_ROOT / data_dir_arg).resolve()
    if alt2.exists():
        return alt2
    if HARDCODED_DATA_FALLBACK.exists():
        console.print(
            f"  [yellow]Warning:[/yellow] Could not resolve '{data_dir_arg}', "
            f"falling back to {HARDCODED_DATA_FALLBACK}"
        )
        return HARDCODED_DATA_FALLBACK
    raise FileNotFoundError(
        f"Data directory not found: '{data_dir_arg}'. "
        f"Set --data-dir to the absolute path of your data/ folder."
    )


# ---------------------------------------------------------------------------
# Calibration (adapted from calibrate_thresholds.py)
# ---------------------------------------------------------------------------

def _collect_calibration_scores(
    normal_dir: Path,
    detector,
) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    """Run all normal CSVs through detector, collect L1/L2 scores per pump."""
    csv_files = sorted(normal_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {normal_dir}")

    l1_scores: dict[int, list[float]] = {}
    l2_scores: dict[int, list[float]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Calibrating thresholds...", total=len(csv_files))

        for csv_path in csv_files:
            pump_id = _parse_pump_from_filename(csv_path)
            if pump_id is None:
                logger.warning("Cannot parse pump ID from %s, skipping", csv_path.name)
                progress.advance(task)
                continue

            try:
                df = _load_day(csv_path)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", csv_path.name, exc)
                progress.advance(task)
                continue

            if len(df) < 5:
                logger.warning("Skipping %s (<5 rows after preprocessing)", csv_path.name)
                progress.advance(task)
                continue

            detector.reset_pump(pump_id)
            l1_scores.setdefault(pump_id, [])
            l2_scores.setdefault(pump_id, [])

            for ts, row in df.iterrows():
                try:
                    result = detector.process_timestep(pump_id, ts, row)
                    l1_scores[pump_id].append(float(result.l1_mahalanobis))
                    if result.l2_smoothed_mse is not None:
                        l2_scores[pump_id].append(float(result.l2_smoothed_mse))
                except Exception as exc:
                    logger.debug("Error at %s timestep: %s", csv_path.name, exc)

            progress.advance(task)

    return l1_scores, l2_scores


def _compute_thresholds(
    scores_by_pump: dict[int, list[float]],
    p_warning: float,
    p_alarm: float,
    min_samples: int = 100,
) -> tuple[dict, dict[int, dict]]:
    """Compute global and per-pump threshold statistics from score distributions."""
    all_scores: list[float] = []
    for pump_scores in scores_by_pump.values():
        all_scores.extend(pump_scores)

    if not all_scores:
        raise ValueError("No scores collected — cannot compute thresholds.")

    arr = np.asarray(all_scores, dtype=float)
    global_block: dict[str, float] = {
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
            per_pump[pump_id] = {"fallback_to_global": True, "n_samples": len(pump_scores)}
            console.print(
                f"  Pump {pump_id}: only {len(pump_scores)} samples — falling back to global thresholds"
            )
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


def _build_l1_calibrated(global_block: dict, per_pump: dict[int, dict], n_samples: int) -> dict:
    from datetime import date
    return {
        "description": "Calibrated L1 thresholds from full normal training data distribution",
        "calibration_date": date.today().isoformat(),
        "n_calibration_samples": n_samples,
        "global": {"mahalanobis": global_block},
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


def _build_l2_calibrated(global_block: dict, per_pump: dict[int, dict], n_samples: int) -> dict:
    from datetime import date
    return {
        "description": "Calibrated L2 thresholds from full normal training data distribution",
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


def _calibrate_full(
    data_dir: Path,
    output_dir: Path,
    p_warning: float = 95.0,
    p_alarm: float = 99.0,
) -> tuple[dict, dict]:
    """Run calibration on all normal training CSVs. Returns (l1_thresholds, l2_thresholds)."""
    normal_dir = data_dir / "train"
    if not normal_dir.exists():
        raise FileNotFoundError(f"Normal training directory not found: {normal_dir}")

    n_files = len(list(normal_dir.glob("*.csv")))
    console.print(
        f"\n[bold]Phase 1: Calibrating thresholds[/bold] on {n_files} normal training files..."
    )

    detector_raw = create_streaming_detector()

    l1_scores, l2_scores = _collect_calibration_scores(normal_dir, detector_raw)

    total_l1 = sum(len(v) for v in l1_scores.values())
    total_l2 = sum(len(v) for v in l2_scores.values())
    console.print(
        f"  Collected {total_l1} L1 scores, {total_l2} L2 scores "
        f"from {n_files} normal files."
    )

    l1_global, l1_per_pump = _compute_thresholds(l1_scores, p_warning, p_alarm)
    l1_output = _build_l1_calibrated(l1_global, l1_per_pump, total_l1)

    l2_global, l2_per_pump = _compute_thresholds(l2_scores, p_warning, p_alarm)
    l2_output = _build_l2_calibrated(l2_global, l2_per_pump, total_l2)

    cal_dir = output_dir / "calibrated_thresholds_full"
    cal_dir.mkdir(parents=True, exist_ok=True)
    with open(cal_dir / "l1_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(l1_output, f, indent=2)
    with open(cal_dir / "l2_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(l2_output, f, indent=2)

    console.print(
        f"  [green]Calibration complete.[/green] "
        f"L1 warning={l1_global['warning']:.3f}, alarm={l1_global['alarm']:.3f} | "
        f"L2 warning={l2_global['warning']:.4f}, alarm={l2_global['alarm']:.4f}"
    )
    console.print(f"  Thresholds saved to {cal_dir}/")

    return l1_output, l2_output, total_l1, total_l2, n_files


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_day(
    csv_path: Path,
    pump_id: int,
    label: str,
    detector,
    v1_bridge: CondRegV1Bridge,
    is_showcase: bool,
    min_alert_steps: int = 3,
) -> DayResult | None:
    """Evaluate both models on a single pump-day CSV. Returns None if skipped."""
    try:
        df = _load_day(csv_path)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", csv_path.name, exc)
        return None

    if len(df) < 5:
        logger.warning("Skipping %s (<5 rows after preprocessing)", csv_path.name)
        return None

    detector.reset_pump(pump_id)
    v1_bridge.reset_pump(pump_id)

    v1_alarm_steps = 0
    v1_warning_steps = 0
    v1_first_alarm: int | None = None
    v1_health_sum = 0.0
    v1_health_timeline: list[float] = []

    v2_alarm_steps = 0
    v2_warning_steps = 0
    v2_first_alarm: int | None = None
    v2_health_sum = 0.0
    v2_health_timeline: list[float] = []

    n_timesteps = 0

    for idx, (ts, row) in enumerate(df.iterrows()):
        n_timesteps += 1

        try:
            v2_result = detector.process_timestep(pump_id, ts, row)
            v2_health = float(v2_result.ensemble_health)
            v2_health_sum += v2_health
            if is_showcase:
                v2_health_timeline.append(v2_health)

            if v2_result.ensemble_status == "ALARM":
                v2_alarm_steps += 1
                if v2_first_alarm is None:
                    v2_first_alarm = idx
            elif v2_result.ensemble_status == "WARNING":
                v2_warning_steps += 1
        except Exception as exc:
            logger.debug("V2 error at %s idx %d: %s", csv_path.name, idx, exc)

        try:
            v1_result = v1_bridge.process_timestep(pump_id, ts, row)
            v1_health = float(v1_result.health)
            v1_health_sum += v1_health
            if is_showcase:
                v1_health_timeline.append(v1_health)

            if v1_result.status == "ALARM":
                v1_alarm_steps += 1
                if v1_first_alarm is None:
                    v1_first_alarm = idx
            elif v1_result.status == "WARNING":
                v1_warning_steps += 1
        except Exception as exc:
            logger.debug("V1 error at %s idx %d: %s", csv_path.name, idx, exc)

    if n_timesteps == 0:
        return None

    v1_avg = v1_health_sum / n_timesteps if n_timesteps > 0 else float("nan")
    v2_avg = v2_health_sum / n_timesteps if n_timesteps > 0 else float("nan")

    return DayResult(
        csv_name=csv_path.name,
        pump_id=pump_id,
        label=label,
        n_timesteps=n_timesteps,
        v1_n_alarm_steps=v1_alarm_steps,
        v1_n_warning_steps=v1_warning_steps,
        v1_had_false_alarm=(label == "normal" and v1_alarm_steps > 0),
        v1_detected_failure=(label == "abnormal" and v1_alarm_steps > 0),
        v1_first_alarm_timestep=v1_first_alarm,
        v1_avg_health=v1_avg,
        v1_health_timeline=v1_health_timeline,
        v2_n_alarm_steps=v2_alarm_steps,
        v2_n_warning_steps=v2_warning_steps,
        v2_had_false_alarm=(label == "normal" and (v2_alarm_steps + v2_warning_steps) >= min_alert_steps),
        v2_detected_failure=(label == "abnormal" and (v2_alarm_steps + v2_warning_steps) >= min_alert_steps),
        v2_first_alarm_timestep=v2_first_alarm,
        v2_avg_health=v2_avg,
        v2_health_timeline=v2_health_timeline,
        is_showcase_day=is_showcase,
    )


def _evaluate_all(
    data_dir: Path,
    detector,
    v1_bridge: CondRegV1Bridge,
    showcase_csv: Path | None,
    min_alert_steps: int = 3,
) -> list[DayResult]:
    """Evaluate both models on all 210 CSVs (110 normal + 100 abnormal)."""
    normal_files = sorted((data_dir / "train").glob("*.csv"))
    abnormal_files = sorted((data_dir / "test").glob("*.csv"))

    all_files: list[tuple[Path, str]] = (
        [(f, "normal") for f in normal_files] +
        [(f, "abnormal") for f in abnormal_files]
    )

    total = len(all_files)
    console.print(
        f"\n[bold]Phase 2: Evaluating both models[/bold] on {total} files "
        f"({len(normal_files)} normal + {len(abnormal_files)} abnormal)..."
    )

    results: list[DayResult] = []
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=total)

        for csv_path, label in all_files:
            pump_id = _parse_pump_from_filename(csv_path)
            if pump_id is None:
                logger.warning("Cannot parse pump ID from %s, skipping", csv_path.name)
                skipped += 1
                progress.advance(task)
                continue

            is_showcase = showcase_csv is not None and csv_path == showcase_csv
            result = _evaluate_day(csv_path, pump_id, label, detector, v1_bridge, is_showcase, min_alert_steps)

            if result is not None:
                results.append(result)
            else:
                skipped += 1

            progress.advance(task)

    console.print(
        f"  Evaluated {len(results)} days, skipped {skipped}."
    )
    return results


# ---------------------------------------------------------------------------
# Summary and output
# ---------------------------------------------------------------------------

def _print_summary_table(results: list[DayResult]) -> None:
    table = Table(title="Batch Evaluation Summary", show_lines=True)
    table.add_column("Pump", justify="center")
    table.add_column("Normal", justify="right")
    table.add_column("Abnormal", justify="right")
    table.add_column("v1 False Alarm Days", justify="right", style="red")
    table.add_column("v2 False Alarm Days", justify="right", style="green")
    table.add_column("v1 Detected", justify="right", style="red")
    table.add_column("v2 Detected", justify="right", style="green")

    by_pump: dict[int, list[DayResult]] = {}
    for r in results:
        by_pump.setdefault(r.pump_id, []).append(r)

    for pump_id in sorted(by_pump.keys()):
        days = by_pump[pump_id]
        normal = [d for d in days if d.label == "normal"]
        abnormal = [d for d in days if d.label == "abnormal"]

        v1_fa = sum(1 for d in normal if d.v1_had_false_alarm)
        v2_fa = sum(1 for d in normal if d.v2_had_false_alarm)
        v1_det = sum(1 for d in abnormal if d.v1_detected_failure)
        v2_det = sum(1 for d in abnormal if d.v2_detected_failure)

        table.add_row(
            str(pump_id),
            str(len(normal)),
            str(len(abnormal)),
            f"{v1_fa}/{len(normal)} ({100*v1_fa/max(len(normal),1):.0f}%)",
            f"{v2_fa}/{len(normal)} ({100*v2_fa/max(len(normal),1):.0f}%)",
            f"{v1_det}/{len(abnormal)} ({100*v1_det/max(len(abnormal),1):.0f}%)",
            f"{v2_det}/{len(abnormal)} ({100*v2_det/max(len(abnormal),1):.0f}%)",
        )

    all_normal = [d for d in results if d.label == "normal"]
    all_abnormal = [d for d in results if d.label == "abnormal"]
    v1_fa_all = sum(1 for d in all_normal if d.v1_had_false_alarm)
    v2_fa_all = sum(1 for d in all_normal if d.v2_had_false_alarm)
    v1_det_all = sum(1 for d in all_abnormal if d.v1_detected_failure)
    v2_det_all = sum(1 for d in all_abnormal if d.v2_detected_failure)
    table.add_row(
        "[bold]ALL[/bold]",
        str(len(all_normal)),
        str(len(all_abnormal)),
        f"{v1_fa_all}/{len(all_normal)} ({100*v1_fa_all/max(len(all_normal),1):.0f}%)",
        f"{v2_fa_all}/{len(all_normal)} ({100*v2_fa_all/max(len(all_normal),1):.0f}%)",
        f"{v1_det_all}/{len(all_abnormal)} ({100*v1_det_all/max(len(all_abnormal),1):.0f}%)",
        f"{v2_det_all}/{len(all_abnormal)} ({100*v2_det_all/max(len(all_abnormal),1):.0f}%)",
    )

    console.print(table)


def _save_results(results: BatchResults, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    size_kb = output_path.stat().st_size / 1024
    console.print(f"\n[green]Results saved:[/green] {output_path} ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-dataset batch evaluation: Old Model (v1) vs Ensemble (v2)"
    )
    parser.add_argument(
        "--data-dir",
        default="../../data",
        help="Path to data directory (must contain train/ and test/ subdirs). "
             "Default: ../../data (relative to ensemble/ working dir)",
    )
    parser.add_argument(
        "--output-dir",
        default="demos/batch_results",
        help="Directory to write results JSON and calibrated thresholds. "
             "Default: demos/batch_results",
    )
    parser.add_argument(
        "--percentiles",
        default="95,99",
        help="Warning,alarm percentiles for threshold calibration. Default: 95,99",
    )
    parser.add_argument(
        "--min-alert-steps",
        type=int,
        default=3,
        help="Minimum combined WARNING+ALARM steps required to flag a day (K parameter). Default: 3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = _resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (ENSEMBLE_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    p_warning, p_alarm = (float(x) for x in args.percentiles.split(","))

    min_alert_steps = args.min_alert_steps

    console.print(f"\n[bold white]Batch Evaluation — Full Dataset Comparison[/bold white]")
    console.print(f"  Data dir       : {data_dir}")
    console.print(f"  Output dir     : {output_dir}")
    console.print(f"  Percentiles    : P{p_warning:.0f} (warning), P{p_alarm:.0f} (alarm)")
    console.print(f"  Min alert steps: K={min_alert_steps} (WARNING+ALARM combined)")

    t_start = time.perf_counter()

    # Phase 1: Calibrate thresholds on all normal training data
    l1_thresholds, l2_thresholds, n_l1, n_l2, n_files = _calibrate_full(
        data_dir, output_dir, p_warning, p_alarm
    )

    # Recreate detector with calibrated thresholds
    console.print("\n[dim]Loading models with calibrated thresholds...[/dim]")
    detector = create_streaming_detector(
        l1_thresholds=l1_thresholds,
        l2_thresholds=l2_thresholds,
    )
    v1_bridge = CondRegV1Bridge()
    console.print("  [green]Models ready.[/green]")

    # Select showcase day: first pump-3 abnormal CSV alphabetically
    abnormal_dir = data_dir / "test"
    showcase_csv: Path | None = None
    showcase_pump_id = 3
    showcase_csv_name = ""
    if abnormal_dir.exists():
        pump3_files = sorted(
            f for f in abnormal_dir.glob("*.csv")
            if _parse_pump_from_filename(f) == showcase_pump_id
        )
        if pump3_files:
            showcase_csv = pump3_files[0]
            showcase_csv_name = showcase_csv.name
            console.print(f"  Showcase day: {showcase_csv_name}")
        else:
            console.print("  [yellow]No pump-3 abnormal files found for showcase.[/yellow]")

    # Phase 2: Evaluate both models on all 210 files
    results = _evaluate_all(data_dir, detector, v1_bridge, showcase_csv, min_alert_steps)

    # Print summary table
    console.print()
    _print_summary_table(results)

    # Save results
    batch = BatchResults(
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        data_dir=str(data_dir),
        calibration_n_samples_l1=n_l1,
        calibration_n_samples_l2=n_l2,
        calibration_n_normal_files=n_files,
        showcase_pump_id=showcase_pump_id,
        showcase_csv_name=showcase_csv_name,
        min_alert_steps=min_alert_steps,
        days=results,
    )

    output_path = output_dir / "batch_results.json"
    _save_results(batch, output_path)

    elapsed = time.perf_counter() - t_start
    console.print(
        f"\n[bold]Done[/bold] in {elapsed:.1f}s. "
        f"Run report_generator.py to generate the HTML report."
    )
    console.print(
        f"  uv run python demos/report_generator.py "
        f"--results-file {output_path} "
        f"--output-file {output_dir}/comparison_report.html"
    )


if __name__ == "__main__":
    main()
