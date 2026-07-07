from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .system_config import SystemConfig, load_config
from .classifier import load_custom_detectors, classify_period
from .baseline import PumpBaseline, compute_baselines
from .loader import discover_files, load_csv
from .period_detector import detect_periods

OUTPUT_COLUMNS = [
    "date",
    "pump",
    "period_start",
    "period_end",
    "duration_minutes",
    "classification",
    "reasons",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-pass pump operation annotation pipeline."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to system YAML config file")
    parser.add_argument("--data-root", type=Path, default=None, help="Override data_root from config")
    parser.add_argument("--baseline-root", type=Path, default=None,
                        help="Directory of known-normal files for baseline computation (overrides data_root for baselines)")
    parser.add_argument("--output", type=Path, default=None, help="Override output path from config")
    parser.add_argument("--sigma", type=float, default=None, help="Override anomaly_sigma threshold")
    parser.add_argument("--no-html", action="store_true",
                        help="Skip HTML report generation")
    parser.add_argument("--no-data-selector", action="store_true",
                        help="Skip data selection and train/test split")
    parser.add_argument(
        "--no-ml-ensemble",
        action="store_true",
        help="Disable the ML ensemble detector (Stage 1 / rule-based-only run).",
    )
    return parser.parse_args()


def _apply_overrides(
    cfg: SystemConfig,
    data_root: Path | None,
    baseline_root: Path | None,
    output: Path | None,
    sigma: float | None,
) -> tuple[SystemConfig, Path, Path | None]:
    updated_cfg = cfg

    if data_root is not None:
        updated_cfg = replace(updated_cfg, data_root=data_root)

    if sigma is not None:
        if sigma <= 0:
            raise ValueError("--sigma must be > 0")
        updated_thresholds = replace(updated_cfg.thresholds, anomaly_sigma=float(sigma))
        updated_cfg = replace(updated_cfg, thresholds=updated_thresholds)

    output_path = output if output is not None else updated_cfg.output_dir / f"{updated_cfg.system_name}_annotations.csv"
    return updated_cfg, output_path, baseline_root


def _safe_sampling_interval(df: pd.DataFrame) -> float:
    interval = float(df.attrs.get("sampling_interval_sec", 5.0))
    if not np.isfinite(interval) or interval <= 0:
        return 5.0
    return interval


def _normalize_reasons(values: Iterable[object] | None) -> str:
    if values is None:
        return ""

    cleaned: list[str] = []
    for value in values:
        if value is None:
            continue

        if isinstance(value, float) and np.isnan(value):
            continue

        if pd.isna(value):
            continue

        text = str(value).strip()
        if text:
            cleaned.append(text)

    return ";".join(cleaned)


def _print_discovery_summary(file_list: list[tuple[object, int, Path]], cfg: SystemConfig) -> None:
    print("\n=== Discovery Summary ===")
    total = len(file_list)
    print(f"Total files found: {total}")

    if not file_list:
        print("Date range: n/a")
    else:
        start_date = file_list[0][0]
        end_date = file_list[-1][0]
        print(f"Date range: {start_date} to {end_date}")

    per_pump = {pump_id: 0 for pump_id in cfg.pump_ids}
    for _, pump_id, _ in file_list:
        per_pump[pump_id] = per_pump.get(pump_id, 0) + 1

    for pump_id in sorted(per_pump):
        print(f"Pump {pump_id}: {per_pump[pump_id]} files")


def _format_temp_summary(baseline: PumpBaseline, cfg: SystemConfig) -> str:
    selected = [col for col in cfg.columns.temp_cols if col in baseline.temp_medians]
    if not selected:
        return "temps=n/a"

    parts: list[str] = []
    for col in selected:
        median = baseline.temp_medians.get(col, float("nan"))
        std = baseline.temp_stds.get(col, float("nan"))
        parts.append(f"{col}={median:.2f}+/-{std:.2f}")

    return ", ".join(parts)


def _print_baseline_summary(baselines: dict[int, PumpBaseline], cfg: SystemConfig) -> None:
    print("\n=== Pass 1: Baselines ===")
    if not baselines:
        print("No baselines computed.")
        return

    for pump_id in sorted(cfg.pump_ids):
        baseline = baselines.get(pump_id)
        if baseline is None:
            print(f"Pump {pump_id}: baseline unavailable")
            continue

        print(
            "Pump "
            f"{pump_id}: "
            f"speed={baseline.speed_median:.2f}+/-{baseline.speed_std:.2f}, "
            f"current={baseline.current_median:.2f}+/-{baseline.current_std:.2f}, "
            f"{_format_temp_summary(baseline, cfg)}"
        )
        for col in cfg.columns.prox_cols:
            if col in baseline.prox_medians:
                med = baseline.prox_medians[col]
                std = baseline.prox_stds[col]
                p95 = baseline.prox_maxes.get(col, float("nan"))
                print(f"  Pump {pump_id} vibration {col}: mean={med:.4f}+/-{std:.4f}, P95_max={p95:.4f}")
        for col in cfg.columns.baseline_extra_cols:
            if col in baseline.temp_medians:
                med = baseline.temp_medians[col]
                std = baseline.temp_stds[col]
                print(f"  Pump {pump_id} extra {col}: {med:.4f}+/-{std:.4f}")


def _build_output_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    output_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_df = output_df.sort_values(by=["date", "pump", "period_start"]).reset_index(drop=True)
    return output_df


def _print_final_summary(output_df: pd.DataFrame, cfg: SystemConfig) -> None:
    print("\n=== Final Summary ===")

    total_periods = int(len(output_df))
    normal_count = int((output_df["classification"] == "normal").sum()) if total_periods else 0
    abnormal_count = int((output_df["classification"] == "abnormal").sum()) if total_periods else 0

    print(f"Total periods: {total_periods}")
    print(f"Normal: {normal_count}")
    print(f"Abnormal: {abnormal_count}")

    print("Per-pump breakdown:")
    if total_periods == 0:
        for pump_id in cfg.pump_ids:
            print(f"  Pump {pump_id}: total=0, normal=0, abnormal=0")
        return

    for pump_id in sorted(cfg.pump_ids):
        pump_df = output_df[output_df["pump"] == pump_id]
        pump_total = int(len(pump_df))
        pump_normal = int((pump_df["classification"] == "normal").sum())
        pump_abnormal = int((pump_df["classification"] == "abnormal").sum())
        print(f"  Pump {pump_id}: total={pump_total}, normal={pump_normal}, abnormal={pump_abnormal}")


def _run_validation(output_df: pd.DataFrame, cfg: SystemConfig) -> None:
    if not cfg.examples:
        return

    print("\n=== Pass 3: Validation ===")
    correct = 0

    for example in cfg.examples:
        matched = output_df[
            (output_df["date"] == example.date) &
            (output_df["pump"] == example.pump)
        ]

        if matched.empty:
            got = "missing"
            ok = False
        else:
            classes = [str(v) for v in matched["classification"].dropna().astype(str).tolist()]
            unique_classes = sorted(set(classes))
            ok = example.expected in unique_classes
            got = example.expected if ok else ",".join(unique_classes)

        if ok:
            correct += 1
            print(f"OK {example.date} Pump {example.pump}: expected={example.expected}, got={got}")
        else:
            print(
                f"X {example.date} Pump {example.pump}: "
                f"expected={example.expected}, got={got} <- MISMATCH"
            )

    total = len(cfg.examples)
    percent = (100.0 * correct / total) if total else 0.0
    print(f"Validation: {correct}/{total} correct ({percent:.0f}%)")
    if correct == total and total > 0:
        print("All labelled examples correctly classified.")


def _run_data_selector(
    annotations_path: Path,
    cfg: SystemConfig,
    output_df: pd.DataFrame,
) -> dict | None:
    """Run data selection and return dataset info for the report."""
    from .data_selector import select_train_test, copy_selected_files, write_manifest

    sel_cfg = cfg.data_selector

    print("\n=== Data Selection ===")
    train, test = select_train_test(annotations_path, cfg)

    # Count total pump-days from the output dataframe
    if not output_df.empty:
        pump_days = output_df.groupby(["date", "pump"]).ngroups
    else:
        pump_days = 0

    selected_pump_days = len(train) + len(test)
    excluded = pump_days - selected_pump_days

    print(f"Training candidates: {len(train)}")
    print(f"Testing candidates:  {len(test)}")
    print(f"Excluded (ambiguous): {excluded}")

    # Per-pump breakdown
    for label, selections in [("Train", train), ("Test", test)]:
        pumps: dict[int, int] = {}
        for s in selections:
            pumps[s["pump"]] = pumps.get(s["pump"], 0) + 1
        for p in sorted(pumps):
            print(f"  {label} Pump {p}: {pumps[p]} days")

    copy_stats: dict[str, dict[str, int]] = {}

    if sel_cfg.train_output is not None:
        print(f"\nCopying training files to {sel_cfg.train_output}")
        ok, skip, removed = copy_selected_files(train, sel_cfg.train_output, sel_cfg.use_symlinks)
        print(f"  Copied: {ok}, Skipped: {skip}, Stale removed: {removed}")
        copy_stats["train"] = {"copied": ok, "skipped": skip, "stale_removed": removed}
        manifest_path = sel_cfg.train_output / "train_manifest.csv"
        write_manifest(train, manifest_path)
        print(f"  Manifest: {manifest_path}")

    if sel_cfg.test_output is not None:
        print(f"\nCopying testing files to {sel_cfg.test_output}")
        ok, skip, removed = copy_selected_files(test, sel_cfg.test_output, sel_cfg.use_symlinks)
        print(f"  Copied: {ok}, Skipped: {skip}, Stale removed: {removed}")
        copy_stats["test"] = {"copied": ok, "skipped": skip, "stale_removed": removed}
        manifest_path = sel_cfg.test_output / "test_manifest.csv"
        write_manifest(test, manifest_path)
        print(f"  Manifest: {manifest_path}")

    return {
        "train": train,
        "test": test,
        "excluded_count": excluded,
        "total_pump_days": pump_days,
        "copy_stats": copy_stats,
        "config": {
            "min_normal_duration_minutes": sel_cfg.min_normal_duration_minutes,
            "max_periods_per_day": sel_cfg.max_periods_per_day,
            "exclude_reasons": list(sel_cfg.exclude_reasons),
        },
    }


def run_pipeline(
    cfg: SystemConfig,
    output_path: Path,
    baseline_root: Path | None = None,
    no_ml_ensemble: bool = False,
) -> tuple[int, pd.DataFrame | None]:
    print("Starting pump annotation pipeline")
    print(f"System: {cfg.system_name}")
    print(f"Data root: {cfg.data_root}")
    print(f"Output file: {output_path}")
    print(f"Anomaly sigma: {cfg.thresholds.anomaly_sigma:.3f}")

    file_list = discover_files(cfg)
    _print_discovery_summary(file_list, cfg)

    if baseline_root is not None:
        baseline_cfg = replace(cfg, data_root=baseline_root)
        baseline_file_list = discover_files(baseline_cfg)
        print(f"\n=== Baseline Source: {baseline_root} ===")
        _print_discovery_summary(baseline_file_list, baseline_cfg)
        if not baseline_file_list:
            print("ERROR: No files found in baseline-root directory. Cannot compute baselines.")
            return 1, None
    else:
        baseline_file_list = file_list

    if not file_list:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        empty_df.to_csv(output_path, index=False)
        print(f"No input files found. Wrote empty output to {output_path}")
        _print_final_summary(empty_df, cfg)
        _run_validation(empty_df, cfg)
        return 1, None

    custom_detectors = load_custom_detectors(cfg)
    if no_ml_ensemble:
        custom_detectors = {k: v for k, v in custom_detectors.items() if k != "ml_ensemble"}
        print("[info] ML ensemble detector disabled (--no-ml-ensemble flag).")
    max_passes = 1  # Single pass (refinement disabled while tuning variability cap)
    exclude_files: set[Path] = set()

    for pass_num in range(1, max_passes + 1):
        pass_label = "initial" if pass_num == 1 else f"refined (pass {pass_num})"

        if exclude_files:
            print(f"\n=== Baseline Pass {pass_num} ({pass_label}) - excluding {len(exclude_files)} abnormal files ===")

        baselines = compute_baselines(baseline_file_list, cfg, exclude_files=exclude_files if exclude_files else None)
        _print_baseline_summary(baselines, cfg)

        print(f"\n=== Classification (pass {pass_num}) ===")
        results: list[dict[str, object]] = []
        skipped_no_baseline: set[int] = set()
        total_files = len(file_list)
        new_exclude: set[Path] = set()

        for idx, (file_date, pump_id, path) in enumerate(file_list, start=1):
            if idx % 100 == 0 or idx == total_files:
                print(
                    f"[progress] processed {idx}/{total_files} files, "
                    f"annotated periods so far: {len(results)}"
                )

            df = load_csv(path, cfg)
            if df is None or df.empty:
                continue

            sampling_interval = _safe_sampling_interval(df)
            periods = detect_periods(df, sampling_interval, cfg)
            if not periods:
                continue

            baseline = baselines.get(pump_id)
            if baseline is None:
                if pump_id not in skipped_no_baseline:
                    print(f"[warning] missing baseline for pump {pump_id}; skipping its periods")
                    skipped_no_baseline.add(pump_id)
                continue

            file_has_abnormal = False
            for period in periods:
                result = classify_period(df, period, baseline, cfg, custom_detectors)
                reasons = "" if result.classification == "normal" else _normalize_reasons(result.reasons)

                results.append(
                    {
                        "date": file_date.strftime("%Y-%m-%d"),
                        "pump": int(pump_id),
                        "period_start": period.start.strftime("%H:%M"),
                        "period_end": period.end.strftime("%H:%M"),
                        "duration_minutes": round(float(period.duration_seconds) / 60.0, 1),
                        "classification": result.classification,
                        "reasons": reasons,
                    }
                )
                if result.classification == "abnormal":
                    file_has_abnormal = True

            if file_has_abnormal:
                new_exclude.add(path)

        # Check if refinement would be meaningful
        if pass_num < max_passes:
            added = new_exclude - exclude_files
            if not added:
                print(f"\nNo new abnormal files found in pass {pass_num}; skipping refinement.")
                break

            # Safety check: don't exclude too many files
            remaining_per_pump: dict[int, int] = {}
            for _, pump_id, path in file_list:
                if path not in new_exclude:
                    remaining_per_pump[pump_id] = remaining_per_pump.get(pump_id, 0) + 1

            min_remaining = min(remaining_per_pump.values()) if remaining_per_pump else 0
            if min_remaining < 50:
                print(f"\nWARNING: Refinement would leave only {min_remaining} files for a pump. Keeping current baseline.")
                break

            exclude_files = new_exclude
            print(f"\nRefinement: excluding {len(exclude_files)} files with abnormal periods for baseline recomputation.")

    output_df = _build_output_dataframe(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"\nSaved annotations to {output_path}")
    _print_final_summary(output_df, cfg)
    _run_validation(output_df, cfg)
    return 0, output_df


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg, out_csv, baseline_root = _apply_overrides(cfg, args.data_root, args.baseline_root, args.output, args.sigma)
    rc, df_out = run_pipeline(cfg, out_csv, baseline_root=baseline_root,
                              no_ml_ensemble=args.no_ml_ensemble)

    dataset_info = None
    if rc == 0 and df_out is not None and not args.no_data_selector:
        dataset_info = _run_data_selector(out_csv, cfg, df_out)

    if rc == 0 and df_out is not None and not args.no_html:
        from .report import generate_report
        report_path = Path(out_csv).with_suffix(".html") if out_csv else Path(f"{cfg.system_name}_report.html")
        generate_report(df_out, cfg, report_path, dataset_info=dataset_info)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
