"""Data selector — curates high-confidence train/test datasets from pipeline annotations.

Reads the annotation CSV produced by the pipeline and selects:
- Training data: days where ALL periods are confidently normal
- Testing data: days where at least one period is clearly abnormal

Conservative selection ensures no ambiguous days contaminate the training set.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from .system_config import SystemConfig, load_config


def _find_source_files(date_str: str, pump_id: int, cfg: SystemConfig) -> list[Path]:
    """Find all source CSV files for a given date and pump (including _pN splits)."""
    try:
        dt = datetime.strptime(date_str, cfg.date_format)
    except ValueError:
        return []

    day_dir = cfg.data_root / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{dt.day:02d}"
    if not day_dir.exists():
        return []

    prefix = f"fw_pump_{pump_id}_{date_str}"
    matches = sorted(
        p for p in day_dir.iterdir()
        if p.name.startswith(prefix) and p.name.endswith(".csv") and p.is_file()
    )
    return matches


def select_train_test(
    annotations_path: Path,
    cfg: SystemConfig,
) -> tuple[list[dict], list[dict]]:
    """Select high-confidence training and testing days from annotations.
    
    Returns (train_selections, test_selections) where each is a list of dicts
    with keys: date, pump, source_path, reason (why selected/rejected).
    """
    df = pd.read_csv(annotations_path)
    
    if df.empty:
        return [], []
    
    sel_cfg = cfg.data_selector
    train_selections: list[dict] = []
    test_selections: list[dict] = []
    
    # Group by (date, pump)
    grouped = df.groupby(["date", "pump"])
    
    for (date_str, pump_id), group in grouped:
        date_str = str(date_str)
        pump_id = int(pump_id)
        
        n_periods = len(group)
        classifications = group["classification"].tolist()
        durations = group["duration_minutes"].astype(float).tolist()
        all_reasons = group["reasons"].fillna("").tolist()
        
        has_abnormal = any(c == "abnormal" for c in classifications)
        all_normal = all(c == "normal" for c in classifications)
        
        source_paths = _find_source_files(date_str, pump_id, cfg)
        
        # --- TEST selection: any abnormal period ---
        if has_abnormal:
            # Check if we should exclude based on reasons
            abnormal_reasons = [r for r, c in zip(all_reasons, classifications) if c == "abnormal" and r]
            
            if sel_cfg.exclude_reasons:
                # Skip if ALL abnormal reasons are in the exclude list
                all_excluded = all(
                    all(token.split(":")[0] in sel_cfg.exclude_reasons for token in r.split(";") if token)
                    for r in abnormal_reasons if r
                )
                if all_excluded and abnormal_reasons:
                    continue
            
            test_selections.append({
                "date": date_str,
                "pump": pump_id,
                "source_path": ";".join(str(p) for p in source_paths) if source_paths else None,
                "n_periods": n_periods,
                "reasons": ";".join(set(r for r in all_reasons if r)),
                "selection": "test",
            })
            continue
        
        # --- TRAIN selection: conservative criteria ---
        if not all_normal:
            continue
        
        # Check period count
        if n_periods > sel_cfg.max_periods_per_day:
            continue
        
        # Check all periods are above minimum duration
        if any(d < sel_cfg.min_normal_duration_minutes for d in durations):
            continue
        
        # Must have at least 1 period (skip idle days)
        if n_periods == 0:
            continue
        
        train_selections.append({
            "date": date_str,
            "pump": pump_id,
            "source_path": ";".join(str(p) for p in source_paths) if source_paths else None,
            "n_periods": n_periods,
            "min_duration": min(durations),
            "max_duration": max(durations),
            "selection": "train",
        })
    
    return train_selections, test_selections


def copy_selected_files(
    selections: list[dict],
    output_dir: Path,
    use_symlinks: bool = False,
) -> tuple[int, int, int]:
    """Copy or symlink selected source files to output directory.
    
    Returns (success_count, skip_count, removed_stale_count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the set of file names that are expected to be present after copy.
    expected_filenames: set[str] = set()
    for item in selections:
        source = item.get("source_path")
        if source is None:
            continue
        for src in str(source).split(";"):
            src = src.strip()
            if not src:
                continue
            source_path = Path(src)
            if source_path.exists():
                expected_filenames.add(source_path.name)

    removed_stale = 0
    for existing_csv in output_dir.glob("*.csv"):
        if existing_csv.name.endswith("_manifest.csv"):
            continue
        if existing_csv.name not in expected_filenames:
            existing_csv.unlink()
            removed_stale += 1
    
    success = 0
    skipped = 0
    
    for item in selections:
        source = item.get("source_path")
        if source is None:
            skipped += 1
            continue

        paths = [Path(s.strip()) for s in str(source).split(";") if s.strip()]
        if not paths or not any(p.exists() for p in paths):
            skipped += 1
            continue

        for source_path in paths:
            if not source_path.exists():
                continue
            target_path = output_dir / source_path.name

            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()

            if use_symlinks:
                target_path.symlink_to(source_path.resolve())
            else:
                shutil.copy2(source_path, target_path)

            success += 1
    
    return success, skipped, removed_stale


def write_manifest(selections: list[dict], output_path: Path) -> None:
    """Write a manifest CSV listing selected files and their metadata."""
    if not selections:
        pd.DataFrame().to_csv(output_path, index=False)
        return
    
    manifest_df = pd.DataFrame(selections)
    manifest_df = manifest_df.sort_values(["date", "pump"]).reset_index(drop=True)
    manifest_df.to_csv(output_path, index=False)


def run_selector(cfg: SystemConfig, annotations_path: Path) -> None:
    """Run the full data selection pipeline."""
    sel_cfg = cfg.data_selector
    
    print(f"Reading annotations from {annotations_path}")
    train, test = select_train_test(annotations_path, cfg)
    
    print(f"\n=== Selection Summary ===")
    print(f"Training candidates: {len(train)}")
    print(f"Testing candidates:  {len(test)}")
    
    # Per-pump breakdown
    for label, selections in [("Train", train), ("Test", test)]:
        pumps: dict[int, int] = {}
        for s in selections:
            pumps[s["pump"]] = pumps.get(s["pump"], 0) + 1
        for p in sorted(pumps):
            print(f"  {label} Pump {p}: {pumps[p]} days")
    
    total_excluded = 0  # Days that are neither train nor test
    # (calculated from annotations)
    try:
        df = pd.read_csv(annotations_path)
        all_pump_days = set()
        for _, row in df.iterrows():
            all_pump_days.add((str(row["date"]), int(row["pump"])))
        selected_pump_days = set()
        for s in train + test:
            selected_pump_days.add((s["date"], s["pump"]))
        total_excluded = len(all_pump_days) - len(selected_pump_days)
    except Exception:
        pass
    
    print(f"Excluded (ambiguous): {total_excluded}")
    
    # Copy/link files
    if sel_cfg.train_output is not None:
        print(f"\nCopying training files to {sel_cfg.train_output}")
        ok, skip, removed = copy_selected_files(train, sel_cfg.train_output, sel_cfg.use_symlinks)
        print(f"  Copied: {ok}, Skipped (missing source): {skip}, Stale removed: {removed}")
        manifest_path = sel_cfg.train_output / "train_manifest.csv"
        write_manifest(train, manifest_path)
        print(f"  Manifest: {manifest_path}")
    
    if sel_cfg.test_output is not None:
        print(f"\nCopying testing files to {sel_cfg.test_output}")
        ok, skip, removed = copy_selected_files(test, sel_cfg.test_output, sel_cfg.use_symlinks)
        print(f"  Copied: {ok}, Skipped (missing source): {skip}, Stale removed: {removed}")
        manifest_path = sel_cfg.test_output / "test_manifest.csv"
        write_manifest(test, manifest_path)
        print(f"  Manifest: {manifest_path}")
    
    print("\nData selection complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select high-confidence train/test data from pipeline annotations."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to system YAML config")
    parser.add_argument("--annotations", type=Path, required=True, help="Path to annotations CSV")
    parser.add_argument("--train-output", type=Path, default=None, help="Override training output dir")
    parser.add_argument("--test-output", type=Path, default=None, help="Override testing output dir")
    parser.add_argument("--min-duration", type=float, default=None, help="Override min normal duration (minutes)")
    parser.add_argument("--max-periods", type=int, default=None, help="Override max periods per day")
    parser.add_argument("--symlinks", action="store_true", default=False, help="Use symlinks instead of copies")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    
    # Apply CLI overrides via dataclass replace
    from dataclasses import replace
    sel = cfg.data_selector
    
    if args.train_output is not None:
        sel = replace(sel, train_output=args.train_output)
    if args.test_output is not None:
        sel = replace(sel, test_output=args.test_output)
    if args.min_duration is not None:
        sel = replace(sel, min_normal_duration_minutes=args.min_duration)
    if args.max_periods is not None:
        sel = replace(sel, max_periods_per_day=args.max_periods)
    if args.symlinks:
        sel = replace(sel, use_symlinks=True)
    
    cfg = replace(cfg, data_selector=sel)
    
    run_selector(cfg, args.annotations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
