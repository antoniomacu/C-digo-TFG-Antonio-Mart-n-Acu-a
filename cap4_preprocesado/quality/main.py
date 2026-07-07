from __future__ import annotations

import argparse
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from quality.quality_config import load_quality_and_system_config, QualityConfig
from pipeline.system_config import SystemConfig
from quality.labeler import label_dataframe, is_all_frozen
from quality.cleaner import clean_dataframe
from quality.operational_filter import filter_operational
from quality.resampler import resample_dataframe
from quality.reporter import generate_text_summary, generate_html_dashboard

VALID_STEPS = {"label", "clean", "filter", "resample"}


def _resolve_path(value: str | Path, config_path: Path) -> Path:
    base_dir = config_path.parent
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (base_dir / raw).resolve()


def _extract_groups(match: re.Match[str]) -> tuple[int, str]:
    group_dict = match.groupdict()

    if "pump_id" in group_dict:
        pump_id = int(group_dict["pump_id"])
    else:
        pump_id = int(match.group(1))

    if "date" in group_dict:
        date_str = str(group_dict["date"])
    else:
        date_str = str(match.group(2))

    return pump_id, date_str


def parse_steps(steps_raw: str) -> set[str]:
    requested = {step.strip().lower() for step in steps_raw.split(",") if step.strip()}
    if not requested:
        return {"label", "clean", "filter", "resample"}

    invalid = sorted(requested - VALID_STEPS)
    if invalid:
        raise SystemExit(
            f"Invalid --steps values: {', '.join(invalid)}. "
            f"Valid values: {', '.join(sorted(VALID_STEPS))}"
        )
    return requested


def parse_pumps(pumps_raw: str, allowed_pumps: list[int]) -> set[int] | None:
    if not pumps_raw.strip():
        return None

    selected: set[int] = set()
    allowed_set = set(allowed_pumps)

    for token in pumps_raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            pump_id = int(token)
        except ValueError as exc:
            raise SystemExit(f"Invalid pump ID in --pump: {token}") from exc

        if pump_id not in allowed_set:
            raise SystemExit(
                f"Pump ID {pump_id} not found in config pump_ids: {sorted(allowed_set)}"
            )
        selected.add(pump_id)

    return selected or None


def process_file(
    path: Path,
    system_config: SystemConfig,
    quality_config: QualityConfig,
    output_dir: Path,
    steps: set[str],
    dry_run: bool,
) -> dict:
    pattern = system_config.compiled_file_pattern
    match = pattern.match(path.name)

    pump_id = None
    date_str = "unknown"
    if match:
        pump_id, date_str = _extract_groups(match)

    raw_df = pd.read_csv(path)
    input_rows = len(raw_df)

    timestamp_col = system_config.columns.timestamp_col
    if timestamp_col not in raw_df.columns:
        return {
            "file": path.name,
            "pump_id": pump_id,
            "date": date_str,
            "input_rows": input_rows,
            "output_rows": 0,
            "removed_rows": input_rows,
            "skipped": True,
            "skip_reason": "missing_timestamp",
            "quality_counts": {},
            "sentinel_rows": 0,
            "written": False,
            "error": "",
        }

    raw_df[timestamp_col] = pd.to_datetime(raw_df[timestamp_col], errors="coerce")
    working_df = raw_df.dropna(subset=[timestamp_col]).copy()
    working_df = working_df.sort_values(timestamp_col)
    working_df = working_df.drop_duplicates(subset=[timestamp_col], keep="first")
    working_df = working_df.set_index(timestamp_col)

    sensor_columns = [
        sensor for sensor in quality_config.get_sensor_columns() if sensor in working_df.columns
    ]

    for sensor in sensor_columns:
        working_df[sensor] = pd.to_numeric(working_df[sensor], errors="coerce")

    if "label" in steps:
        working_df = label_dataframe(working_df, quality_config)

    quality_counts: dict[str, dict[int, int]] = {}
    for sensor in sensor_columns:
        quality_col = f"{sensor}_quality"
        if quality_col not in working_df.columns:
            continue
        counts = working_df[quality_col].value_counts(dropna=False).to_dict()
        quality_counts[sensor] = {int(code): int(count) for code, count in counts.items()}

    if quality_config.skip_all_frozen_files and is_all_frozen(
        working_df,
        sensor_columns,
        quality_config.frozen_tolerance,
    ):
        return {
            "file": path.name,
            "pump_id": pump_id,
            "date": date_str,
            "input_rows": input_rows,
            "output_rows": 0,
            "removed_rows": input_rows,
            "skipped": True,
            "skip_reason": "all_frozen",
            "quality_counts": quality_counts,
            "sentinel_rows": 0,
            "written": False,
            "error": "",
        }

    processed_df = working_df

    if "clean" in steps:
        processed_df = clean_dataframe(
            processed_df,
            quality_config,
            sensor_columns,
        )

    if "filter" in steps and quality_config.operational_filter:
        processed_df = filter_operational(
            processed_df,
            speed_col=system_config.columns.speed_col,
            threshold=quality_config.speed_on_threshold,
            min_consecutive=quality_config.min_consecutive_operational,
        )

    if "resample" in steps:
        processed_df = resample_dataframe(processed_df, quality_config, sensor_columns)

    sentinel_rows = int(processed_df.attrs.get("sentinel_rows", 0))
    output_rows = len(processed_df)

    skipped = output_rows == 0
    skip_reason = "no_rows_after_steps" if skipped else ""
    written = False

    if not skipped and not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / path.name
        processed_df.reset_index().to_csv(out_path, index=False)
        written = True

    return {
        "file": path.name,
        "pump_id": pump_id,
        "date": date_str,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "removed_rows": max(0, input_rows - output_rows),
        "skipped": skipped,
        "skip_reason": skip_reason,
        "quality_counts": quality_counts,
        "sentinel_rows": sentinel_rows,
        "written": written,
        "error": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generalized quality pipeline from YAML configuration.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML config file (for example configs/htf_pumps.yaml).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override system data_root from config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override quality.output_dir from config.",
    )
    parser.add_argument(
        "--pump",
        type=str,
        default="",
        help="Comma-separated pump IDs to process (default: all from config).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats without writing output files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file details.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="label,clean,filter,resample",
        help="Comma-separated steps: label,clean,filter,resample.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for file processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = args.config.expanduser().resolve()
    quality_config, system_config = load_quality_and_system_config(config_path)

    input_dir = (
        _resolve_path(args.input_dir, config_path)
        if args.input_dir is not None
        else _resolve_path(system_config.data_root, config_path)
    )
    output_dir = (
        _resolve_path(args.output_dir, config_path)
        if args.output_dir is not None
        else _resolve_path(quality_config.output_dir, config_path)
    )

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    selected_pumps = parse_pumps(args.pump, system_config.pump_ids)
    steps = parse_steps(args.steps)

    pattern = system_config.compiled_file_pattern
    file_info: list[tuple[int, str, Path]] = []

    for path in sorted(input_dir.glob("*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue

        pump_id, date_str = _extract_groups(match)
        if selected_pumps is not None and pump_id not in selected_pumps:
            continue

        file_info.append((pump_id, date_str, path))

    if not file_info:
        raise SystemExit(f"No matching files found in: {input_dir}")

    files_by_pump: dict[int, list[tuple[str, Path]]] = defaultdict(list)
    for pump_id, date_str, path in file_info:
        files_by_pump[pump_id].append((date_str, path))

    total_files = len(file_info)
    print(f"Quality Pipeline: {system_config.system_name}")
    print(f"Input:  {input_dir} ({total_files} files)")
    print(f"Output: {output_dir}")
    print(f"Steps:  {', '.join(sorted(steps))}")

    all_stats: list[dict] = []

    def _run(single_path: Path) -> dict:
        try:
            return process_file(
                path=single_path,
                system_config=system_config,
                quality_config=quality_config,
                output_dir=output_dir,
                steps=steps,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # pragma: no cover
            return {
                "file": single_path.name,
                "pump_id": None,
                "date": "unknown",
                "input_rows": 0,
                "output_rows": 0,
                "removed_rows": 0,
                "skipped": True,
                "skip_reason": "error",
                "quality_counts": {},
                "sentinel_rows": 0,
                "written": False,
                "error": str(exc),
            }

    for pump_id in sorted(files_by_pump):
        pump_files = sorted(files_by_pump[pump_id], key=lambda item: item[0])
        print(f"\nProcessing pump {pump_id} ({len(pump_files)} files)")

        paths = [path for _, path in pump_files]
        if args.workers <= 1:
            stats_list = [_run(path) for path in paths]
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                stats_list = list(executor.map(_run, paths))

        for stats in stats_list:
            all_stats.append(stats)
            if args.verbose:
                if stats.get("error"):
                    print(f"  {stats['file']}: error ({stats['error']})")
                elif stats["skipped"]:
                    print(
                        f"  {stats['file']}: {stats['input_rows']} -> {stats['output_rows']} rows "
                        f"(skipped: {stats['skip_reason']})"
                    )
                else:
                    print(
                        f"  {stats['file']}: {stats['input_rows']} -> {stats['output_rows']} rows "
                        f"(removed: {stats['removed_rows']})"
                    )

    summary_path = output_dir / quality_config.summary_output
    generate_text_summary(
        all_stats=all_stats,
        system_name=system_config.system_name,
        output_path=summary_path,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        dashboard_path = _resolve_path(quality_config.dashboard_output, config_path)
        generate_html_dashboard(
            all_stats=all_stats,
            quality_config=quality_config,
            system_name=system_config.system_name,
            output_path=dashboard_path,
        )


if __name__ == "__main__":
    main()
