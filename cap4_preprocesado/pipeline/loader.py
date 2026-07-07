import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .system_config import SystemConfig


def discover_files(cfg: SystemConfig) -> list[tuple[date, int, Path]]:
    """Discover pump CSV files and return sorted (date, pump_id, path) tuples."""
    found: list[tuple[date, int, Path]] = []

    for path in cfg.data_root.rglob("*.csv"):
        match = cfg.compiled_file_pattern.match(path.name)
        if not match:
            continue

        pump_id = int(match.group(1))
        if pump_id not in cfg.pump_ids:
            continue

        try:
            file_date = datetime.strptime(match.group(2), cfg.date_format).date()
        except ValueError:
            continue

        found.append((file_date, pump_id, path))

    found.sort(key=lambda x: (x[0], x[1], x[2]))
    return found


def load_csv(filepath: Path, cfg: SystemConfig) -> pd.DataFrame | None:
    """Load one pump CSV, normalize schema, and attach sampling interval metadata."""
    try:
        raw_df = pd.read_csv(filepath, header=0)
    except Exception as exc:  # pragma: no cover - defensive for corrupt files
        warnings.warn(f"Failed to read {filepath}: {exc}")
        return None

    if raw_df.empty:
        warnings.warn(f"Empty file skipped: {filepath}")
        return None

    expected_source_count = cfg.columns.expected_source_count
    post_drop_count = expected_source_count - len(cfg.columns.exclude_indices)
    clean_names = cfg.columns.clean_names

    # Phase 1: extract the stable positional core columns.
    if raw_df.shape[1] >= expected_source_count:
        core = raw_df.iloc[:, :expected_source_count].copy()
        core = core.drop(columns=core.columns[cfg.columns.exclude_indices])
    elif raw_df.shape[1] == post_drop_count:
        core = raw_df.iloc[:, :post_drop_count].copy()
    else:
        warnings.warn(
            f"Unexpected column count in {filepath}: {raw_df.shape[1]} "
            f"(expected >= {expected_source_count} or {post_drop_count})"
        )
        return None

    if core.shape[1] != len(clean_names):
        warnings.warn(
            f"Unexpected post-drop column count in {filepath}: {core.shape[1]} (expected {len(clean_names)})"
        )
        return None

    core.columns = clean_names

    # Phase 2: attach variable columns by source header name.
    for source_name, clean_name in cfg.columns.supplementary_columns.items():
        if source_name in raw_df.columns:
            core[clean_name] = raw_df[source_name].values
        else:
            core[clean_name] = np.nan

    df = core

    timestamp_col = cfg.columns.timestamp_col
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).reset_index(drop=True)

    if df.empty:
        warnings.warn(f"No valid timestamp rows after parsing: {filepath}")
        return None

    if len(df) > 1:
        deltas = df[timestamp_col].diff().dt.total_seconds().dropna()
        sampling_interval_sec = float(np.nanmedian(deltas)) if not deltas.empty else float("nan")
    else:
        sampling_interval_sec = float("nan")

    df.attrs["sampling_interval_sec"] = sampling_interval_sec
    df.attrs["source_file"] = str(filepath)
    return df
