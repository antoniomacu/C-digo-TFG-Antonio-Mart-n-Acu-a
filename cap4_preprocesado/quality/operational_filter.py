from __future__ import annotations

import numpy as np
import pandas as pd


def _get_valid_operational_ranges(
    df: pd.DataFrame,
    speed_col: str,
    threshold: float,
    min_consecutive: int,
) -> list[tuple[int, int]]:
    """Return [start, end) positional ranges for valid operational segments."""
    if df.empty or speed_col not in df.columns:
        return []

    if min_consecutive <= 0:
        min_consecutive = 1

    speed = pd.to_numeric(df[speed_col], errors="coerce")
    operational = speed.ge(threshold).to_numpy(dtype=bool)

    transitions = np.diff(operational.astype(np.int8), prepend=0, append=0)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    ranges: list[tuple[int, int]] = []
    for start, end in zip(starts, ends):
        if (end - start) >= min_consecutive:
            ranges.append((int(start), int(end)))

    return ranges


def filter_operational(
    df: pd.DataFrame,
    speed_col: str,
    threshold: float,
    min_consecutive: int,
) -> pd.DataFrame:
    """Filter to rows belonging to valid operational periods."""
    if df.empty or speed_col not in df.columns:
        return df.iloc[0:0].copy()

    ranges = _get_valid_operational_ranges(df, speed_col, threshold, min_consecutive)
    if not ranges:
        return df.iloc[0:0].copy()

    keep_mask = np.zeros(len(df), dtype=bool)
    for start, end in ranges:
        keep_mask[start:end] = True

    return df.iloc[keep_mask].copy()


def split_operational_periods(
    df: pd.DataFrame,
    speed_col: str,
    threshold: float,
    min_consecutive: int,
) -> list[pd.DataFrame]:
    """Split a dataframe into valid operational periods as separate dataframes."""
    if df.empty or speed_col not in df.columns:
        return []

    ranges = _get_valid_operational_ranges(df, speed_col, threshold, min_consecutive)
    if not ranges:
        return []

    periods: list[pd.DataFrame] = []
    for start, end in ranges:
        periods.append(df.iloc[start:end].copy())

    return periods
