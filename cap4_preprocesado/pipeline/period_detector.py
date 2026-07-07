from dataclasses import dataclass

import numpy as np
import pandas as pd

from .system_config import SystemConfig


@dataclass
class OperationPeriod:
    start: pd.Timestamp
    end: pd.Timestamp
    duration_seconds: float
    n_rows: int


def detect_periods(
    df: pd.DataFrame,
    sampling_interval_sec: float,
    cfg: SystemConfig,
) -> list[OperationPeriod]:
    """Detect contiguous pump ON periods, merge short gaps, and filter short operations."""
    speed_col = cfg.columns.speed_col
    timestamp_col = cfg.columns.timestamp_col

    if df.empty or speed_col not in df.columns or timestamp_col not in df.columns:
        return []

    is_on = (df[speed_col] > cfg.thresholds.speed_on_threshold).fillna(False)
    on_array = is_on.astype(int).to_numpy()

    if on_array.size == 0 or on_array.max() == 0:
        return []

    changes = np.diff(on_array)
    starts = list(np.where(changes == 1)[0] + 1)
    ends = list(np.where(changes == -1)[0])

    if on_array[0] == 1:
        starts.insert(0, 0)
    if on_array[-1] == 1:
        ends.append(len(on_array) - 1)

    segments = list(zip(starts, ends))
    if not segments:
        return []

    timestamps = df[timestamp_col]
    merged: list[tuple[int, int]] = [segments[0]]

    for next_start, next_end in segments[1:]:
        cur_start, cur_end = merged[-1]
        gap_seconds = (timestamps.iloc[next_start] - timestamps.iloc[cur_end]).total_seconds() - sampling_interval_sec

        if gap_seconds <= cfg.thresholds.gap_merge_threshold:
            merged[-1] = (cur_start, next_end)
        else:
            merged.append((next_start, next_end))

    required_duration = cfg.thresholds.min_period_duration
    if sampling_interval_sec >= 240.0:
        required_duration = max(required_duration, sampling_interval_sec * 3.0)

    periods: list[OperationPeriod] = []
    for start_idx, end_idx in merged:
        start_ts = timestamps.iloc[start_idx]
        end_ts = timestamps.iloc[end_idx]
        n_rows = int(end_idx - start_idx + 1)

        duration_seconds = float((end_ts - start_ts).total_seconds() + sampling_interval_sec)
        if duration_seconds < required_duration:
            continue

        periods.append(
            OperationPeriod(
                start=start_ts,
                end=end_ts,
                duration_seconds=duration_seconds,
                n_rows=n_rows,
            )
        )

    return periods


def get_steady_state_mask(df: pd.DataFrame, period: OperationPeriod, cfg: SystemConfig) -> pd.Series:
    """Return a mask for steady-state rows within one operation period."""
    steady_mask = pd.Series(False, index=df.index)
    timestamp_col = cfg.columns.timestamp_col

    if df.empty or timestamp_col not in df.columns:
        return steady_mask

    ts = df[timestamp_col]
    period_mask = (ts >= period.start) & (ts <= period.end)
    if period_mask.sum() == 0:
        return steady_mask

    steady_start = period.start + pd.Timedelta(seconds=cfg.thresholds.ramp_exclude_seconds)
    steady_end = period.end - pd.Timedelta(seconds=cfg.thresholds.ramp_exclude_seconds)

    if steady_start < steady_end:
        candidate = period_mask & (ts >= steady_start) & (ts <= steady_end)
        steady_window_seconds = float((steady_end - steady_start).total_seconds())
        if steady_window_seconds >= 120.0 and candidate.any():
            return candidate

    # Fallback for short periods: use the middle 50% of rows.
    period_indices = np.flatnonzero(period_mask.to_numpy())
    n = int(period_indices.size)
    start_offset = int(np.floor(n * 0.25))
    end_offset = int(np.ceil(n * 0.75))
    if end_offset <= start_offset:
        end_offset = min(n, start_offset + 1)

    selected = period_indices[start_offset:end_offset]
    if selected.size > 0:
        steady_mask.iloc[selected] = True

    return steady_mask
