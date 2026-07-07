from __future__ import annotations

import numpy as np
import pandas as pd

from quality.quality_config import QualityConfig, SensorQualityConfig

Q_OK = 0
Q_MISSING = 1
Q_NULL = 2
Q_OUT_OF_RANGE = 3
Q_FALL_DOWN = 4
Q_FROZEN = 5

CODE_NAMES = {
    Q_OK: "OK",
    Q_MISSING: "Missing",
    Q_NULL: "Null",
    Q_OUT_OF_RANGE: "Out of range",
    Q_FALL_DOWN: "Fall down",
    Q_FROZEN: "Frozen",
}


def compute_quality_column(
    df: pd.DataFrame,
    col_name: str,
    sensor_config: SensorQualityConfig,
    timestep_seconds: int,
    frozen_tolerance: float,
    frozen_period_seconds: int,
    variability_threshold: float,
    rolling_window_samples: int,
    rolling_min_periods: int,
) -> pd.DataFrame:
    """Compute and append a quality column for a single sensor column."""

    if col_name not in df.columns:
        return df

    qcol = f"{col_name}_quality"
    out = df

    missing_mask: np.ndarray | None = None
    if not out.empty and timestep_seconds > 0:
        expected = pd.date_range(
            start=out.index.min(),
            end=out.index.max(),
            freq=f"{timestep_seconds}s",
        )
        idx_set = set(out.index)
        missing_ts = sorted(set(expected) - idx_set)
        if missing_ts:
            missing_set = set(missing_ts)
            missing_df = pd.DataFrame(
                {col_name: np.nan},
                index=pd.DatetimeIndex(missing_ts, name=out.index.name),
            )
            out = pd.concat([out, missing_df])
            out.sort_index(inplace=True)
            missing_mask = np.array([ts in missing_set for ts in out.index], dtype=bool)

    values = pd.to_numeric(out[col_name], errors="coerce").to_numpy(dtype=np.float64)
    n_values = len(values)
    quality_codes = np.full(n_values, Q_OK, dtype=np.int8)

    is_nan = np.isnan(values)
    quality_codes[is_nan] = Q_NULL
    if missing_mask is not None:
        quality_codes[missing_mask] = Q_MISSING

    if not is_nan.all():
        min_value = sensor_config.min_value
        max_value = sensor_config.max_value

        if min_value is not None or max_value is not None:
            out_of_range = np.zeros(n_values, dtype=bool)
            if min_value is not None:
                out_of_range |= values < float(min_value)
            if max_value is not None:
                out_of_range |= values > float(max_value)
            upgrade = out_of_range & (quality_codes < Q_OUT_OF_RANGE)
            quality_codes[upgrade] = Q_OUT_OF_RANGE

        if (
            sensor_config.check_falldown
            and min_value is not None
            and max_value is not None
        ):
            value_range = abs(float(max_value) - float(min_value))
            if value_range > 0:
                rolling_std = out[col_name].rolling(
                    window=rolling_window_samples,
                    min_periods=rolling_min_periods,
                ).std()
                ratio = rolling_std.to_numpy(dtype=np.float64) / value_range
                fall_down = np.abs(ratio) > variability_threshold
                upgrade = fall_down & (quality_codes < Q_FALL_DOWN)
                quality_codes[upgrade] = Q_FALL_DOWN

        if sensor_config.check_constant and timestep_seconds > 0:
            diff = np.abs(np.diff(values, prepend=np.nan))
            frozen = diff < frozen_tolerance
            points_needed = max(1, int(frozen_period_seconds / timestep_seconds))
            changes = np.diff(frozen.astype(np.int8), prepend=0, append=0)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            for start, end in zip(starts, ends):
                if (end - start) >= points_needed:
                    quality_codes[start:end] = np.maximum(
                        quality_codes[start:end],
                        Q_FROZEN,
                    )

    out[qcol] = quality_codes
    return out


def label_dataframe(df: pd.DataFrame, quality_config: QualityConfig) -> pd.DataFrame:
    """Apply quality labeling to every configured sensor that exists in the DataFrame."""

    out = df
    for sensor_name, sensor_config in quality_config.sensors.items():
        if sensor_name not in out.columns:
            continue

        out = compute_quality_column(
            df=out,
            col_name=sensor_name,
            sensor_config=sensor_config,
            timestep_seconds=quality_config.timestep_seconds,
            frozen_tolerance=quality_config.frozen_tolerance,
            frozen_period_seconds=quality_config.frozen_period_seconds,
            variability_threshold=quality_config.variability_threshold,
            rolling_window_samples=quality_config.rolling_window_samples,
            rolling_min_periods=quality_config.rolling_min_periods,
        )

    return out


def is_all_frozen(
    df: pd.DataFrame,
    sensor_columns: list[str],
    tolerance: float,
) -> bool:
    """Return True when every checked sensor has no variation within tolerance."""

    checked_columns = 0

    for col_name in sensor_columns:
        if col_name not in df.columns:
            continue

        values = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=np.float64)
        not_nan = ~np.isnan(values)
        if int(np.count_nonzero(not_nan)) <= 1:
            continue

        checked_columns += 1
        series = values[not_nan]
        diffs = np.abs(np.diff(series))
        if np.any(diffs >= tolerance):
            return False

    return checked_columns > 0


def detect_sentinel_rows(
    df: pd.DataFrame,
    sensor_columns: list[str],
    sentinel_value: float | None,
    threshold: int,
) -> np.ndarray:
    """Return a mask for rows where at least threshold columns hit sentinel_value."""

    if sentinel_value is None or df.empty or threshold > len(sensor_columns):
        return np.zeros(len(df), dtype=bool)

    existing_columns = [col for col in sensor_columns if col in df.columns]
    if not existing_columns:
        return np.zeros(len(df), dtype=bool)

    if threshold <= 0:
        return np.ones(len(df), dtype=bool)

    matrix = (
        df[existing_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    sentinel_hits = np.isclose(matrix, float(sentinel_value))
    return sentinel_hits.sum(axis=1) >= threshold
