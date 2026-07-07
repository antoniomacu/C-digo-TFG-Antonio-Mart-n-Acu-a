from __future__ import annotations

import pandas as pd

from quality.quality_config import QualityConfig


def resample_dataframe(
    df: pd.DataFrame,
    quality_config: QualityConfig,
    sensor_columns: list[str],
) -> pd.DataFrame:
    """Resample a time-series dataframe using quality configuration rules."""

    if df.empty:
        return df

    resample_config = quality_config.resample
    if resample_config is None:
        return df

    value_cols = [column for column in sensor_columns if column in df.columns]
    quality_cols = [
        column for column in df.columns if str(column).endswith("_quality")
    ]

    used_cols = set(value_cols) | set(quality_cols)
    other_cols = [column for column in df.columns if column not in used_cols]

    target_interval = resample_config.target_interval
    parts: list[pd.DataFrame] = []

    if value_cols:
        values_resampled = df[value_cols].resample(
            target_interval,
            closed="left",
            label="left",
        ).agg(resample_config.value_agg)
        parts.append(values_resampled)

    if quality_cols:
        quality_resampled = df[quality_cols].resample(
            target_interval,
            closed="left",
            label="left",
        ).agg(resample_config.quality_agg)
        parts.append(quality_resampled)

    if other_cols:
        other_resampled = df[other_cols].resample(
            target_interval,
            closed="left",
            label="left",
        ).agg("first")
        parts.append(other_resampled)

    if not parts:
        # Keep a resampled index even when there are no data columns to aggregate.
        resampled_index = df.resample(
            target_interval,
            closed="left",
            label="left",
        ).size().index
        return pd.DataFrame(index=resampled_index)

    result = pd.concat(parts, axis=1)

    ordered_columns = [column for column in df.columns if column in result.columns]
    return result[ordered_columns]
