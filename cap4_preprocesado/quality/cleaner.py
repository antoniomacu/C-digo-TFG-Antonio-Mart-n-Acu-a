from __future__ import annotations

import numpy as np
import pandas as pd

from quality.labeler import detect_sentinel_rows
from quality.quality_config import QualityConfig


def clean_dataframe(
    df: pd.DataFrame,
    quality_config: QualityConfig,
    sensor_columns: list[str],
) -> pd.DataFrame:
    out = df.copy()

    if quality_config.sentinel_value is not None:
        sentinel_mask = detect_sentinel_rows(
            out,
            sensor_columns,
            quality_config.sentinel_value,
            int(quality_config.sentinel_column_threshold or 0),
        )
        sentinel_mask = np.asarray(sentinel_mask, dtype=bool)
        sentinel_count = int(np.count_nonzero(sentinel_mask))
        if sentinel_count > 0:
            existing_sensor_columns = [
                column for column in sensor_columns if column in out.columns
            ]
            if existing_sensor_columns:
                out.loc[sentinel_mask, existing_sensor_columns] = np.nan
        out.attrs["sentinel_rows"] = sentinel_count

    codes_to_nan = set(quality_config.cleaning.codes_to_nan)
    if codes_to_nan:
        for sensor_column in sensor_columns:
            quality_column = f"{sensor_column}_quality"
            if sensor_column not in out.columns or quality_column not in out.columns:
                continue

            quality_values = pd.to_numeric(out[quality_column], errors="coerce")
            bad_quality_mask = quality_values.isin(codes_to_nan)
            if bad_quality_mask.any():
                out.loc[bad_quality_mask, sensor_column] = np.nan

    if quality_config.cleaning.drop_quality_columns:
        quality_columns = [
            column for column in out.columns if column.endswith("_quality")
        ]
        if quality_columns:
            out = out.drop(columns=quality_columns)

    return out
