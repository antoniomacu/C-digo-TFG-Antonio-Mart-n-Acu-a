"""Unit tests for Savitzky-Golay filter parity between training and streaming paths.

Training applies SG per pump-day CSV file independently (Preprocessor._load_files).
Streaming must do the same — apply SG per-day segment, not across day boundaries.
"""
import numpy as np
import pandas as pd
import pytest
from argparse import Namespace

from bin.model.preprocessing import Preprocessor


def _make_two_day_df(rows_per_day: int = 20) -> pd.DataFrame:
    """Build a synthetic 2-day sensor DataFrame with a step at the boundary.

    The step discontinuity at the day boundary makes cross-day vs per-day SG
    filtering produce visibly different results near the boundary.
    """
    day1_index = pd.date_range("2024-06-01 08:00", periods=rows_per_day, freq="10min")
    day2_index = pd.date_range("2024-06-02 08:00", periods=rows_per_day, freq="10min")
    index = pd.DatetimeIndex(day1_index.tolist() + day2_index.tolist())

    rng = np.random.default_rng(42)
    n = len(index)

    data = {
        "sensor_a": np.concatenate([
            10.0 + rng.normal(0, 0.3, rows_per_day),   # day 1 baseline
            20.0 + rng.normal(0, 0.3, rows_per_day),   # day 2 step up
        ]),
        "sensor_b": np.concatenate([
            5.0 + rng.normal(0, 0.1, rows_per_day),
            5.0 + rng.normal(0, 0.1, rows_per_day),    # no step (control)
        ]),
        "pump_id_1": np.ones(n, dtype=int),
        "pump_id_2": np.zeros(n, dtype=int),
        "pump_id_3": np.zeros(n, dtype=int),
        "pump_id_4": np.zeros(n, dtype=int),
    }
    return pd.DataFrame(data, index=index)


def _apply_sg_per_day(preprocessor: Preprocessor, df: pd.DataFrame) -> pd.DataFrame:
    """Apply SG per-day segment — the streaming fix we are testing.

    Days with < 5 rows are skipped entirely, matching training's behavior
    (Preprocessor._load_files skips short files with ``continue``).
    """
    segments = []
    for _, day_df in df.groupby(df.index.date):
        if len(day_df) >= 5:
            segments.append(preprocessor.filter_savitzky_golay(day_df))
        # else: skip — training never sees short-day files
    if not segments:
        return df.iloc[:0]  # empty DataFrame with same columns
    return pd.concat(segments).sort_index()


@pytest.fixture
def preprocessor() -> Preprocessor:
    """Minimal Preprocessor instance (only SG filter method used)."""
    params = Namespace(
        norm_method="zscore",
        train_path=".",
        test_path=".",
        past_history=1,
    )
    return Preprocessor(params)


class TestSavgolParity:
    """Verify streaming per-day SG matches training per-file SG."""

    def test_per_day_sg_matches_training(self, preprocessor: Preprocessor):
        """Per-day SG (streaming fix) must produce identical output to
        training's per-file SG application."""
        df = _make_two_day_df()

        # Training style: split into per-day DataFrames, apply SG to each, concat
        day1 = df[df.index.date == df.index[0].date()]
        day2 = df[df.index.date == df.index[-1].date()]
        training_result = pd.concat([
            preprocessor.filter_savitzky_golay(day1),
            preprocessor.filter_savitzky_golay(day2),
        ]).sort_index()

        # Streaming fix style: groupby date
        streaming_result = _apply_sg_per_day(preprocessor, df)

        pd.testing.assert_frame_equal(training_result, streaming_result)

    def test_whole_buffer_sg_differs_at_boundary(self, preprocessor: Preprocessor):
        """Whole-buffer SG (the old streaming bug) must differ from per-day SG
        at boundary samples, confirming the bug was real."""
        df = _make_two_day_df()

        whole_buffer = preprocessor.filter_savitzky_golay(df)
        per_day = _apply_sg_per_day(preprocessor, df)

        # sensor_a has a 10→20 step at the boundary; SG with window=5 affects
        # the 2 samples adjacent to the boundary on each side
        boundary_cols = ["sensor_a"]
        boundary_rows = list(range(18, 22))  # last 2 of day1, first 2 of day2
        diff = (whole_buffer.iloc[boundary_rows][boundary_cols]
                - per_day.iloc[boundary_rows][boundary_cols]).abs()
        assert diff.max().max() > 0.01, (
            "Expected whole-buffer SG to differ from per-day SG at boundary, "
            "but they were nearly identical — the test fixture may not expose the bug."
        )

    def test_short_day_segment_skipped(self, preprocessor: Preprocessor):
        """A day segment with < 5 rows should be skipped entirely, matching
        training which skips short files via ``continue``."""
        # 3 rows for day 1 (too short), 20 rows for day 2
        short_index = pd.date_range("2024-06-01 08:00", periods=3, freq="10min")
        long_index = pd.date_range("2024-06-02 08:00", periods=20, freq="10min")
        index = pd.DatetimeIndex(short_index.tolist() + long_index.tolist())

        rng = np.random.default_rng(99)
        data = {
            "sensor_a": rng.normal(10.0, 0.5, len(index)),
            "pump_id_1": np.ones(len(index), dtype=int),
            "pump_id_2": np.zeros(len(index), dtype=int),
            "pump_id_3": np.zeros(len(index), dtype=int),
            "pump_id_4": np.zeros(len(index), dtype=int),
        }
        df = pd.DataFrame(data, index=index)

        result = _apply_sg_per_day(preprocessor, df)

        # Short day segment should be absent from result
        assert len(result) == 20, (
            f"Expected only the 20-row day segment, got {len(result)} rows"
        )
        assert all(d.day == 2 for d in result.index.date), (
            "Only day 2 should remain — day 1 (3 rows) should be skipped"
        )
        # Long day segment: SG applied (values should differ from raw)
        long_diff = (result["sensor_a"] - df.iloc[3:]["sensor_a"]).abs()
        assert long_diff.max() > 1e-10, "SG filter should modify the long segment"
