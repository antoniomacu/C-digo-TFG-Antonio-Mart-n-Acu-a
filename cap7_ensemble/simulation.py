"""Real-time simulation engine for ensemble pump anomaly detection.

Replays one full day timestep-by-timestep and runs:
- Level 1 (digital twin residual + Mahalanobis) from first timestep
- Level 2 (temporal model) once the rolling buffer is available
- Ensemble fusion of both levels
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure sibling packages (cond_reg, bin) are importable when running this file directly.
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from model.level1_detector import Level1Detector
from model.streaming import StreamingTimestepResult, create_streaming_detector


SHORT_NAMES = {
    "Main HTF Pump Current Consumption": "Current",
    "Main HTF Pump Flow": "Flow",
    "Main HTF Pump Outlet Pressure": "Pressure",
    "Main HTF Pump NDE Outboard bearing": "NDE Outboard",
    "Main HTF Pump NDE Inboard bearing": "NDE Inboard",
    "Main HTF Pump DE bearing": "DE Bearing",
    "Main HTF Pump Motor bearing Temp 1": "Motor Bearing T1",
    "Main HTF Pump Motor bearing Temp 2": "Motor Bearing T2",
    "Main HTF Pump Motor U winding Temp 1": "Winding T1",
    "Main HTF Pump Motor U winding Temp 2": "Winding T2",
    "Main HTF Pump Motor U winding Temp 3": "Winding T3",
    "Main HTF Pump DE Side Bearing vibration": "DE Vibration",
    "Main HTF Pump NDE Side Bearing vibration": "NDE Vibration",
}

@dataclass
class SimulationFrame:
    """All values collected at one simulation timestep."""

    timestep: int
    timestamp: str
    l1_mahalanobis: float
    l1_status: str
    l1_health: float
    l1_z_scores: dict
    l1_actual: dict
    l1_predicted: dict
    l1_residuals: dict
    l2_raw_mse: float | None = None
    l2_smoothed_mse: float | None = None
    l2_status: str | None = None
    l2_health: float | None = None
    ensemble_status: str = "NORMAL"
    ensemble_health: float = 100.0


@dataclass
class SimulationResult:
    """Complete simulation output for one pump-day."""

    pump_id: int
    date: str
    source: str
    csv_path: str
    frames: list[SimulationFrame] = field(default_factory=list)
    l1_warning_threshold: float = 0.0
    l1_alarm_threshold: float = 0.0
    l1_mean: float = 0.0
    l2_warning_threshold: float = 0.0
    l2_alarm_threshold: float = 0.0
    l2_mean: float = 0.0
    l2_window_warning: float = 0.0
    l2_window_alarm: float = 0.0
    l2_buffer_size: int = 36
    sensor_names: list[str] = field(default_factory=list)


def _classify_l2_status(smoothed_mse: float, warning_threshold: float, alarm_threshold: float) -> str:
    if smoothed_mse >= alarm_threshold:
        return "ALARM"
    if smoothed_mse >= warning_threshold:
        return "WARNING"
    return "NORMAL"


def _find_nearest_l2_index(
    target_ts: pd.Timestamp,
    l2_index: pd.DatetimeIndex,
    tolerance_minutes: int = 3,
) -> int | None:
    """Find nearest index in Level 2 preprocessed timeline within tolerance."""
    if l2_index.empty:
        return None
    idx = l2_index.get_indexer(
        [pd.Timestamp(target_ts)],
        method="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )[0]
    if idx == -1:
        return None
    return int(idx)


def _get_l1_thresholds_for_pump(pump_id: int, l1_thresholds: dict) -> tuple[float, float, float]:
    global_block = l1_thresholds.get("global", {})
    block = l1_thresholds.get("per_pump", {}).get(str(pump_id), global_block)
    maha = block.get("mahalanobis", global_block.get("mahalanobis", {}))

    warning = float(maha.get("warning", global_block.get("mahalanobis", {}).get("warning", 1.0)))
    alarm = float(maha.get("alarm", global_block.get("mahalanobis", {}).get("alarm", warning)))
    if alarm < warning:
        alarm = warning
    mean = float(maha.get("mean", global_block.get("mahalanobis", {}).get("mean", warning)))
    return warning, alarm, mean


def _get_l2_thresholds_for_pump(pump_id: int, l2_thresholds: dict) -> tuple[float, float, float, float, float, float]:
    global_block = l2_thresholds.get("global", {})
    block = l2_thresholds.get("per_pump", {}).get(str(pump_id), global_block)

    warning = float(block.get("warning", global_block.get("warning", 1.0)))
    alarm = float(block.get("alarm", global_block.get("alarm", warning)))
    if alarm < warning:
        alarm = warning

    mean = float(block.get("mean", global_block.get("mean", warning)))
    win_warning = float(block.get("window_warning_smoothed", warning))
    win_alarm = float(block.get("window_alarm_smoothed", alarm))
    if win_alarm < win_warning:
        win_alarm = win_warning
    alpha = float(block.get("smoothing_alpha", global_block.get("smoothing_alpha", 0.3)))
    return warning, alarm, mean, win_warning, win_alarm, alpha


def _resample_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 5-min grid when the source cadence is denser than 3 minutes."""
    if len(df.index) < 3:
        return df.sort_index()

    diffs = pd.Series(df.index).sort_values().diff().dropna().dt.total_seconds() / 60.0
    if diffs.empty or float(diffs.median()) >= 3.0:
        return df.sort_index()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    resampled = df[numeric_cols].resample("5min").mean()

    if "pump_id" in resampled.columns:
        resampled["pump_id"] = resampled["pump_id"].round().astype("Int64")
        resampled["pump_id"] = resampled["pump_id"].ffill().bfill().astype(int)

    return resampled.dropna().sort_index()


def _filter_stable_operation(df: pd.DataFrame) -> pd.DataFrame:
    """Trim startup/shutdown transients using daily pump speed envelope."""
    speed_col = "Main HTF Pump Speed"
    if speed_col not in df.columns:
        return df

    speed_series = pd.to_numeric(df[speed_col], errors="coerce")
    peak_speed = float(speed_series.max()) if not speed_series.empty else float("nan")
    if not (np.isfinite(peak_speed) and peak_speed > 0.0):
        return df

    stable_threshold = peak_speed * 0.90
    stable_mask = (speed_series >= stable_threshold).fillna(False)
    stable_indices = df.index[stable_mask]
    if len(stable_indices) < 10:
        return df

    return df.loc[stable_indices[0] : stable_indices[-1]]


def run_simulation(
    csv_path: str,
    source_label: str,
    l1_detector: Level1Detector,
    l2_model,
    l2_preprocessor,
    l2_norm_params: dict,
    l2_hparams: dict,
    l1_thresholds: dict,
    l2_thresholds: dict,
) -> SimulationResult:
    """Replay one CSV as a production-like real-time stream."""
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"CSV missing timestamp column: {csv_path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if df.empty:
        raise ValueError(f"CSV has no valid rows after timestamp parsing: {csv_path}")

    if "pump_id" not in df.columns:
        raise ValueError(f"CSV missing pump_id column: {csv_path}")

    df = _resample_if_needed(df)
    if df.empty:
        raise ValueError(f"CSV has no valid rows after optional resampling: {csv_path}")

    df = _filter_stable_operation(df)

    if df.empty:
        raise ValueError(f"CSV has no valid rows after transient trimming: {csv_path}")

    pump_id = int(pd.to_numeric(df["pump_id"], errors="coerce").dropna().iloc[0])
    date = str(df.index[0].date())

    l1_warning, l1_alarm, l1_mean = _get_l1_thresholds_for_pump(pump_id, l1_thresholds)
    (
        l2_warning,
        l2_alarm,
        l2_mean,
        l2_window_warning,
        l2_window_alarm,
        smoothing_alpha,
    ) = _get_l2_thresholds_for_pump(pump_id, l2_thresholds)

    streaming_detector = create_streaming_detector(
        l1_detector=l1_detector,
        l2_model=l2_model,
        l2_preprocessor=l2_preprocessor,
        l2_norm_params=l2_norm_params,
        l2_hparams=l2_hparams,
        l1_thresholds=l1_thresholds,
        l2_thresholds=l2_thresholds,
        ema_alpha=smoothing_alpha,
    )
    streaming_detector.reset_pump(pump_id)

    result = SimulationResult(
        pump_id=pump_id,
        date=date,
        source=source_label,
        csv_path=str(csv_path),
        l1_warning_threshold=l1_warning,
        l1_alarm_threshold=l1_alarm,
        l1_mean=l1_mean,
        l2_warning_threshold=l2_warning,
        l2_alarm_threshold=l2_alarm,
        l2_mean=l2_mean,
        l2_window_warning=l2_window_warning,
        l2_window_alarm=l2_window_alarm,
        l2_buffer_size=int(l2_hparams.get("past_history", 36)),
        sensor_names=list(l1_detector._output_columns),
    )

    for t in range(len(df)):
        row = df.iloc[t]
        ts_result: StreamingTimestepResult = streaming_detector.process_timestep(
            pump_id=pump_id,
            timestamp=df.index[t],
            row=row,
        )

        if np.isnan(ts_result.l1_mahalanobis):
            continue

        l2_status = None if ts_result.l2_status == "BUFFERING" else ts_result.l2_status

        result.frames.append(
            SimulationFrame(
                timestep=ts_result.timestep,
                timestamp=ts_result.timestamp,
                l1_mahalanobis=float(ts_result.l1_mahalanobis),
                l1_status=ts_result.l1_status,
                l1_health=float(ts_result.l1_health),
                l1_z_scores=ts_result.l1_z_scores,
                l1_actual=ts_result.l1_actual,
                l1_predicted=ts_result.l1_predicted,
                l1_residuals=ts_result.l1_residuals,
                l2_raw_mse=ts_result.l2_raw_mse,
                l2_smoothed_mse=ts_result.l2_smoothed_mse,
                l2_status=l2_status,
                l2_health=ts_result.l2_health,
                ensemble_status=ts_result.ensemble_status,
                ensemble_health=float(ts_result.ensemble_health),
            )
        )

    return result



def load_models():
    """Load Level 1 and Level 2 production resources."""
    detector = create_streaming_detector()

    return (
        detector._l1_detector,
        detector._l1_thresholds,
        detector._l2_model,
        detector._l2_preprocessor,
        detector._l2_norm_params,
        detector._l2_hparams,
        detector._l2_thresholds,
    )


def serialize_result(result: SimulationResult) -> dict:
    """Convert SimulationResult to a JSON-serializable dict optimized for dcc.Store.

    Uses columnar layout (arrays per field) instead of per-frame dicts to minimize payload.
    """
    n = len(result.frames)
    sensor_short = [SHORT_NAMES.get(s, s) for s in result.sensor_names]

    # Pre-allocate columnar arrays
    data = {
        "pump_id": result.pump_id,
        "date": result.date,
        "source": result.source,
        "csv_path": result.csv_path,
        "n_frames": n,
        "l2_buffer_size": result.l2_buffer_size,
        "thresholds": {
            "l1_warning": result.l1_warning_threshold,
            "l1_alarm": result.l1_alarm_threshold,
            "l1_mean": result.l1_mean,
            "l2_warning": result.l2_warning_threshold,
            "l2_alarm": result.l2_alarm_threshold,
            "l2_mean": result.l2_mean,
            "l2_window_warning": result.l2_window_warning,
            "l2_window_alarm": result.l2_window_alarm,
        },
        "sensor_names": result.sensor_names,
        "sensor_short": sensor_short,
        "timestamps": [],
        "l1_mahalanobis": [],
        "l1_status": [],
        "l1_health": [],
        "l2_raw_mse": [],
        "l2_smoothed_mse": [],
        "l2_status": [],
        "l2_health": [],
        "ensemble_status": [],
        "ensemble_health": [],
        "z_scores": {s: [] for s in sensor_short},
        "actuals": {s: [] for s in sensor_short},
        "predicted": {s: [] for s in sensor_short},
    }

    for frame in result.frames:
        data["timestamps"].append(frame.timestamp)
        data["l1_mahalanobis"].append(frame.l1_mahalanobis)
        data["l1_status"].append(frame.l1_status)
        data["l1_health"].append(frame.l1_health)
        data["l2_raw_mse"].append(frame.l2_raw_mse)
        data["l2_smoothed_mse"].append(frame.l2_smoothed_mse)
        data["l2_status"].append(frame.l2_status)
        data["l2_health"].append(frame.l2_health)
        data["ensemble_status"].append(frame.ensemble_status)
        data["ensemble_health"].append(frame.ensemble_health)

        for full_name, short_name in zip(result.sensor_names, sensor_short):
            data["z_scores"][short_name].append(frame.l1_z_scores.get(full_name, 0.0))
            data["actuals"][short_name].append(frame.l1_actual.get(full_name, 0.0))
            data["predicted"][short_name].append(frame.l1_predicted.get(full_name, 0.0))

    return data


def list_available_csvs(data_root: Path | None = None) -> list[dict]:
    """Scan train/test directories for CSV files and return dropdown options.

    Returns list of dicts: {"label": "NORMAL — pump_3 2026-02-26", "value": "/abs/path.csv", "source": "normal"}
    """
    if data_root is None:
        data_root = WORKSPACE_ROOT / "data" / "inference_data"

    options = []
    for subdir, source in [("train", "normal"), ("test", "abnormal")]:
        csv_dir = data_root / subdir
        if not csv_dir.exists():
            continue
        for csv_file in sorted(csv_dir.glob("*.csv")):
            # Extract pump info from filename like "pump_3_2026-02-26_08_29-17_24.csv"
            stem = csv_file.stem
            parts = stem.split("_")
            if len(parts) >= 3:
                pump = f"pump_{parts[1]}"
                date = parts[2] if len(parts) > 2 else ""
                label = f"{source.upper()} — {pump} {date}"
            else:
                label = f"{source.upper()} — {stem}"

            options.append(
                {
                    "label": label,
                    "value": str(csv_file),
                    "source": source,
                }
            )

    return options
