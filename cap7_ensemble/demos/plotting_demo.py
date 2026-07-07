"""Single-pump animated plotting demo for streaming ensemble outputs.

Usage:
    uv run python demos/plotting_demo.py [--pump 1] [--source any] [--speed 500]
"""

from __future__ import annotations

import sys
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# Ensure local imports resolve when running from ensemble/demos.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.streaming import StreamingTimestepResult, create_streaming_detector
from demos.threshold_loader import load_calibrated_thresholds, build_detector_overrides, print_threshold_source
from demos.streaming_demo import (
    SHORT_NAMES,
    _extract_day_date,
    _load_day,
    _parse_pump_from_filename,
    _resolve_data_dir,
)

SENSOR_GROUPS: list[tuple[str, list[str]]] = [
    (
        "Electrical & Pressure",
        [
            "Main HTF Pump Current Consumption",
            "Main HTF Pump Flow",
            "Main HTF Pump Outlet Pressure",
        ],
    ),
    (
        "Bearings",
        [
            "Main HTF Pump NDE Outboard bearing",
            "Main HTF Pump NDE Inboard bearing",
            "Main HTF Pump DE bearing",
        ],
    ),
    (
        "Motor & Vibration",
        [
            "Main HTF Pump Motor bearing Temp 1",
            "Main HTF Pump Motor bearing Temp 2",
            "Main HTF Pump DE Side Bearing vibration",
            "Main HTF Pump NDE Side Bearing vibration",
        ],
    ),
    (
        "Winding Temps",
        [
            "Main HTF Pump Motor U winding Temp 1",
            "Main HTF Pump Motor U winding Temp 2",
            "Main HTF Pump Motor U winding Temp 3",
        ],
    ),
]

STATUS_COLOR = {
    "NORMAL": "green",
    "WARNING": "yellow",
    "ALARM": "red",
    "BUFFERING": "cyan",
}


class PlotRuntime:
    def __init__(self, sensors: list[str]) -> None:
        self.x: list[int] = []
        self.actual: dict[str, list[float]] = {sensor: [] for sensor in sensors}
        self.predicted: dict[str, list[float]] = {sensor: [] for sensor in sensors}
        self.in_anomaly: dict[str, bool] = {sensor: False for sensor in sensors}
        self.anomaly_start: dict[str, int | None] = {sensor: None for sensor in sensors}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animated single-pump plotting demo")
    parser.add_argument("--pump", type=int, default=1, help="Pump ID to visualize")
    parser.add_argument("--csv", type=str, default=None, help="Specific CSV file path")
    parser.add_argument("--data-dir", type=str, default="../../data/inference_data", help="Inference data root")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for day selection")
    parser.add_argument(
        "--source",
        type=str,
        choices=["normal", "abnormal", "any"],
        default="any",
        help="Pick from normal, abnormal, or either",
    )
    parser.add_argument("--speed", type=int, default=500, help="Milliseconds between animation frames")
    return parser.parse_args()


def _list_available_sources(data_dir: Path, pump_id: int) -> None:
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    normal_days = 0
    abnormal_days = 0
    pumps_seen: set[int] = set()

    for source_dir, label in ((train_dir, "normal"), (test_dir, "abnormal")):
        if not source_dir.exists():
            continue
        for csv_path in source_dir.glob("*.csv"):
            parsed = _parse_pump_from_filename(csv_path)
            if parsed is None:
                continue
            pumps_seen.add(parsed)
            if parsed != pump_id:
                continue
            if label == "normal":
                normal_days += 1
            else:
                abnormal_days += 1

    pump_values = sorted(pumps_seen)
    pump_text = ", ".join(str(value) for value in pump_values) if pump_values else "none"

    print(f"No CSV candidates found for pump={pump_id} with the requested source filter.")
    print(f"Available pumps in {data_dir}: {pump_text}")
    print(f"Pump {pump_id} has normal(train)={normal_days}, abnormal(test)={abnormal_days}")


def _choose_csv(args: argparse.Namespace) -> tuple[Path, int, str]:
    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        parsed_pump = _parse_pump_from_filename(csv_path)
        if parsed_pump is not None and parsed_pump != args.pump:
            print(
                f"Warning: --pump={args.pump} disagrees with filename pump={parsed_pump}. "
                "Using filename pump id."
            )
        pump_id = parsed_pump if parsed_pump is not None else int(args.pump)

        source_name = "any"
        parent_name = csv_path.parent.name.lower()
        if parent_name == "train":
            source_name = "normal"
        elif parent_name == "test":
            source_name = "abnormal"

        return csv_path, pump_id, source_name

    data_dir = _resolve_data_dir(args.data_dir)

    candidates: list[tuple[Path, str]] = []

    if args.source in {"normal", "any"}:
        train_dir = data_dir / "train"
        if train_dir.exists():
            for csv_path in sorted(train_dir.glob("*.csv")):
                if _parse_pump_from_filename(csv_path) == args.pump:
                    candidates.append((csv_path, "normal"))

    if args.source in {"abnormal", "any"}:
        test_dir = data_dir / "test"
        if test_dir.exists():
            for csv_path in sorted(test_dir.glob("*.csv")):
                if _parse_pump_from_filename(csv_path) == args.pump:
                    candidates.append((csv_path, "abnormal"))

    if not candidates:
        _list_available_sources(data_dir=data_dir, pump_id=args.pump)
        raise SystemExit(1)

    rng = random.Random(args.seed)
    selected_path, selected_source = rng.choice(candidates)
    return selected_path, int(args.pump), selected_source


def _set_axis_limits(ax: plt.Axes, x_values: list[int], y_actual: list[float], y_predicted: list[float]) -> None:
    max_x = max(x_values) if x_values else 1
    ax.set_xlim(0, max(1, max_x))

    y_values = np.array(y_actual + y_predicted, dtype=float)
    finite = y_values[np.isfinite(y_values)]
    if finite.size == 0:
        return

    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if np.isclose(y_min, y_max):
        pad = max(abs(y_min) * 0.05, 1e-6)
    else:
        pad = (y_max - y_min) * 0.1

    ax.set_ylim(y_min - pad, y_max + pad)


def main() -> int:
    args = _parse_args()

    csv_path, pump_id, source_name = _choose_csv(args)
    day_label = _extract_day_date(csv_path) or "unknown-day"

    df = _load_day(csv_path)
    if len(df) < 5:
        print(f"Error: day has too few rows after filtering/resampling ({len(df)} rows).")
        return 1

    l1_cal, l2_cal = load_calibrated_thresholds()
    detector = create_streaming_detector(**build_detector_overrides(l1_cal, l2_cal))
    print_threshold_source(l1_cal, l2_cal)
    detector.reset_pump(pump_id)

    sensors = [sensor for _, group_sensors in SENSOR_GROUPS for sensor in group_sensors]
    runtime = PlotRuntime(sensors=sensors)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(nrows=4, ncols=4, figure=fig)

    axes_by_sensor: dict[str, plt.Axes] = {}
    actual_lines: dict[str, any] = {}
    predicted_lines: dict[str, any] = {}

    for row_idx, (group_name, group_sensors) in enumerate(SENSOR_GROUPS):
        row_axes: list[plt.Axes] = []
        for col_idx, sensor in enumerate(group_sensors):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            row_axes.append(ax)
            axes_by_sensor[sensor] = ax

            line_actual, = ax.plot([], [], color="tab:blue", linewidth=2.0, label="Actual")
            line_pred, = ax.plot([], [], color="tab:orange", linestyle="--", linewidth=1.8, label="Predicted")
            actual_lines[sensor] = line_actual
            predicted_lines[sensor] = line_pred

            ax.set_title(SHORT_NAMES.get(sensor, sensor), fontsize=10)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Value")

            if col_idx == 0:
                ax.legend(loc="upper right", fontsize=8)

        if row_axes:
            y_min = min(ax.get_position().ymin for ax in row_axes)
            y_max = max(ax.get_position().ymax for ax in row_axes)
            y_center = (y_min + y_max) / 2.0
            fig.text(0.015, y_center, group_name, va="center", ha="left", fontsize=11, fontweight="bold")

    total = len(df)
    metadata_title = f"Source: {source_name} | File: {csv_path.name}"
    header = fig.suptitle(
        f"Pump {pump_id} - {day_label} - Step 0/{total}\n"
        "Ensemble: NORMAL | Health: 100.0\n"
        "L1: NORMAL | Mahalanobis: 0.00  |  L2: BUFFERING | Smoothed MSE: -\n"
        f"{metadata_title}",
        fontsize=13,
    )

    def init() -> list:
        runtime.x.clear()
        for sensor in sensors:
            runtime.actual[sensor].clear()
            runtime.predicted[sensor].clear()
            runtime.in_anomaly[sensor] = False
            runtime.anomaly_start[sensor] = None
            actual_lines[sensor].set_data([], [])
            predicted_lines[sensor].set_data([], [])
        return []

    def update(frame_idx: int) -> list:
        row = df.iloc[frame_idx].drop(labels=["timestamp"], errors="ignore")
        timestamp = df.index[frame_idx]
        result: StreamingTimestepResult = detector.process_timestep(pump_id=pump_id, timestamp=timestamp, row=row)

        runtime.x.append(frame_idx)

        for sensor in sensors:
            actual_value = result.l1_actual.get(sensor, np.nan)
            predicted_value = result.l1_predicted.get(sensor, np.nan)

            runtime.actual[sensor].append(float(actual_value) if pd.notna(actual_value) else np.nan)
            runtime.predicted[sensor].append(float(predicted_value) if pd.notna(predicted_value) else np.nan)

            actual_lines[sensor].set_data(runtime.x, runtime.actual[sensor])
            predicted_lines[sensor].set_data(runtime.x, runtime.predicted[sensor])

            ax = axes_by_sensor[sensor]
            _set_axis_limits(ax, runtime.x, runtime.actual[sensor], runtime.predicted[sensor])

            z_score = float(result.l1_z_scores.get(sensor, np.nan))
            is_anomalous = bool(np.isfinite(z_score) and abs(z_score) > 3.0)

            if is_anomalous and not runtime.in_anomaly[sensor]:
                runtime.in_anomaly[sensor] = True
                runtime.anomaly_start[sensor] = frame_idx

            if not is_anomalous and runtime.in_anomaly[sensor]:
                start = runtime.anomaly_start[sensor]
                if start is not None:
                    ax.axvspan(start, frame_idx, color="red", alpha=0.15, zorder=0)
                runtime.in_anomaly[sensor] = False
                runtime.anomaly_start[sensor] = None

            if frame_idx == total - 1 and runtime.in_anomaly[sensor]:
                start = runtime.anomaly_start[sensor]
                if start is not None:
                    ax.axvspan(start, frame_idx, color="red", alpha=0.15, zorder=0)
                runtime.in_anomaly[sensor] = False
                runtime.anomaly_start[sensor] = None

        l2_text = "BUFFERING" if result.l2_status == "BUFFERING" else result.l2_status
        l2_mse = "-" if result.l2_smoothed_mse is None else f"{result.l2_smoothed_mse:.4f}"
        header.set_text(
            f"Pump {pump_id} - {day_label} - Step {frame_idx + 1}/{total}\n"
            f"Ensemble: {result.ensemble_status} | Health: {result.ensemble_health:.1f}\n"
            f"L1: {result.l1_status} | Mahalanobis: {result.l1_mahalanobis:.2f}  |  "
            f"L2: {l2_text} | Smoothed MSE: {l2_mse}\n"
            f"{metadata_title}"
        )

        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=len(df),
        init_func=init,
        interval=args.speed,
        blit=False,
        repeat=False,
    )
    fig._anim = anim

    plt.tight_layout(rect=[0.04, 0, 1, 0.92])
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
