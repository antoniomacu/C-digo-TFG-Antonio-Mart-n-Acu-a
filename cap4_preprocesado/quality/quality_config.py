from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pipeline.system_config import SystemConfig, load_config


@dataclass(frozen=True)
class SensorQualityConfig:
    min_value: float | int | None
    max_value: float | int | None
    check_constant: bool
    check_falldown: bool


@dataclass(frozen=True)
class CleaningConfig:
    codes_to_nan: list[int]
    drop_quality_columns: bool


@dataclass(frozen=True)
class ResampleConfig:
    target_interval: str
    value_agg: str
    quality_agg: str


@dataclass(frozen=True)
class QualityConfig:
    timestep_seconds: int
    speed_on_threshold: float
    min_consecutive_operational: int
    sensors: dict[str, SensorQualityConfig]
    frozen_tolerance: float
    frozen_period_seconds: int
    variability_threshold: float
    rolling_window_samples: int
    rolling_min_periods: int
    sentinel_value: float | None
    sentinel_column_threshold: int | None
    cleaning: CleaningConfig
    resample: ResampleConfig | None
    output_dir: str
    dashboard_output: str
    skip_all_frozen_files: bool = True
    operational_filter: bool = True
    split_operational_periods: bool = False
    summary_output: str = "quality_summary.txt"

    def get_sensor_columns(self) -> list[str]:
        return list(self.sensors.keys())


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{field_name}' must be a mapping")
    return value


def _require_key(mapping: dict[str, Any], key: str, section: str = "root") -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required field '{key}' in section '{section}'")
    return mapping[key]


def _build_sensor_config(
    sensors_raw: dict[str, Any],
    system_config: SystemConfig,
) -> dict[str, SensorQualityConfig]:
    valid_sensor_names = set(system_config.columns.clean_names)
    sensors: dict[str, SensorQualityConfig] = {}

    for sensor_name, sensor_raw in sensors_raw.items():
        if sensor_name not in valid_sensor_names:
            raise ValueError(
                f"Sensor '{sensor_name}' is not present in columns.clean_names"
            )

        sensor_data = _require_mapping(sensor_raw, f"quality.sensors.{sensor_name}")
        _require_key(sensor_data, "min_value", f"quality.sensors.{sensor_name}")
        _require_key(sensor_data, "max_value", f"quality.sensors.{sensor_name}")

        sensors[sensor_name] = SensorQualityConfig(
            min_value=sensor_data.get("min_value"),
            max_value=sensor_data.get("max_value"),
            check_constant=bool(sensor_data.get("check_constant", False)),
            check_falldown=bool(sensor_data.get("check_falldown", False)),
        )

    if not sensors:
        raise ValueError("'quality.sensors' must define at least one sensor")

    return sensors


def _build_cleaning_config(raw: Any) -> CleaningConfig:
    data = _require_mapping(raw, "quality.cleaning")
    return CleaningConfig(
        codes_to_nan=[int(code) for code in list(data.get("codes_to_nan", []))],
        drop_quality_columns=bool(data.get("drop_quality_columns", True)),
    )


def _build_resample_config(raw: Any) -> ResampleConfig | None:
    if raw is None:
        return None

    data = _require_mapping(raw, "quality.resample")
    _require_key(data, "target_interval", "quality.resample")
    _require_key(data, "value_agg", "quality.resample")
    _require_key(data, "quality_agg", "quality.resample")

    return ResampleConfig(
        target_interval=str(data["target_interval"]),
        value_agg=str(data["value_agg"]),
        quality_agg=str(data["quality_agg"]),
    )


def load_quality_config(yaml_path: str | Path) -> QualityConfig:
    """Load quality configuration from the 'quality' section of a system YAML."""

    resolved_path = Path(yaml_path).expanduser().resolve()
    system_config = load_config(resolved_path)

    with resolved_path.open("r", encoding="utf-8") as file_handle:
        raw = yaml.safe_load(file_handle)

    if raw is None:
        raise ValueError("Configuration YAML is empty")

    root = _require_mapping(raw, "root")
    quality_raw = _require_mapping(_require_key(root, "quality", "root"), "quality")

    required_quality_fields = [
        "timestep_seconds",
        "speed_on_threshold",
        "min_consecutive_operational",
        "sensors",
        "frozen_tolerance",
        "frozen_period_seconds",
        "variability_threshold",
        "rolling_window_samples",
        "rolling_min_periods",
        "cleaning",
        "output_dir",
        "dashboard_output",
    ]
    for field_name in required_quality_fields:
        _require_key(quality_raw, field_name, "quality")

    sensors_raw = _require_mapping(quality_raw["sensors"], "quality.sensors")

    return QualityConfig(
        timestep_seconds=int(quality_raw["timestep_seconds"]),
        speed_on_threshold=float(quality_raw["speed_on_threshold"]),
        min_consecutive_operational=int(quality_raw["min_consecutive_operational"]),
        sensors=_build_sensor_config(sensors_raw, system_config),
        frozen_tolerance=float(quality_raw["frozen_tolerance"]),
        frozen_period_seconds=int(quality_raw["frozen_period_seconds"]),
        variability_threshold=float(quality_raw["variability_threshold"]),
        rolling_window_samples=int(quality_raw["rolling_window_samples"]),
        rolling_min_periods=int(quality_raw["rolling_min_periods"]),
        sentinel_value=(
            None
            if quality_raw.get("sentinel_value") is None
            else float(quality_raw.get("sentinel_value"))
        ),
        sentinel_column_threshold=(
            None
            if quality_raw.get("sentinel_column_threshold") is None
            else int(quality_raw.get("sentinel_column_threshold"))
        ),
        cleaning=_build_cleaning_config(quality_raw["cleaning"]),
        resample=_build_resample_config(quality_raw.get("resample")),
        output_dir=str(quality_raw["output_dir"]),
        dashboard_output=str(quality_raw["dashboard_output"]),
        skip_all_frozen_files=bool(quality_raw.get("skip_all_frozen_files", True)),
        operational_filter=bool(quality_raw.get("operational_filter", True)),
        split_operational_periods=bool(
            quality_raw.get("split_operational_periods", False)
        ),
        summary_output=str(quality_raw.get("summary_output", "quality_summary.txt")),
    )


def load_quality_and_system_config(
    yaml_path: str | Path,
) -> tuple[QualityConfig, SystemConfig]:
    """Load both quality and system configs from the same YAML file."""

    quality_config = load_quality_config(yaml_path)
    system_config = load_config(yaml_path)
    return quality_config, system_config