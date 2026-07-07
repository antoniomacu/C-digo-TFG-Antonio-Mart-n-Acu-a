from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ColumnConfig:
    """Column mapping configuration."""

    expected_source_count: int
    exclude_indices: list[int]
    clean_names: list[str]
    timestamp_col: str  # internal name for timestamp column
    speed_col: str  # internal name for speed column
    current_col: str  # internal name for current column
    pressure_col: str | None  # optional - some systems have no pressure sensor
    temp_cols: list[str]  # internal names for temperature columns
    prox_cols: list[str]  # internal names for vibration/proximitor columns
    pressure_off_col: str | None = None  # optional OFF-state pressure column; falls back to pressure_col
    baseline_extra_cols: list[str] = field(default_factory=list)  # non-temp columns that need baseline stats
    supplementary_columns: dict[str, str] = field(default_factory=dict)  # map source header names to internal names


@dataclass(frozen=True)
class Thresholds:
    """Detection thresholds."""

    min_period_duration: float = 600.0
    speed_on_threshold: float = 100.0
    gap_merge_threshold: float = 30.0
    ramp_exclude_seconds: float = 300.0
    anomaly_sigma: float = 3.0
    vibration_spike_factor: float = 2.0
    pressure_off_absolute_threshold: float = 5.0
    pressure_off_rebound_threshold: float = 4.0
    pressure_off_extended_sigma: float = 2.5
    off_state_sigma: float = 4.0
    frozen_sensor_std_threshold: float = 0.01
    frozen_sensor_min_samples: int = 10
    outlet_pressure_variability_cap: float = 0.0
    min_detectors_for_abnormal: int = 1
    absolute_temp_limits: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelledExample:
    """A single labelled example for validation."""

    date: str  # YYYY-MM-DD
    pump: int
    expected: str  # "normal" or "abnormal"


@dataclass(frozen=True)
class DataSelectorConfig:
    """Configuration for train/test data selection."""

    min_normal_duration_minutes: float = 120.0
    max_periods_per_day: int = 2
    train_output: Path | None = None
    test_output: Path | None = None
    use_symlinks: bool = True
    exclude_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SystemConfig:
    """Complete system configuration loaded from YAML."""

    system_name: str
    data_root: Path
    output_dir: Path
    file_pattern: str  # regex with groups for pump_id and date
    date_format: str  # strptime format for date in filename
    pump_ids: list[int]
    columns: ColumnConfig
    thresholds: Thresholds
    detectors: list[str]  # ordered list of detector names to run
    custom_detectors_dir: Path | None = None
    examples: list[LabelledExample] = field(default_factory=list)
    data_selector: DataSelectorConfig = field(default_factory=DataSelectorConfig)

    @property
    def compiled_file_pattern(self) -> re.Pattern:
        return re.compile(self.file_pattern)


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{field_name}' must be a mapping")
    return value


def _require_key(mapping: dict[str, Any], key: str, section: str = "root") -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required field '{key}' in section '{section}'")
    return mapping[key]


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (base_dir / raw).resolve()


def _build_columns(columns_raw: dict[str, Any]) -> ColumnConfig:
    required_keys = [
        "expected_source_count",
        "exclude_indices",
        "clean_names",
        "timestamp_col",
        "speed_col",
        "current_col",
        "temp_cols",
        "prox_cols",
    ]
    for key in required_keys:
        _require_key(columns_raw, key, "columns")

    pressure_col = (
        None
        if columns_raw.get("pressure_col") is None
        else str(columns_raw.get("pressure_col"))
    )
    pressure_off_col_raw = columns_raw.get("pressure_off_col")
    pressure_off_col = (
        pressure_col if pressure_off_col_raw is None else str(pressure_off_col_raw)
    )
    supplementary_columns_raw = _require_mapping(
        columns_raw.get("supplementary_columns", {}),
        "columns.supplementary_columns",
    )
    supplementary_columns = {
        str(source_name): str(clean_name)
        for source_name, clean_name in supplementary_columns_raw.items()
    }

    return ColumnConfig(
        expected_source_count=int(columns_raw["expected_source_count"]),
        exclude_indices=list(columns_raw["exclude_indices"]),
        clean_names=list(columns_raw["clean_names"]),
        timestamp_col=str(columns_raw["timestamp_col"]),
        speed_col=str(columns_raw["speed_col"]),
        current_col=str(columns_raw["current_col"]),
        pressure_col=pressure_col,
        pressure_off_col=pressure_off_col,
        temp_cols=list(columns_raw["temp_cols"]),
        prox_cols=list(columns_raw["prox_cols"]),
        baseline_extra_cols=list(columns_raw.get("baseline_extra_cols", [])),
        supplementary_columns=supplementary_columns,
    )


def _build_thresholds(thresholds_raw: Any) -> Thresholds:
    if thresholds_raw is None:
        return Thresholds()
    data = _require_mapping(thresholds_raw, "thresholds")
    try:
        return Thresholds(
            min_period_duration=float(data.get("min_period_duration", 600.0)),
            speed_on_threshold=float(data.get("speed_on_threshold", 100.0)),
            gap_merge_threshold=float(data.get("gap_merge_threshold", 30.0)),
            ramp_exclude_seconds=float(data.get("ramp_exclude_seconds", 300.0)),
            anomaly_sigma=float(data.get("anomaly_sigma", 3.0)),
            vibration_spike_factor=float(data.get("vibration_spike_factor", 2.0)),
            pressure_off_absolute_threshold=float(data.get("pressure_off_absolute_threshold", 5.0)),
            pressure_off_rebound_threshold=float(data.get("pressure_off_rebound_threshold", 4.0)),
            pressure_off_extended_sigma=float(data.get("pressure_off_extended_sigma", 2.5)),
            off_state_sigma=float(data.get("off_state_sigma", 4.0)),
            frozen_sensor_std_threshold=float(data.get("frozen_sensor_std_threshold", 0.01)),
            frozen_sensor_min_samples=int(data.get("frozen_sensor_min_samples", 10)),
            outlet_pressure_variability_cap=float(data.get("outlet_pressure_variability_cap", 0.0)),
            min_detectors_for_abnormal=int(data.get("min_detectors_for_abnormal", 1)),
            absolute_temp_limits=dict(data.get("absolute_temp_limits", {})),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid thresholds configuration: {exc}") from exc


def _build_examples(examples_raw: Any) -> list[LabelledExample]:
    if examples_raw is None:
        return []

    examples_map = _require_mapping(examples_raw, "examples")
    labelled: list[LabelledExample] = []

    for expected in ("normal", "abnormal"):
        entries = examples_map.get(expected, [])
        if not isinstance(entries, list):
            raise ValueError(f"examples.{expected} must be a list")
        for item in entries:
            if not isinstance(item, dict):
                raise ValueError(f"Each item in examples.{expected} must be a mapping")
            if "date" not in item or "pump" not in item:
                raise ValueError(
                    f"Each item in examples.{expected} must include 'date' and 'pump'"
                )
            labelled.append(
                LabelledExample(
                    date=str(item["date"]),
                    pump=int(item["pump"]),
                    expected=expected,
                )
            )

    return labelled


def _build_data_selector(raw: Any, base_dir: Path) -> DataSelectorConfig:
    if raw is None:
        return DataSelectorConfig()
    data = _require_mapping(raw, "data_selector")
    train_out = _resolve_path(data.get("train_output"), base_dir)
    test_out = _resolve_path(data.get("test_output"), base_dir)
    return DataSelectorConfig(
        min_normal_duration_minutes=float(data.get("min_normal_duration_minutes", 120.0)),
        max_periods_per_day=int(data.get("max_periods_per_day", 2)),
        train_output=train_out,
        test_output=test_out,
        use_symlinks=bool(data.get("use_symlinks", True)),
        exclude_reasons=list(data.get("exclude_reasons", [])),
    )


def _validate_config(columns: ColumnConfig, file_pattern: str) -> None:
    for col_name, field_name in (
        (columns.timestamp_col, "timestamp_col"),
        (columns.speed_col, "speed_col"),
        (columns.current_col, "current_col"),
    ):
        if col_name not in columns.clean_names:
            raise ValueError(
                f"'{field_name}' value '{col_name}' must be included in columns.clean_names"
            )

    try:
        re.compile(file_pattern)
    except re.error as exc:
        raise ValueError(f"Invalid file_pattern regex: {exc}") from exc


def load_config(yaml_path: Path) -> SystemConfig:
    """Load and validate system configuration from a YAML file."""

    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("Configuration YAML is empty")

    root = _require_mapping(raw, "root")

    required_root_fields = [
        "system_name",
        "data_root",
        "output_dir",
        "file_pattern",
        "date_format",
        "pump_ids",
        "columns",
        "detectors",
    ]
    for field_name in required_root_fields:
        _require_key(root, field_name, "root")

    base_dir = yaml_path.parent

    columns = _build_columns(_require_mapping(root["columns"], "columns"))
    thresholds = _build_thresholds(root.get("thresholds"))
    examples = _build_examples(root.get("examples"))
    data_selector = _build_data_selector(root.get("data_selector"), base_dir)

    detectors_raw = root["detectors"]
    if not isinstance(detectors_raw, list):
        raise ValueError("'detectors' must be a list")

    detectors = [str(name) for name in detectors_raw]

    config = SystemConfig(
        system_name=str(root["system_name"]),
        data_root=_resolve_path(str(root["data_root"]), base_dir),
        output_dir=_resolve_path(str(root["output_dir"]), base_dir),
        file_pattern=str(root["file_pattern"]),
        date_format=str(root["date_format"]),
        pump_ids=[int(p) for p in list(root["pump_ids"])],
        columns=columns,
        thresholds=thresholds,
        detectors=detectors,
        custom_detectors_dir=_resolve_path(root.get("custom_detectors_dir"), base_dir),
        examples=examples,
        data_selector=data_selector,
    )

    if config.data_root is None or config.output_dir is None:
        raise ValueError("'data_root' and 'output_dir' must be valid paths")

    _validate_config(config.columns, config.file_pattern)

    return config


def generate_template(output_path: Path) -> None:
    """Generate a well-commented YAML template with all available keys."""

    template = """# Pump annotation system configuration template
# All paths can be absolute or relative to this YAML file.

system_name: your_system_name
data_root: ../data
output_dir: ../

# Regex must capture two groups in order: (pump_id), (date)
# Example filename: pump_1_2026-03-13.csv
file_pattern: "^pump_(\\\\d+)_(\\\\d{4}-\\\\d{2}-\\\\d{2})\\\\.csv$"
# Python datetime.strptime format matching the date capture group above.
date_format: "%Y-%m-%d"

# Pumps to process
pump_ids: [1, 2, 3]

columns:
  # Number of columns expected in source CSV before exclusions.
  expected_source_count: 16
  # 0-based source indices to drop before assigning clean_names.
  exclude_indices: [7, 8]
  # Internal schema after exclusions/renaming.
  clean_names:
    - timestamp
    - motor_current
    - speed
    - npsh_pressure
    - nde_thrust_temp
    - nde_radial_temp
    - de_radial_temp
    - motor_nde_temp
    - motor_de_temp
    - nde_radial_prox_x
    - nde_radial_prox_y
    - de_radial_prox_x
    - de_radial_prox_y
    - nde_thrust_prox
  # Required core columns
  timestamp_col: timestamp
  speed_col: speed
  current_col: motor_current
  # Optional for systems without pressure instrumentation
  pressure_col: npsh_pressure
  temp_cols:
    - nde_thrust_temp
    - nde_radial_temp
    - de_radial_temp
    - motor_nde_temp
    - motor_de_temp
  prox_cols:
    - nde_radial_prox_x
    - nde_radial_prox_y
    - de_radial_prox_x
    - de_radial_prox_y
    - nde_thrust_prox

# Any missing threshold key falls back to defaults in system_config.py
thresholds:
  min_period_duration: 600
  speed_on_threshold: 100.0
  gap_merge_threshold: 30
  ramp_exclude_seconds: 300
  anomaly_sigma: 3.0
  vibration_spike_factor: 2.0
  pressure_off_absolute_threshold: 5.0
  pressure_off_rebound_threshold: 4.0
  off_state_sigma: 4.0

# Ordered detector execution list
detectors:
  - speed_stability
  - current_anomaly
  - current_speed_ratio
  - pressure_on
  - pressure_off
  - pressure_off_variability
  - temperatures
  - vibration
  - off_state_current

# Optional location for custom detector modules
# custom_detectors_dir: ../detectors

# Optional labelled examples for validation/reporting
examples:
  normal:
    - {date: "2026-02-16", pump: 1}
  abnormal:
    - {date: "2025-09-29", pump: 3}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template, encoding="utf-8")
