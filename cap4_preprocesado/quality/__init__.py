"""Generalized data quality pipeline for pump systems.

Usage:
	uv run -m quality --config configs/htf_pumps.yaml
	uv run -m quality --config configs/feedwater_pumps.yaml --pump 1,2
"""

from quality.quality_config import (
	QualityConfig,
	SensorQualityConfig,
	CleaningConfig,
	ResampleConfig,
	load_quality_config,
	load_quality_and_system_config,
)
from quality.labeler import (
	Q_OK,
	Q_MISSING,
	Q_NULL,
	Q_OUT_OF_RANGE,
	Q_FALL_DOWN,
	Q_FROZEN,
	CODE_NAMES,
	label_dataframe,
	is_all_frozen,
	detect_sentinel_rows,
)
from quality.cleaner import clean_dataframe
from quality.operational_filter import filter_operational, split_operational_periods
from quality.resampler import resample_dataframe
from quality.reporter import generate_text_summary, generate_html_dashboard

__all__ = [
	"QualityConfig",
	"SensorQualityConfig",
	"CleaningConfig",
	"ResampleConfig",
	"load_quality_config",
	"load_quality_and_system_config",
	"Q_OK",
	"Q_MISSING",
	"Q_NULL",
	"Q_OUT_OF_RANGE",
	"Q_FALL_DOWN",
	"Q_FROZEN",
	"CODE_NAMES",
	"label_dataframe",
	"is_all_frozen",
	"detect_sentinel_rows",
	"clean_dataframe",
	"filter_operational",
	"split_operational_periods",
	"resample_dataframe",
	"generate_text_summary",
	"generate_html_dashboard",
]
