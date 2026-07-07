"""cond_reg_v2 model package - Temporal Conditional VAE with attention."""

from importlib import import_module
from typing import Any

__all__ = [
	"TemporalCVAE",
	"PumpDataset",
	"PumpPredictor",
	"FailureDetector",
	"HyperparameterTuner",
	"ThresholdCalibrator",
]


def __getattr__(name: str) -> Any:
	if name in {"TemporalCVAE", "PumpDataset"}:
		module = import_module(".models", __name__)
		return getattr(module, name)
	if name == "PumpPredictor":
		module = import_module(".inference", __name__)
		return getattr(module, name)
	if name == "FailureDetector":
		module = import_module(".failure_detector", __name__)
		return getattr(module, name)
	if name == "HyperparameterTuner":
		module = import_module(".fine_tuning", __name__)
		return getattr(module, name)
	if name == "ThresholdCalibrator":
		module = import_module(".threshold_calibration", __name__)
		return getattr(module, name)
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
