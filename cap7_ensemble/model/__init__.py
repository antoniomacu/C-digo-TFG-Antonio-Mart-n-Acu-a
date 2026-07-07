"""
Ensemble Pump Anomaly Detection
================================

Two-level ensemble combining instantaneous digital twin monitoring (Level 1)
with temporal VAE degradation detection (Level 2) for comprehensive
industrial pump health assessment.

Quick Start::

    from ensemble.model import EnsembleDetector

    detector = EnsembleDetector()
    report = detector.classify(["pump_1_2025-10-06.csv"])
    print(report.to_dict())

Or from the command line::

    ensemble-inference pump_1.csv pump_3.csv --verbose
"""

from .ensemble import EnsembleDetector
from .inference import EnsembleInference
from .monitoring import (
    DailyScoreSummary,
    DriftAlert,
    DriftMonitor,
    RecalibrationRecord,
    RecalibrationTracker,
    SeasonalTracker,
)
from .scoring import (
    WindowResult,
    WindowEnsembleResult,
    ChannelHealth,
    ChannelHealthSummary,
    compute_channel_health,
    compute_ensemble_health,
    compute_health_score,
)
from .streaming import (
    StreamingEnsembleDetector,
    StreamingTimestepResult,
    create_streaming_detector,
)

__all__ = [
    "DailyScoreSummary",
    "DriftAlert",
    "DriftMonitor",
    "EnsembleDetector",
    "EnsembleInference",
    "RecalibrationRecord",
    "RecalibrationTracker",
    "SeasonalTracker",
    "WindowResult",
    "WindowEnsembleResult",
    "StreamingEnsembleDetector",
    "StreamingTimestepResult",
    "create_streaming_detector",
    "compute_health_score",
    "compute_ensemble_health",
    "ChannelHealth",
    "ChannelHealthSummary",
    "compute_channel_health",
]
