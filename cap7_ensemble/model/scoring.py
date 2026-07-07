from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

STATUS_NORMAL = "NORMAL"
STATUS_WARNING = "WARNING"
STATUS_ALARM = "ALARM"


@dataclass
class FeatureAnomaly:
    """A single anomalous feature at one timestep."""

    feature: str
    actual: float
    predicted: float
    residual: float
    z_score: float
    is_anomalous: bool  # True if |z_score| > feature threshold


@dataclass
class ChannelHealth:
    """Per-sensor health indicator for one timestep.

    Diagnostic only — explains which sensors drive the pump's aggregate state.
    Does not produce per-channel alerts.
    """

    feature: str
    health: float  # 0-100 sigmoid score
    z_score: float  # |residual| / std_residual
    residual: float  # actual - predicted (signed, physical units)


@dataclass
class ChannelHealthSummary:
    """Day-level aggregation of per-channel health for one pump."""

    feature: str
    mean_health: float
    min_health: float
    mean_z_score: float
    max_z_score: float
    n_timesteps_below_50: int


@dataclass
class TimestepResult:
    """Level 1 result for a single timestep."""

    timestamp: str
    mahalanobis: float
    status: str  # "NORMAL" / "WARNING" / "ALARM"
    feature_scores: list[FeatureAnomaly] = field(default_factory=list)


@dataclass
class Level1Result:
    """Aggregated Level 1 results for one pump-day."""

    pump_id: int
    date: str
    status: str  # "NORMAL" / "WARNING" / "ALARM"
    day_mean_mahalanobis: float
    day_max_mahalanobis: float
    fraction_above_warning: float
    fraction_above_alarm: float
    normalized_severity: float  # 0-1 scale
    n_timesteps: int
    top_anomalous_timesteps: list[TimestepResult] = field(default_factory=list)
    all_timestep_results: list[TimestepResult] = field(default_factory=list, repr=False)
    channel_health_summary: list[ChannelHealthSummary] = field(
        default_factory=list, repr=False
    )


@dataclass
class WindowResult:
    """Level 2 result for a single sliding window."""

    window_index: int
    timestamp: str
    raw_mse: float
    smoothed_mse: float
    status: str  # "NORMAL" / "WARNING" / "ALARM"


@dataclass
class Level2Result:
    """Level 2 results for one pump-day."""

    pump_id: int
    date: str
    status: str  # "NORMAL" / "WARNING" / "ALARM"
    day_error_mse: float
    warning_threshold: float
    alarm_threshold: float
    normalized_severity: float  # 0-1 scale
    n_samples: int = 0
    # Per-window streaming results
    window_results: list[WindowResult] = field(default_factory=list)
    window_warning_threshold: float = 0.0
    window_alarm_threshold: float = 0.0
    smoothing_alpha: float = 0.3
    fraction_windows_warning: float = 0.0
    fraction_windows_alarm: float = 0.0


@dataclass
class WindowEnsembleResult:
    """Ensemble result for a single window, aligning L2 window with L1 timesteps."""

    window_index: int
    timestamp: str
    level2_smoothed_mse: float
    level2_status: str
    level1_max_mahalanobis: float  # max Mahalanobis of L1 timesteps within this window
    level1_status: str
    ensemble_status: str
    ensemble_severity: float


@dataclass
class PumpEnsembleResult:
    """Combined ensemble result for one pump-day."""

    pump_id: int
    date: str
    overall_status: str  # "NORMAL" / "WARNING" / "ALARM"
    overall_severity: float  # 0-2 scale (max possible)
    level1: Level1Result
    level2: Optional[Level2Result]  # None if insufficient data for temporal window
    ensemble_reasoning: str
    window_ensemble_results: list[WindowEnsembleResult] = field(default_factory=list)
    channel_health_summary: list[ChannelHealthSummary] = field(default_factory=list)


@dataclass
class EnsembleReport:
    """Full ensemble inference report."""

    overall_status: str  # worst status across all pumps
    overall_severity: float  # max severity across all pumps
    pump_results: list[PumpEnsembleResult] = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    model_versions: dict = field(default_factory=dict)

    def to_dict(
        self, include_window_details: bool = True, include_all_timesteps: bool = False
    ) -> dict:
        d = asdict(self)
        for pr in d.get("pump_results", []):
            if not include_window_details:
                pr.pop("window_ensemble_results", None)
                if pr.get("level2"):
                    pr["level2"].pop("window_results", None)
            if not include_all_timesteps:
                if pr.get("level1"):
                    pr["level1"].pop("all_timestep_results", None)
        return d


def classify_status(
    score: float, warning_threshold: float, alarm_threshold: float
) -> str:
    """Return 'NORMAL', 'WARNING', or 'ALARM' based on score vs thresholds."""
    if alarm_threshold < warning_threshold:
        raise ValueError(
            "alarm_threshold must be greater than or equal to warning_threshold"
        )

    if score >= alarm_threshold:
        return STATUS_ALARM
    if score >= warning_threshold:
        return STATUS_WARNING
    return STATUS_NORMAL


def normalize_severity(score: float, alarm_threshold: float) -> float:
    """
    Normalize a raw anomaly score to [0, 1] range.
    0 = perfectly normal, 1 = at alarm threshold, >1 = beyond alarm.
    Clamped to [0, 2].
    """
    if alarm_threshold <= 0:
        return 0.0

    normalized = score / alarm_threshold
    if normalized < 0.0:
        return 0.0
    if normalized > 2.0:
        return 2.0
    return normalized


def compute_severity(level1_severity: float, level2_severity: float | None) -> float:
    """
    Compute composite severity score.

    severity = max(L1, L2) + 0.3 * min(L1, L2)

    The max ensures the worst signal dominates.
    The 0.3 * min adds a bonus when both levels detect anomalies (corroboration).

    If level2 is None, severity = level1_severity.
    """
    if level2_severity is None:
        return max(0.0, min(level1_severity, 2.0))

    l1 = max(0.0, level1_severity)
    l2 = max(0.0, level2_severity)
    severity = max(l1, l2) + 0.3 * min(l1, l2)
    return max(0.0, min(severity, 2.0))


def fuse_statuses(level1_status: str, level2_status: str | None) -> tuple[str, str]:
    """
    Combine Level 1 and Level 2 statuses into ensemble decision.

    Returns (ensemble_status, reasoning_text)

    Decision matrix:
    L1 NORMAL  + L2 NORMAL  -> NORMAL
    L1 WARNING + L2 NORMAL  -> WARNING (instantaneous deviations)
    L1 ALARM   + L2 NORMAL  -> ALARM (acute event)
    L1 NORMAL  + L2 WARNING -> WARNING (degradation trend)
    L1 NORMAL  + L2 ALARM   -> ALARM (degradation confirmed)
    L1 WARNING + L2 WARNING -> ALARM (corroborated by both levels)
    L1 WARNING + L2 ALARM   -> ALARM (confirmed degradation + deviations)
    L1 ALARM   + L2 WARNING -> ALARM (confirmed acute + degradation)
    L1 ALARM   + L2 ALARM   -> ALARM (critical - both levels)

    If Level 2 is None (insufficient data), fall back to Level 1 only.
    """
    valid_statuses = {STATUS_NORMAL, STATUS_WARNING, STATUS_ALARM}
    if level1_status not in valid_statuses:
        raise ValueError(f"Unknown level1 status: {level1_status}")

    if level2_status is None:
        return (
            level1_status,
            "Level 2 unavailable (insufficient temporal window); ensemble falls back to Level 1.",
        )

    if level2_status not in valid_statuses:
        raise ValueError(f"Unknown level2 status: {level2_status}")

    # Explicit matrix so fusion behavior is stable and auditable.
    matrix: dict[tuple[str, str], tuple[str, str]] = {
        (STATUS_NORMAL, STATUS_NORMAL): (
            STATUS_NORMAL,
            "Both levels are NORMAL; no acute deviations or temporal degradation detected.",
        ),
        (STATUS_WARNING, STATUS_NORMAL): (
            STATUS_WARNING,
            "Level 1 WARNING with Level 2 NORMAL; instantaneous deviations detected without temporal confirmation.",
        ),
        (STATUS_ALARM, STATUS_NORMAL): (
            STATUS_ALARM,
            "Level 1 ALARM with Level 2 NORMAL; acute event detected by instantaneous monitoring.",
        ),
        (STATUS_NORMAL, STATUS_WARNING): (
            STATUS_WARNING,
            "Level 1 NORMAL with Level 2 WARNING; temporal pattern suggests degradation trend.",
        ),
        (STATUS_NORMAL, STATUS_ALARM): (
            STATUS_ALARM,
            "Level 1 NORMAL with Level 2 ALARM; degradation is strongly confirmed over time.",
        ),
        (STATUS_WARNING, STATUS_WARNING): (
            STATUS_WARNING,
            "Both layers show mild deviations; continued monitoring.",
        ),
        (STATUS_WARNING, STATUS_ALARM): (
            STATUS_ALARM,
            "Level 1 WARNING and Level 2 ALARM; confirmed degradation with active deviations.",
        ),
        (STATUS_ALARM, STATUS_WARNING): (
            STATUS_ALARM,
            "Level 1 ALARM and Level 2 WARNING; acute issue corroborated by temporal degradation.",
        ),
        (STATUS_ALARM, STATUS_ALARM): (
            STATUS_ALARM,
            "Both levels are ALARM; critical condition confirmed across instantaneous and temporal analyses.",
        ),
    }

    return matrix[(level1_status, level2_status)]


def compute_health_score(
    score: float, training_mean: float, alarm_threshold: float
) -> float:
    """Map anomaly score to 0-100 health using an inverted sigmoid."""
    if alarm_threshold <= training_mean:
        return 100.0 if score <= training_mean else 0.0
    x = (score - training_mean) / (alarm_threshold - training_mean)
    # Softer slope preserves useful health gradient above alarm.
    exponent = 6.0 * (x - 0.5)
    exponent = max(-20.0, min(20.0, float(exponent)))
    health = 100.0 / (1.0 + math.exp(exponent))
    return max(0.0, min(100.0, float(health)))


def compute_ensemble_health(l1_health: float, l2_health: float | None) -> float:
    """Combine health values, dominated by the worse signal."""
    if l2_health is None:
        return float(l1_health)
    return float(0.6 * min(l1_health, l2_health) + 0.4 * max(l1_health, l2_health))


def compute_channel_health(
    residual: float,
    std_residual: float,
    p99_abs_residual: float,
    mean_residual: float = 0.0,
) -> tuple[float, float]:
    """Compute per-channel health score (diagnostic only).

    The score is designed to *agree* with the multivariate Mahalanobis alarm
    decision rather than contradict it. Two properties make that hold:

    1. **Mean-centering.** The deviation is measured from the channel's
       calibrated mean residual (``z = |residual - mean_residual| / std``),
       exactly as the Mahalanobis path subtracts the mean residual vector
       (see ``Level1Detector._score_timestep``). A channel's known, expected
       bias therefore does not count as ill-health.
    2. **Envelope-anchored band.** Health 50 is placed at the channel's own
       99th-percentile deviation (``z = p99/std``), with the alarm tail
       (health -> 0) at twice that. A steady ~1.5-sigma channel — which the
       covariance-whitened detector treats as normal — therefore reads green,
       while a genuine multi-sigma excursion on a failure day reddens.

    Parameters
    ----------
    residual : signed residual (actual - predicted) in physical units.
    std_residual : training std of this channel's residuals for this pump.
    p99_abs_residual : 99th percentile of |residual| during training.
    mean_residual : training mean residual for this channel (the calibrated
        systematic bias). Defaults to 0.0 so existing callers are unchanged.

    Returns
    -------
    health : 0-100 sigmoid health.
    z_score : |residual - mean_residual| / std_residual (centered).
    """
    std_safe = max(abs(std_residual), 1e-9)
    p99_safe = max(abs(p99_abs_residual), 1e-9)
    z = abs(residual - mean_residual) / std_safe
    # Health 50 at the 99th-percentile deviation, red tail at twice that.
    z_alarm = 2.0 * (p99_safe / std_safe)
    health = compute_health_score(z, training_mean=0.0, alarm_threshold=z_alarm)
    return health, z


__all__ = [
    "STATUS_NORMAL",
    "STATUS_WARNING",
    "STATUS_ALARM",
    "FeatureAnomaly",
    "ChannelHealth",
    "ChannelHealthSummary",
    "TimestepResult",
    "Level1Result",
    "WindowResult",
    "Level2Result",
    "WindowEnsembleResult",
    "PumpEnsembleResult",
    "EnsembleReport",
    "classify_status",
    "normalize_severity",
    "compute_severity",
    "fuse_statuses",
    "compute_health_score",
    "compute_ensemble_health",
    "compute_channel_health",
]
