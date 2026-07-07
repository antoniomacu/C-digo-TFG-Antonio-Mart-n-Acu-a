from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pipeline.baseline import PumpBaseline
from pipeline.custom_detectors._ml_features import extract_period_features
from pipeline.period_detector import OperationPeriod, get_steady_state_mask
from pipeline.system_config import SystemConfig

_MODEL_CACHE: dict[str, Any] = {}
_WARNED = False
_DIMENSION_WARNED = False


def _warn_once(message: str, *, category: type[Warning] = RuntimeWarning) -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(message, category=category, stacklevel=2)
        _WARNED = True


def _warn_dimension_once(message: str, *, category: type[Warning] = RuntimeWarning) -> None:
    global _DIMENSION_WARNED
    if not _DIMENSION_WARNED:
        warnings.warn(message, category=category, stacklevel=2)
        _DIMENSION_WARNED = True


def _load_artifact() -> Any | None:
    if "artifact" in _MODEL_CACHE:
        return _MODEL_CACHE["artifact"]
    if _MODEL_CACHE.get("missing", False):
        return None

    model_path = Path(__file__).parent / "ml_ensemble_models.joblib"
    if not model_path.exists():
        _warn_once(f"ML ensemble model artifact not found: {model_path}")
        _MODEL_CACHE["missing"] = True
        return None

    try:
        artifact = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - safety net for runtime environment
        _warn_once(f"Failed to load ML ensemble artifact from {model_path}: {exc}")
        _MODEL_CACHE["missing"] = True
        return None

    _MODEL_CACHE["artifact"] = artifact
    return artifact


def _check_ml_ensemble(
    df: pd.DataFrame,
    period: OperationPeriod,
    baseline: PumpBaseline,
    cfg: SystemConfig,
) -> tuple[bool, str]:
    # Baseline is part of the detector interface; this detector does not use it directly.
    _ = baseline

    artifact = _load_artifact()
    if artifact is None:
        return False, ""

    try:
        steady_mask = get_steady_state_mask(df, period, cfg)
        steady_df = df.loc[steady_mask]
        if steady_df.empty:
            return False, ""

        features = extract_period_features(steady_df, cfg)
        if features is None:
            return False, ""

        feature_names = artifact.get("feature_names", [])
        expected_dim = len(feature_names)
        if features.shape[0] != expected_dim:
            _warn_dimension_once(
                "ML ensemble feature dimension mismatch: "
                f"got {features.shape[0]}, expected {expected_dim}"
            )
            return False, ""

        X = np.asarray(features).reshape(1, -1)

        lof_pred = artifact["lof"].predict(X)
        iforest_pred = artifact["iforest"].predict(X)

        diff = np.asarray(features) - np.asarray(artifact["mahal_mean"])
        d_sq = float(diff @ np.asarray(artifact["mahal_cov_inv"]) @ diff)
        mahal_flag = d_sq > float(artifact["mahal_threshold"])

        flagged_models = [
            "lof" if int(lof_pred[0]) == -1 else "",
            "iforest" if int(iforest_pred[0]) == -1 else "",
            "mahalanobis" if mahal_flag else "",
        ]
        vote_count = sum(1 for name in flagged_models if name)

        if vote_count >= 3:
            return True, "ml_ensemble"
        return False, ""
    except Exception as exc:  # pragma: no cover - hard safety net to protect pipeline runtime
        _warn_once(f"ML ensemble detector failed during scoring: {exc}")
        return False, ""


DETECTORS = {
    "ml_ensemble": _check_ml_ensemble,
}
