from __future__ import annotations

import json
import logging
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .level1_detector import Level1Detector
from .monitoring import AlarmMonitor
from .scoring import (
    compute_ensemble_health,
    compute_health_score,
    fuse_statuses,
    compute_channel_health,
    ChannelHealth,
)


logger = logging.getLogger(__name__)


@dataclass
class StreamingTimestepResult:
    timestep: int
    timestamp: str
    pump_id: int

    # Level 1 (always available from timestep 0)
    l1_mahalanobis: float
    l1_status: str
    l1_health: float
    l1_z_scores: dict[str, float]
    l1_actual: dict[str, float]
    l1_predicted: dict[str, float]
    l1_residuals: dict[str, float]
    l1_anomalous_features: list[str]

    # Level 2 (None while BUFFERING)
    l2_raw_mse: float | None = None
    l2_smoothed_mse: float | None = None
    l2_status: str = "BUFFERING"
    l2_health: float | None = None

    # Ensemble
    ensemble_status: str = "NORMAL"  # spike-filtered status (post-AlarmMonitor)
    ensemble_status_raw: str = "NORMAL"  # unfiltered fusion output (pre-AlarmMonitor)
    ensemble_health: float = 100.0
    ensemble_reasoning: str = ""

    # AlarmMonitor outputs
    alarm_alert_fired: bool = False  # True = new operator notification should fire

    # Session-level K-aggregation (counts WARNING+ALARM steps since reset_pump)
    session_alert_steps: int = 0
    session_flagged: bool = False
    # Per-channel health (diagnostic — explains what drives ensemble state)
    channel_health: list = field(default_factory=list)


@dataclass
class _PumpStreamState:
    pump_id: int
    timestep_count: int = 0
    l2_raw_buffer: list = field(default_factory=list)
    l2_ema_prev: float | None = None
    # Per-pump thresholds (populated on init)
    l1_warning: float = 0.0
    l1_alarm: float = 0.0
    l1_training_mean: float = 0.0
    l2_window_warning: float = 0.0
    l2_window_alarm: float = 0.0
    l2_training_mean: float = 0.0
    # Session-level alert step counter (reset via reset_pump); used for K-aggregation
    session_alert_steps: int = 0
    # Per-pump AlarmMonitor instance (spike filter + rate limiter)
    alarm_monitor: AlarmMonitor = field(default_factory=AlarmMonitor)


class _Layer(nn.Module):
    """Simple FC block used by the lightweight inference-only VAE."""

    def __init__(self, in_dim: int, out_dim: int, batch_norm: bool):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Encoder(nn.Module):
    def __init__(self, hparams: dict[str, Any]):
        super().__init__()
        in_dim = len(hparams["input_variables"]) * int(hparams["past_history"])
        hidden = [int(s) for s in str(hparams["layer_sizes"]).split(",")]
        batch_norm = bool(hparams.get("batch_norm", True))
        dims = [in_dim] + hidden
        self.layers = nn.Sequential(
            *[_Layer(dims[i], dims[i + 1], batch_norm) for i in range(len(dims) - 1)]
        )
        self.mu = nn.Linear(dims[-1], int(hparams["latent_dim"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layers(x)
        return self.mu(h)


class _Decoder(nn.Module):
    def __init__(self, hparams: dict[str, Any]):
        super().__init__()
        hidden_rev = [int(s) for s in reversed(str(hparams["layer_sizes"]).split(","))]
        dims = [int(hparams["latent_dim"])] + hidden_rev
        batch_norm = bool(hparams.get("batch_norm", True))
        out_dim = len(hparams["output_variables"]) * int(hparams["past_history"])
        self.layers = nn.Sequential(
            *[_Layer(dims[i], dims[i + 1], batch_norm) for i in range(len(dims) - 1)]
        )
        self.reconstructed = nn.Linear(dims[-1], out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.reconstructed(self.layers(z))


class _InferenceVAE(nn.Module):
    """Inference-only wrapper compatible with Level 2 checkpoint weights."""

    def __init__(self, hparams: dict[str, Any]):
        super().__init__()
        self.encoder = _Encoder(hparams)
        self.decoder = _Decoder(hparams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class StreamingEnsembleDetector:
    def __init__(
        self,
        l1_detector: Level1Detector,
        l2_model: nn.Module,
        l2_preprocessor,
        l2_norm_params: dict,
        l2_hparams: dict,
        l1_thresholds: dict,
        l2_thresholds: dict,
        ema_alpha: float = 0.3,
        min_session_alert_steps: int = 3,
        min_alarm_duration: int = 2,
        alarm_rate_limit_seconds: float = 3600.0,
        l2_seasonal_thresholds: dict | None = None,
    ):
        self._l1_detector = l1_detector
        self._l2_model = l2_model.eval()
        self._l2_seasonal_thresholds = l2_seasonal_thresholds
        self._l2_preprocessor = l2_preprocessor
        self._l2_norm_params = l2_norm_params
        self._l2_hparams = l2_hparams
        self._l1_thresholds = l1_thresholds
        self._l2_thresholds = l2_thresholds
        self._ema_alpha = float(ema_alpha)
        self._min_session_alert_steps = int(min_session_alert_steps)
        self._min_alarm_duration = int(min_alarm_duration)
        self._alarm_rate_limit_seconds = float(alarm_rate_limit_seconds)

        self._l1_input_cols = [
            "Ambient temperature",
            "Main HTF Pump Speed",
            "Main HTF Pump Inlet Temperature",
            "pump_id",
        ]
        self._l1_output_cols = list(self._l1_detector._output_columns)
        self._l1_past_history = int(
            getattr(self._l1_detector._predictor, "_past_history", 3)
        )

        self._l2_past_history = int(self._l2_hparams.get("past_history", 36))
        self._l2_input_vars = list(self._l2_hparams["input_variables"])
        self._l2_output_vars = list(self._l2_hparams["output_variables"])
        self._l2_norm_method = str(self._l2_hparams.get("norm_method", "min-max"))

        self._pump_states: dict[int, _PumpStreamState] = {}
        self._l1_history: dict[int, list[tuple[pd.Timestamp, pd.Series]]] = {}

        # Seasonal threshold tracking — detect season boundary crossings in process_timestep.
        self._active_season: str | None = None

    @property
    def pump_ids(self) -> list[int]:
        return sorted(self._pump_states.keys())

    def reset_pump(self, pump_id: int) -> None:
        pump = int(pump_id)
        state = self._pump_states.pop(pump, None)
        if state is not None:
            state.alarm_monitor.reset()
        self._l1_history.pop(pump, None)

    def reset_all(self) -> None:
        self._pump_states.clear()
        self._l1_history.clear()

    def set_season(self, season: str) -> None:
        """Activate season for L1 and L2. Updates thresholds in all existing pump states."""
        self._active_season = season
        self._l1_detector.set_season(season)
        if self._l2_seasonal_thresholds is not None:
            for pump_id, state in self._pump_states.items():
                w, a, m = _get_l2_thresholds_for_pump(
                    pump_id, self._l2_thresholds, season, self._l2_seasonal_thresholds
                )
                state.l2_window_warning = w
                state.l2_window_alarm = a
                state.l2_training_mean = m

    def process_timestep(
        self, pump_id: int, timestamp, row: pd.Series
    ) -> StreamingTimestepResult:
        pump = int(pump_id)
        ts = pd.Timestamp(timestamp)

        # Detect season boundary and propagate to both detectors.
        try:
            from .monitoring import SeasonalTracker

            new_season = SeasonalTracker.tag_season(ts.date())
            if new_season != self._active_season:
                self.set_season(new_season)
                logger.info("Season boundary crossed: now '%s'", new_season)
        except Exception:
            pass

        state = self._get_or_create_state(pump)

        # Keep local copies so per-pump streaming remains isolated.
        row_series = row.copy()
        row_series["pump_id"] = pump

        # L2 stores raw rows and preprocesses on demand for temporal context.
        state.l2_raw_buffer.append((ts, row_series.copy()))

        l1_history = self._l1_history.setdefault(pump, [])
        l1_history.append((ts, row_series.copy()))
        if len(l1_history) > self._l1_past_history:
            del l1_history[0 : len(l1_history) - self._l1_past_history]

        timestamp_iso = ts.isoformat()
        timestep = state.timestep_count
        state.timestep_count += 1

        l1_df = pd.DataFrame(
            [r for _, r in l1_history], index=[t for t, _ in l1_history]
        ).sort_index()
        self._ensure_columns(
            l1_df, self._l1_input_cols + self._l1_output_cols, "Level 1"
        )

        l1_input = l1_df[self._l1_input_cols].apply(pd.to_numeric, errors="coerce")
        predicted_df = self._l1_detector._predictor.predict(l1_input)
        actual_row = pd.to_numeric(
            l1_df[self._l1_output_cols].iloc[-1], errors="coerce"
        ).to_numpy(dtype=float)
        predicted_row = pd.to_numeric(predicted_df.iloc[-1], errors="coerce").to_numpy(
            dtype=float
        )

        valid_mask = np.isfinite(actual_row) & np.isfinite(predicted_row)
        if not bool(valid_mask.any()):
            logger.warning(
                "No valid Level 1 features at %s for pump %s; falling back to neutral L1 score.",
                timestamp_iso,
                pump,
            )
            l2_raw, l2_smoothed, l2_status, l2_health = self._score_l2(
                state=state, current_ts=ts
            )
            ensemble_health = compute_ensemble_health(
                100.0, l2_health if l2_status != "BUFFERING" else None
            )
            ensemble_status_raw = "NORMAL" if l2_status == "BUFFERING" else l2_status
            ensemble_status, alarm_alert_fired = state.alarm_monitor.check(
                ensemble_status_raw, timestamp_iso
            )
            if ensemble_status in ("WARNING", "ALARM"):
                state.session_alert_steps += 1
            session_flagged = state.session_alert_steps >= self._min_session_alert_steps
            return StreamingTimestepResult(
                timestep=timestep,
                timestamp=timestamp_iso,
                pump_id=pump,
                l1_mahalanobis=0.0,
                l1_status="NORMAL",
                l1_health=100.0,
                l1_z_scores={},
                l1_actual={},
                l1_predicted={},
                l1_residuals={},
                l1_anomalous_features=[],
                l2_raw_mse=l2_raw,
                l2_smoothed_mse=l2_smoothed,
                l2_status=l2_status,
                l2_health=l2_health,
                ensemble_status=ensemble_status,
                ensemble_status_raw=ensemble_status_raw,
                ensemble_health=float(ensemble_health),
                ensemble_reasoning="Level 1 unavailable due to missing features; using Level 2 when available.",
                alarm_alert_fired=alarm_alert_fired,
                session_alert_steps=state.session_alert_steps,
                session_flagged=session_flagged,
                channel_health=[],
            )

        thresholds = self._l1_detector._get_pump_thresholds(pump)
        mean_vec = np.asarray(thresholds["mean_vector"], dtype=float)
        residual_for_scoring = np.where(
            valid_mask, actual_row - predicted_row, mean_vec
        )
        actual_for_report = np.where(valid_mask, actual_row, np.nan)
        predicted_for_report = np.where(valid_mask, predicted_row, np.nan)

        ts_result = self._l1_detector._score_timestep(
            residual_row=residual_for_scoring,
            pump_id=pump,
            timestamp=timestamp_iso,
            actual_row=actual_for_report,
            predicted_row=predicted_for_report,
        )

        l1_health = compute_health_score(
            float(ts_result.mahalanobis), state.l1_training_mean, state.l1_alarm
        )
        l1_z_scores = {
            fa.feature: (float(fa.z_score) if valid_mask[i] else float("nan"))
            for i, fa in enumerate(ts_result.feature_scores)
        }
        l1_actual = {
            fa.feature: (float(fa.actual) if valid_mask[i] else float("nan"))
            for i, fa in enumerate(ts_result.feature_scores)
        }
        l1_predicted = {
            fa.feature: (float(fa.predicted) if valid_mask[i] else float("nan"))
            for i, fa in enumerate(ts_result.feature_scores)
        }
        l1_residuals = {
            fa.feature: (float(fa.residual) if valid_mask[i] else float("nan"))
            for i, fa in enumerate(ts_result.feature_scores)
        }
        l1_anomalous_features = [
            fa.feature
            for i, fa in enumerate(ts_result.feature_scores)
            if valid_mask[i] and fa.is_anomalous
        ]

        # Per-channel health (diagnostic)
        channel_health_list: list[ChannelHealth] = []
        _feature_stats = self._get_per_pump_feature_stats(pump)
        for i, feature_name in enumerate(self._l1_output_cols):
            if not valid_mask[i]:
                continue
            _fs = _feature_stats.get(feature_name)
            if _fs is None:
                continue
            _residual_val = float(actual_row[i] - predicted_row[i])
            _h, _z = compute_channel_health(
                residual=_residual_val,
                std_residual=float(_fs.get("std_residual", 1.0)),
                p99_abs_residual=float(_fs.get("p99_abs_residual", 1.0)),
                mean_residual=float(_fs.get("mean_residual", 0.0)),
            )
            channel_health_list.append(
                ChannelHealth(
                    feature=feature_name,
                    health=_h,
                    z_score=_z,
                    residual=_residual_val,
                )
            )

        l2_raw, l2_smoothed, l2_status, l2_health = self._score_l2(
            state=state, current_ts=ts
        )

        if l2_status == "BUFFERING":
            ensemble_status_raw = ts_result.status
            ensemble_reasoning = f"L1={ts_result.status}, L2=BUFFERING -> {ensemble_status_raw} (L2 loading)"
            ensemble_health = compute_ensemble_health(float(l1_health), None)
        else:
            ensemble_status_raw, fusion_reasoning = fuse_statuses(
                ts_result.status, l2_status
            )
            ensemble_reasoning = f"L1={ts_result.status}, L2={l2_status} -> {ensemble_status_raw}. {fusion_reasoning}"
            ensemble_health = compute_ensemble_health(float(l1_health), l2_health)

        ensemble_status, alarm_alert_fired = state.alarm_monitor.check(
            ensemble_status_raw, timestamp_iso
        )

        if ensemble_status in ("WARNING", "ALARM"):
            state.session_alert_steps += 1
        session_flagged = state.session_alert_steps >= self._min_session_alert_steps

        return StreamingTimestepResult(
            timestep=timestep,
            timestamp=timestamp_iso,
            pump_id=pump,
            l1_mahalanobis=float(ts_result.mahalanobis),
            l1_status=ts_result.status,
            l1_health=float(l1_health),
            l1_z_scores=l1_z_scores,
            l1_actual=l1_actual,
            l1_predicted=l1_predicted,
            l1_residuals=l1_residuals,
            l1_anomalous_features=l1_anomalous_features,
            l2_raw_mse=l2_raw,
            l2_smoothed_mse=l2_smoothed,
            l2_status=l2_status,
            l2_health=l2_health,
            ensemble_status=ensemble_status,
            ensemble_status_raw=ensemble_status_raw,
            ensemble_health=float(ensemble_health),
            ensemble_reasoning=ensemble_reasoning,
            alarm_alert_fired=alarm_alert_fired,
            session_alert_steps=state.session_alert_steps,
            session_flagged=session_flagged,
            channel_health=channel_health_list,
        )

    def _score_l2(
        self,
        state: _PumpStreamState,
        current_ts: pd.Timestamp,
    ) -> tuple[float | None, float | None, str, float | None]:
        if len(state.l2_raw_buffer) < self._l2_past_history:
            return None, None, "BUFFERING", None

        idx = [ts for ts, _ in state.l2_raw_buffer]
        rows = [row for _, row in state.l2_raw_buffer]
        raw_df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx)).sort_index()
        raw_df["pump_id"] = int(state.pump_id)

        if len(raw_df) < 5:
            return None, None, "BUFFERING", None

        try:
            df_l2 = self._l2_preprocessor.create_dummies(raw_df.copy())
            df_l2_filled = self._prepare_l2_frame(df_l2)
            if len(df_l2_filled) < 5:
                return None, None, "BUFFERING", None

            # Apply SG filter per-day to match training preprocessing, which
            # processes each pump-day CSV independently.  Without this, the
            # filter smooths across day boundaries — a train/serve skew.
            _sg_segments = []
            for _, _day_df in df_l2_filled.groupby(df_l2_filled.index.date):
                if len(_day_df) >= 5:
                    _sg_segments.append(
                        self._l2_preprocessor.filter_savitzky_golay(_day_df)
                    )
                # Days with < 5 rows are skipped — training never sees
                # them either (Preprocessor._load_files skips short files).
            if not _sg_segments:
                return None, None, "BUFFERING", None
            df_l2_filtered = pd.concat(_sg_segments).sort_index()
            df_l2_norm = self._l2_preprocessor.normalize_data(
                df_l2_filtered, self._l2_norm_params, self._l2_norm_method
            )
            df_l2_norm = self._l2_preprocessor.rebuild_pump_id(df_l2_norm)
            df_work = df_l2_norm.drop(columns=["pump_id"])
        except Exception as exc:
            logger.warning(
                "L2 preprocessing failed at %s for pump %s: %s",
                current_ts.isoformat(),
                state.pump_id,
                exc,
            )
            return None, None, "BUFFERING", None

        if not all(v in df_work.columns for v in self._l2_input_vars):
            raise ValueError("L2 input variables are missing after preprocessing.")
        if not all(v in df_work.columns for v in self._l2_output_vars):
            raise ValueError("L2 output variables are missing after preprocessing.")

        l2_index = pd.DatetimeIndex(df_work.index)
        l2_idx = _find_nearest_l2_index(current_ts, l2_index)
        if l2_idx is None or l2_idx < self._l2_past_history - 1:
            return None, None, "BUFFERING", None

        i0 = l2_idx - self._l2_past_history + 1
        i1 = l2_idx + 1

        x_window = df_work[self._l2_input_vars].iloc[i0:i1].to_numpy(dtype=float)
        y_window = df_work[self._l2_output_vars].iloc[i0:i1].to_numpy(dtype=float)

        if (
            x_window.shape[0] != self._l2_past_history
            or y_window.shape[0] != self._l2_past_history
        ):
            return None, None, "BUFFERING", None

        device = next(self._l2_model.parameters()).device
        x_tensor = torch.tensor(x_window, dtype=torch.float32, device=device).reshape(
            1, -1
        )
        y_tensor = torch.tensor(y_window, dtype=torch.float32, device=device).reshape(
            1, -1
        )

        with torch.no_grad():
            output = self._l2_model(x_tensor)
            recon = output[0] if isinstance(output, tuple) else output
            raw_mse = float(((recon - y_tensor) ** 2).mean().item())

        if state.l2_ema_prev is None:
            # Initialize EMA to training mean so the first window doesn't spike from raw reconstruction error
            smoothed = (
                state.l2_training_mean
                if state.l2_training_mean is not None
                else raw_mse
            )
        else:
            smoothed = (
                self._ema_alpha * raw_mse + (1.0 - self._ema_alpha) * state.l2_ema_prev
            )
        state.l2_ema_prev = smoothed

        l2_status = _classify_l2_status(
            smoothed, state.l2_window_warning, state.l2_window_alarm
        )
        l2_health = compute_health_score(
            smoothed, state.l2_training_mean, state.l2_window_alarm
        )
        return raw_mse, smoothed, l2_status, float(l2_health)

    def _prepare_l2_frame(self, df_l2: pd.DataFrame) -> pd.DataFrame:
        """Prepare L2 frame robustly when some channels are sparse or fully missing."""
        df = df_l2.copy()
        required_cols = list(dict.fromkeys(self._l2_input_vars + self._l2_output_vars))

        for col in required_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.ffill(limit=1).bfill(limit=1)

        for col in required_cols:
            if col not in df.columns:
                continue

            if bool(df[col].isna().all()):
                fill_value = self._l2_norm_params.get(col, {}).get("mean")
                if fill_value is not None:
                    df[col] = float(fill_value)

            if bool(df[col].isna().any()):
                fill_value = self._l2_norm_params.get(col, {}).get("mean")
                if fill_value is not None:
                    df[col] = df[col].fillna(float(fill_value))

        keep_subset = [col for col in required_cols if col in df.columns]
        if keep_subset:
            df = df.dropna(subset=keep_subset)
        else:
            df = df.dropna()

        return df

    def _get_or_create_state(self, pump_id: int) -> _PumpStreamState:
        state = self._pump_states.get(pump_id)
        if state is not None:
            return state

        l1_warning, l1_alarm, l1_training_mean = _get_l1_thresholds_for_pump(
            pump_id, self._l1_thresholds
        )
        l2_window_warning, l2_window_alarm, l2_training_mean = (
            _get_l2_thresholds_for_pump(
                pump_id,
                self._l2_thresholds,
                self._active_season,
                self._l2_seasonal_thresholds,
            )
        )

        state = _PumpStreamState(
            pump_id=pump_id,
            l1_warning=l1_warning,
            l1_alarm=l1_alarm,
            l1_training_mean=l1_training_mean,
            l2_window_warning=l2_window_warning,
            l2_window_alarm=l2_window_alarm,
            l2_training_mean=l2_training_mean,
            alarm_monitor=AlarmMonitor(
                min_alarm_duration=self._min_alarm_duration,
                rate_limit_seconds=self._alarm_rate_limit_seconds,
            ),
        )
        self._pump_states[pump_id] = state
        return state

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, required: list[str], level_name: str) -> None:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required {level_name} columns: {missing}")

    def _get_per_pump_feature_stats(self, pump_id: int) -> dict[str, dict]:
        """Get per-feature residual stats for a pump via the L1 detector.

        Returns dict mapping feature_name → {"std_residual": float, "p99_abs_residual": float}.
        """
        return self._l1_detector._get_per_feature_stats(pump_id)


def _classify_l2_status(
    smoothed_mse: float, warning_threshold: float, alarm_threshold: float
) -> str:
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


def _get_l1_thresholds_for_pump(
    pump_id: int, l1_thresholds: dict
) -> tuple[float, float, float]:
    global_block = l1_thresholds.get("global", {})
    block = l1_thresholds.get("per_pump", {}).get(str(pump_id), global_block)
    if block.get("fallback_to_global", False):
        block = global_block
    maha = block.get("mahalanobis", global_block.get("mahalanobis", {}))

    warning = float(
        maha.get("warning", global_block.get("mahalanobis", {}).get("warning", 1.0))
    )
    alarm = float(
        maha.get("alarm", global_block.get("mahalanobis", {}).get("alarm", warning))
    )
    if alarm < warning:
        alarm = warning
    training_mean = float(
        maha.get("mean", global_block.get("mahalanobis", {}).get("mean", warning))
    )
    return warning, alarm, training_mean


def _get_l2_thresholds_for_pump(
    pump_id: int,
    l2_thresholds: dict,
    season: str | None = None,
    l2_seasonal_thresholds: dict | None = None,
) -> tuple[float, float, float]:
    global_block = l2_thresholds.get("global", {})
    block = l2_thresholds.get("per_pump", {}).get(str(pump_id), global_block)

    warning = float(
        block.get(
            "window_warning_smoothed", global_block.get("window_warning_smoothed", 1.0)
        )
    )
    alarm = float(
        block.get(
            "window_alarm_smoothed", global_block.get("window_alarm_smoothed", warning)
        )
    )
    if alarm < warning:
        alarm = warning
    training_mean = float(block.get("mean", global_block.get("mean", warning)))

    if season is not None and l2_seasonal_thresholds is not None:
        season_pump_map = l2_seasonal_thresholds.get(season, {}).get("per_pump", {})
        pump_seasonal = season_pump_map.get(str(pump_id))
        if pump_seasonal and not pump_seasonal.get("fallback_to_global", False):
            # Seasonal JSON uses "warning"/"alarm" keys; keep global training_mean for health calc.
            s_warning = float(pump_seasonal.get("warning", warning))
            s_alarm = float(pump_seasonal.get("alarm", alarm))
            if s_alarm < s_warning:
                s_alarm = s_warning
            return s_warning, s_alarm, training_mean

    return warning, alarm, training_mean


def _resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    if resolved.type == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise ValueError("MPS requested but not available")
    return resolved


def _load_hparams(hparams_path: Path) -> dict[str, Any]:
    if hparams_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load hparams.yaml") from exc
        with open(hparams_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return dict(data)

    with open(hparams_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data)


def _load_l2_model_from_checkpoint(
    checkpoint_path: Path,
    hparams: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if state_dict is None:
        raise KeyError(f"Checkpoint missing state_dict: {checkpoint_path}")

    filtered: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("encoder.logvar.") or key.startswith("model.encoder.logvar."):
            continue
        if key.startswith("encoder.") or key.startswith("decoder."):
            filtered[key] = value
        elif key.startswith("model.encoder."):
            filtered[key.replace("model.", "", 1)] = value
        elif key.startswith("model.decoder."):
            filtered[key.replace("model.", "", 1)] = value

    if not filtered:
        raise KeyError(
            f"No encoder/decoder weights found in checkpoint: {checkpoint_path}"
        )

    model = _InferenceVAE(hparams)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        logger.warning("L2 checkpoint missing %d keys during load.", len(missing))
    if unexpected:
        logger.warning(
            "L2 checkpoint had %d unexpected keys during load.", len(unexpected)
        )

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def create_streaming_detector(**overrides) -> StreamingEnsembleDetector:
    """Load production assets and return a ready-to-use streaming detector."""
    workspace_root = Path(__file__).resolve().parent.parent.parent

    device_name = overrides.get("device", "auto")
    ema_alpha = float(overrides.get("ema_alpha", 0.3))

    l1_weights_dir = Path(
        overrides.get(
            "l1_weights_dir",
            workspace_root / "ensemble" / "cond_reg_v2" / "model" / "weights",
        )
    )
    l1_detector = overrides.get("l1_detector")
    if l1_detector is None:
        l1_detector = Level1Detector(weights_dir=l1_weights_dir, device=device_name)

    l1_thresholds = overrides.get("l1_thresholds")
    if l1_thresholds is None:
        l1_thresholds_path = Path(
            overrides.get(
                "l1_thresholds_path",
                workspace_root
                / "ensemble"
                / "cond_reg_v2"
                / "model"
                / "weights"
                / "production_thresholds.json",
            )
        )
        with open(l1_thresholds_path, "r", encoding="utf-8") as f:
            l1_thresholds = json.load(f)

    l2_hparams = overrides.get("l2_hparams")
    if l2_hparams is None:
        l2_hparams_path = Path(
            overrides.get(
                "l2_hparams_path",
                workspace_root / "bin" / "final_metrics" / "hparams.yaml",
            )
        )
        l2_hparams = _load_hparams(l2_hparams_path)

    l2_norm_params = overrides.get("l2_norm_params")
    if l2_norm_params is None:
        l2_norm_params_path = Path(
            overrides.get(
                "l2_norm_params_path",
                workspace_root / "bin" / "final_metrics" / "norm_params.json",
            )
        )
        with open(l2_norm_params_path, "r", encoding="utf-8") as f:
            l2_norm_params = json.load(f)

    l2_thresholds = overrides.get("l2_thresholds")
    if l2_thresholds is None:
        l2_thresholds_path = Path(
            overrides.get(
                "l2_thresholds_path",
                workspace_root / "bin" / "final_metrics" / "production_thresholds.json",
            )
        )
        with open(l2_thresholds_path, "r", encoding="utf-8") as f:
            l2_thresholds = json.load(f)

    l2_preprocessor = overrides.get("l2_preprocessor")
    if l2_preprocessor is None:
        import sys

        if str(workspace_root) not in sys.path:
            sys.path.insert(0, str(workspace_root))
        from bin.model.preprocessing import Preprocessor

        l2_preprocessor = Preprocessor(Namespace(**l2_hparams))

    l2_model = overrides.get("l2_model")
    if l2_model is None:
        l2_weights_path = Path(
            overrides.get(
                "l2_weights_path",
                workspace_root / "bin" / "final_metrics" / "model_weights.ckpt",
            )
        )
        torch_device = _resolve_torch_device(str(device_name))
        l2_model = _load_l2_model_from_checkpoint(
            l2_weights_path, l2_hparams, torch_device
        )

    min_session_alert_steps = int(overrides.get("min_session_alert_steps", 3))
    min_alarm_duration = int(overrides.get("min_alarm_duration", 2))
    alarm_rate_limit_seconds = float(overrides.get("alarm_rate_limit_seconds", 3600.0))

    l2_seasonal_thresholds = overrides.get("l2_seasonal_thresholds")
    if l2_seasonal_thresholds is None:
        l2_seasonal_path = Path(
            overrides.get(
                "l2_seasonal_thresholds_path",
                workspace_root
                / "bin"
                / "final_metrics"
                / "production_thresholds_seasonal_l2.json",
            )
        )
        if l2_seasonal_path.exists():
            with open(l2_seasonal_path, "r", encoding="utf-8") as f:
                l2_seasonal_thresholds = json.load(f)

    return StreamingEnsembleDetector(
        l1_detector=l1_detector,
        l2_model=l2_model,
        l2_preprocessor=l2_preprocessor,
        l2_norm_params=l2_norm_params,
        l2_hparams=l2_hparams,
        l1_thresholds=l1_thresholds,
        l2_thresholds=l2_thresholds,
        ema_alpha=ema_alpha,
        min_session_alert_steps=min_session_alert_steps,
        min_alarm_duration=min_alarm_duration,
        alarm_rate_limit_seconds=alarm_rate_limit_seconds,
        l2_seasonal_thresholds=l2_seasonal_thresholds,
    )


__all__ = [
    "StreamingTimestepResult",
    "StreamingEnsembleDetector",
    "create_streaming_detector",
]
