from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class AlarmMonitor:
    """Spike filter and rate limiter for one pump's streaming alarm states.

    Spike filter: demotes transient anomalous windows (fewer than
    ``min_alarm_duration`` consecutive windows) back to NORMAL, eliminating
    single-window flickers from noisy sensors.

    Rate limiter: tracks whether a *new* operator alert should fire.  While a
    pump remains in continuous ALARM the alert flag is True only once per
    ``rate_limit_seconds`` so operators are not flooded with repeated
    notifications for the same event.

    Design note — streaming vs batch K:
        The batch path uses K=3 (any 3+ WARNING/ALARM windows in a day, not
        necessarily consecutive) as the day-level detection criterion.  This
        filter requires ``min_alarm_duration`` *consecutive* anomalous windows
        before propagating the status.  The two invariants are intentionally
        different: batch K is optimised for recall over a full day of data;
        the streaming filter is optimised for precision in real-time, where
        consecutive elevation is a stronger noise-rejection signal than
        scattered windows.
    """

    def __init__(self, min_alarm_duration: int = 2, rate_limit_seconds: float = 3600.0) -> None:
        self.min_alarm_duration = int(min_alarm_duration)
        self.rate_limit_seconds = float(rate_limit_seconds)
        self._consecutive_anomalous: int = 0
        self._last_alert_epoch: float | None = None
        self._in_sustained_alarm: bool = False

    def check(self, raw_status: str, timestamp) -> tuple[str, bool]:
        """Apply spike filter then rate limiter.

        Parameters
        ----------
        raw_status:
            Fused ensemble status from the scoring layer
            (``"NORMAL"`` / ``"WARNING"`` / ``"ALARM"``).
        timestamp:
            Anything that ``pd.Timestamp`` accepts.

        Returns
        -------
        filtered_status:
            Spike-filtered status.  Transient spikes shorter than
            ``min_alarm_duration`` windows are reported as ``"NORMAL"``.
        alert_fired:
            ``True`` when this window is the first new ALARM notification
            after the rate-limit window has elapsed.
        """
        ts_epoch = float(pd.Timestamp(timestamp).timestamp())

        if raw_status not in ("WARNING", "ALARM"):
            self._consecutive_anomalous = 0
            if self._in_sustained_alarm:
                # Alarm broken — clear epoch so next alarm fires immediately
                self._last_alert_epoch = None
            self._in_sustained_alarm = False
            return raw_status, False

        self._consecutive_anomalous += 1

        if self._consecutive_anomalous < self.min_alarm_duration:
            return "NORMAL", False

        alert_fired = False
        if raw_status == "ALARM":
            if (
                self._last_alert_epoch is None
                or (ts_epoch - self._last_alert_epoch) >= self.rate_limit_seconds
            ):
                alert_fired = True
                self._last_alert_epoch = ts_epoch
            self._in_sustained_alarm = True

        return raw_status, alert_fired

    def reset(self) -> None:
        """Reset all counters (call when a pump session restarts)."""
        self._consecutive_anomalous = 0
        self._last_alert_epoch = None
        self._in_sustained_alarm = False


@dataclass
class DailyScoreSummary:
    date: str
    pump_id: int
    l1_mean_mahalanobis: float
    l2_mean_smoothed_mse: float | None
    season: str
    day_status: str


@dataclass
class DriftAlert:
    pump_id: int
    metric: str
    rolling_mean: float
    training_mean: float
    drift_percent: float
    alert_date: str
    severity: str


@dataclass
class RecalibrationRecord:
    date: str
    reason: str
    old_thresholds: dict
    new_thresholds: dict
    delta_report: dict
    applied: bool


class DriftMonitor:
    CSV_COLUMNS = [
        "date",
        "pump_id",
        "l1_mean_mahalanobis",
        "l2_mean_smoothed_mse",
        "season",
        "day_status",
    ]

    def __init__(
        self,
        l1_thresholds_path: Path,
        l2_thresholds_path: Path,
        log_dir: Path,
        rolling_window: int = 30,
        drift_warning_pct: float = 0.15,
        drift_critical_pct: float = 0.30,
    ):
        self.l1_thresholds_path = Path(l1_thresholds_path)
        self.l2_thresholds_path = Path(l2_thresholds_path)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.daily_log_path = self.log_dir / "daily_scores.csv"
        self.rolling_window = rolling_window
        self.drift_warning_pct = drift_warning_pct
        self.drift_critical_pct = drift_critical_pct

        self._training_means = self._load_training_means()

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Threshold file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_training_means(self) -> dict[str, float]:
        l1_thresholds = self._load_json(self.l1_thresholds_path)
        l2_thresholds = self._load_json(self.l2_thresholds_path)

        l1_mean = float(l1_thresholds.get("global", {}).get("mahalanobis", {}).get("mean"))
        l2_mean = float(l2_thresholds.get("global", {}).get("mean"))

        return {
            "l1_mahalanobis": l1_mean,
            "l2_smoothed_mse": l2_mean,
        }

    def _ensure_log_exists(self) -> None:
        if self.daily_log_path.exists():
            return
        empty_df = pd.DataFrame(columns=self.CSV_COLUMNS)
        empty_df.to_csv(self.daily_log_path, index=False)

    def _load_daily_log(self) -> pd.DataFrame:
        self._ensure_log_exists()
        df = pd.read_csv(self.daily_log_path)
        if df.empty:
            return df

        for column in self.CSV_COLUMNS:
            if column not in df.columns:
                raise ValueError(f"Missing required column '{column}' in daily log")

        df["pump_id"] = df["pump_id"].astype(int)
        df["l1_mean_mahalanobis"] = pd.to_numeric(df["l1_mean_mahalanobis"], errors="coerce")
        df["l2_mean_smoothed_mse"] = pd.to_numeric(df["l2_mean_smoothed_mse"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "l1_mean_mahalanobis"]).copy()
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df

    @staticmethod
    def _is_normal_day(day_status: str) -> bool:
        return str(day_status).strip().lower() == "normal"

    @staticmethod
    def _compute_drift_percent(rolling_mean: float, training_mean: float) -> float:
        if np.isclose(training_mean, 0.0):
            return 0.0 if np.isclose(rolling_mean, 0.0) else np.inf
        return (rolling_mean - training_mean) / training_mean

    def _alerts_for_group(self, pump_id: int, normal_group: pd.DataFrame) -> list[DriftAlert]:
        alerts: list[DriftAlert] = []
        if len(normal_group) < 15:
            return alerts

        tail = normal_group.tail(self.rolling_window)
        last_date = str(tail["date"].max())

        metrics = [
            ("l1_mahalanobis", "l1_mean_mahalanobis"),
            ("l2_smoothed_mse", "l2_mean_smoothed_mse"),
        ]

        for metric_name, column_name in metrics:
            series = pd.to_numeric(tail[column_name], errors="coerce").dropna()
            if series.empty:
                continue

            rolling_mean = float(series.mean())
            training_mean = float(self._training_means[metric_name])
            drift_percent = self._compute_drift_percent(rolling_mean, training_mean)
            abs_drift = abs(drift_percent)

            if abs_drift >= self.drift_critical_pct:
                severity = "CRITICAL"
            elif abs_drift >= self.drift_warning_pct:
                severity = "WARNING"
            else:
                continue

            alerts.append(
                DriftAlert(
                    pump_id=pump_id,
                    metric=metric_name,
                    rolling_mean=rolling_mean,
                    training_mean=training_mean,
                    drift_percent=drift_percent,
                    alert_date=last_date,
                    severity=severity,
                )
            )

        return alerts

    def log_day(self, summary: DailyScoreSummary) -> list[DriftAlert]:
        self._ensure_log_exists()
        row = pd.DataFrame(
            [
                {
                    "date": summary.date,
                    "pump_id": summary.pump_id,
                    "l1_mean_mahalanobis": summary.l1_mean_mahalanobis,
                    "l2_mean_smoothed_mse": summary.l2_mean_smoothed_mse,
                    "season": summary.season,
                    "day_status": summary.day_status,
                }
            ]
        )
        row.to_csv(self.daily_log_path, mode="a", header=False, index=False)

        return self.check_drift(pump_id=summary.pump_id)

    def check_drift(self, pump_id: int | None = None) -> list[DriftAlert]:
        df = self._load_daily_log()
        if df.empty:
            return []

        normal_df = df[df["day_status"].apply(self._is_normal_day)].copy()
        if normal_df.empty:
            return []

        normal_df = normal_df.sort_values(by=["pump_id", "date"])

        alerts: list[DriftAlert] = []
        grouped = normal_df.groupby("pump_id")
        for grouped_pump_id, group in grouped:
            if pump_id is not None and int(grouped_pump_id) != int(pump_id):
                continue
            alerts.extend(self._alerts_for_group(int(grouped_pump_id), group))

        return alerts

    def get_rolling_stats(self, pump_id: int | None = None) -> dict:
        df = self._load_daily_log()
        if df.empty:
            return {}

        normal_df = df[df["day_status"].apply(self._is_normal_day)].copy()
        if normal_df.empty:
            return {}

        normal_df = normal_df.sort_values(by=["pump_id", "date"])
        stats: dict[int, dict[str, Any]] = {}

        for grouped_pump_id, group in normal_df.groupby("pump_id"):
            pid = int(grouped_pump_id)
            if pump_id is not None and pid != int(pump_id):
                continue

            tail = group.tail(self.rolling_window)
            l1_series = pd.to_numeric(tail["l1_mean_mahalanobis"], errors="coerce").dropna()
            l2_series = pd.to_numeric(tail["l2_mean_smoothed_mse"], errors="coerce").dropna()

            stats[pid] = {
                "l1_rolling_mean": float(l1_series.mean()) if not l1_series.empty else None,
                "l2_rolling_mean": float(l2_series.mean()) if not l2_series.empty else None,
                "n_normal_days": int(len(group)),
                "last_date": str(group["date"].max()),
            }

        return stats

    def get_daily_log(self) -> pd.DataFrame:
        return self._load_daily_log()


class RecalibrationTracker:
    def __init__(self, state_dir: Path, quarterly_months: tuple[int, ...] = (1, 4, 7, 10)):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "recalibration_state.json"
        self.quarterly_months = quarterly_months
        self._state = self._load_or_init_state()

    @staticmethod
    def _default_state() -> dict[str, Any]:
        return {
            "last_recalibration_date": None,
            "last_reason": None,
            "history": [],
        }

    def _load_or_init_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            state = self._default_state()
            self._write_state(state)
            return state

        with self.state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        for key, default_value in self._default_state().items():
            state.setdefault(key, default_value)

        return state

    def _write_state(self, state: dict[str, Any]) -> None:
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def should_recalibrate(
        self,
        current_date: date | None = None,
        drift_alerts: list[DriftAlert] | None = None,
        post_maintenance: bool = False,
    ) -> tuple[bool, str]:
        now = current_date or date.today()
        alerts = drift_alerts or []

        if post_maintenance:
            return True, "post_maintenance"

        if alerts:
            return True, "drift_alert"

        if now.month in self.quarterly_months:
            last_str = self._state.get("last_recalibration_date")
            if not last_str:
                return True, "quarterly"

            try:
                last_dt = datetime.strptime(last_str, "%Y-%m-%d").date()
            except ValueError:
                return True, "quarterly"

            if (now - last_dt).days > 85:
                return True, "quarterly"

        return False, "none"

    def record_recalibration(self, record: RecalibrationRecord) -> None:
        self._state["last_recalibration_date"] = record.date
        self._state["last_reason"] = record.reason
        self._state.setdefault("history", []).append(asdict(record))
        self._write_state(self._state)

    @staticmethod
    def _load_thresholds(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Threshold file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _extract_pump_metrics(pump_data: dict[str, Any], preferred_keys: list[str]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key in preferred_keys:
            value = pump_data.get(key)
            if isinstance(value, (int, float)):
                metrics[key] = float(value)

        if not metrics:
            for key, value in pump_data.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)

        return metrics

    def compare_thresholds(self, old_path: Path, new_path: Path) -> dict:
        old_thresholds = self._load_thresholds(Path(old_path))
        new_thresholds = self._load_thresholds(Path(new_path))

        old_per_pump = old_thresholds.get("per_pump", {})
        new_per_pump = new_thresholds.get("per_pump", {})

        preferred_metric_keys = [
            "warning",
            "alarm",
            "mean",
            "day_warning",
            "day_alarm",
            "window_warning_smoothed",
            "window_alarm_smoothed",
        ]

        report: dict[str, dict[str, dict[str, float]]] = {}
        pump_ids = sorted(set(old_per_pump.keys()) | set(new_per_pump.keys()), key=lambda x: int(x))

        for pump_id in pump_ids:
            old_metrics = self._extract_pump_metrics(old_per_pump.get(pump_id, {}), preferred_metric_keys)
            new_metrics = self._extract_pump_metrics(new_per_pump.get(pump_id, {}), preferred_metric_keys)

            metric_keys = sorted(set(old_metrics.keys()) | set(new_metrics.keys()))
            pump_report: dict[str, dict[str, float]] = {}

            for metric in metric_keys:
                old_val = float(old_metrics.get(metric, np.nan))
                new_val = float(new_metrics.get(metric, np.nan))

                if np.isnan(old_val) or np.isclose(old_val, 0.0):
                    pct_change = np.inf if not np.isnan(new_val) and not np.isclose(new_val, 0.0) else 0.0
                else:
                    pct_change = (new_val - old_val) / old_val

                pump_report[metric] = {
                    "old": old_val,
                    "new": new_val,
                    "pct_change": float(pct_change),
                }

            report[str(pump_id)] = pump_report

        return report

    def should_apply(self, delta_report: dict, threshold_pct: float = 0.05) -> bool:
        for pump_metrics in delta_report.values():
            for metric_delta in pump_metrics.values():
                pct_change = float(metric_delta.get("pct_change", 0.0))
                if np.isfinite(pct_change) and abs(pct_change) > threshold_pct:
                    return True
                if np.isinf(pct_change):
                    return True
        return False

    def get_history(self) -> list[RecalibrationRecord]:
        history = self._state.get("history", [])
        return [RecalibrationRecord(**record) for record in history]


class SeasonalTracker:
    SEASONS = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
    }
    MIN_DAYS_FOR_ANALYSIS = 10

    def __init__(self, log_dir: Path, baseline_path: Path | None = None):
        self.log_dir = Path(log_dir)
        self.log_path = self.log_dir / "daily_scores.csv"
        self.baseline_path = Path(baseline_path) if baseline_path is not None else None
        self._baseline = self._load_baseline()

    def _load_baseline(self) -> dict[str, Any]:
        if self.baseline_path is None or not self.baseline_path.exists():
            return {}

        with self.baseline_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_normal_log(self) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame(
                columns=[
                    "date",
                    "pump_id",
                    "l1_mean_mahalanobis",
                    "l2_mean_smoothed_mse",
                    "season",
                    "day_status",
                ]
            )

        df = pd.read_csv(self.log_path)
        if df.empty:
            return df

        df["pump_id"] = pd.to_numeric(df["pump_id"], errors="coerce")
        df["l1_mean_mahalanobis"] = pd.to_numeric(df["l1_mean_mahalanobis"], errors="coerce")
        df["l2_mean_smoothed_mse"] = pd.to_numeric(df["l2_mean_smoothed_mse"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        df = df.dropna(subset=["date", "pump_id", "l1_mean_mahalanobis"]).copy()
        df["pump_id"] = df["pump_id"].astype(int)
        df["season"] = df["season"].fillna("").astype(str).str.lower()

        empty_season = df["season"].eq("")
        if empty_season.any():
            df.loc[empty_season, "season"] = df.loc[empty_season, "date"].dt.date.apply(self.tag_season)

        normal_mask = df["day_status"].astype(str).str.strip().str.lower().eq("normal")
        return df[normal_mask].copy()

    @staticmethod
    def tag_season(d: date) -> str:
        month = d.month
        for season, months in SeasonalTracker.SEASONS.items():
            if month in months:
                return season
        raise ValueError(f"Invalid month for date: {d}")

    def get_seasonal_stats(self, pump_id: int | None = None) -> dict:
        df = self._load_normal_log()
        if df.empty:
            return {}

        if pump_id is not None:
            df = df[df["pump_id"] == int(pump_id)]
            if df.empty:
                return {}

        stats: dict[int, dict[str, dict[str, float | int | None]]] = {}

        grouped = df.groupby(["pump_id", "season"])
        for (pid, season), group in grouped:
            stats.setdefault(int(pid), {})[season] = {
                "l1_mean": float(group["l1_mean_mahalanobis"].mean()),
                "l1_std": float(group["l1_mean_mahalanobis"].std(ddof=0)),
                "l2_mean": float(group["l2_mean_smoothed_mse"].mean()) if group["l2_mean_smoothed_mse"].notna().any() else None,
                "l2_std": float(group["l2_mean_smoothed_mse"].std(ddof=0)) if group["l2_mean_smoothed_mse"].notna().any() else None,
                "n_days": int(len(group)),
            }

        return stats

    def _get_baseline_for(self, pump_id: int, season: str) -> dict[str, float] | None:
        pump_baseline = self._baseline.get(str(pump_id), {})
        season_baseline = pump_baseline.get(season)
        if isinstance(season_baseline, dict):
            return season_baseline
        return None

    def check_seasonal_deviation(
        self,
        pump_id: int,
        current_season: str | None = None,
        sigma_threshold: float = 2.0,
    ) -> list[str]:
        season = current_season or self.tag_season(date.today())
        stats = self.get_seasonal_stats(pump_id=pump_id)
        if int(pump_id) not in stats:
            return [f"Pump {pump_id}: no normal-day data available."]

        pump_stats = stats[int(pump_id)]
        season_stats = pump_stats.get(season)
        if not season_stats:
            return [f"Pump {pump_id}: no data for season '{season}'."]

        if int(season_stats["n_days"]) < self.MIN_DAYS_FOR_ANALYSIS:
            return [
                (
                    f"Pump {pump_id} season '{season}': insufficient data "
                    f"({season_stats['n_days']} days < {self.MIN_DAYS_FOR_ANALYSIS})."
                )
            ]

        baseline = self._get_baseline_for(int(pump_id), season)
        if baseline is None:
            baseline = {
                "l1_mean": float(np.mean([v["l1_mean"] for v in pump_stats.values()])),
                "l1_std": float(np.std([v["l1_mean"] for v in pump_stats.values()])),
                "l2_mean": float(np.mean([v["l2_mean"] for v in pump_stats.values() if v["l2_mean"] is not None]))
                if any(v["l2_mean"] is not None for v in pump_stats.values())
                else None,
                "l2_std": float(np.std([v["l2_mean"] for v in pump_stats.values() if v["l2_mean"] is not None]))
                if any(v["l2_mean"] is not None for v in pump_stats.values())
                else None,
            }

        warnings: list[str] = []

        l1_std = float(baseline.get("l1_std", 0.0) or 0.0)
        if l1_std > 0:
            z_l1 = (float(season_stats["l1_mean"]) - float(baseline["l1_mean"])) / l1_std
            if abs(z_l1) >= sigma_threshold:
                warnings.append(
                    f"Pump {pump_id} {season} L1 mean deviates by {z_l1:.2f} sigma from baseline."
                )

        baseline_l2_mean = baseline.get("l2_mean")
        baseline_l2_std = baseline.get("l2_std")
        if baseline_l2_mean is not None and baseline_l2_std is not None and float(baseline_l2_std) > 0 and season_stats["l2_mean"] is not None:
            z_l2 = (float(season_stats["l2_mean"]) - float(baseline_l2_mean)) / float(baseline_l2_std)
            if abs(z_l2) >= sigma_threshold:
                warnings.append(
                    f"Pump {pump_id} {season} L2 mean deviates by {z_l2:.2f} sigma from baseline."
                )

        return warnings

    def promote(
        self,
        l1_seasonal_thresholds: dict,
        l2_seasonal_thresholds: dict,
        output_dir=None,
    ) -> tuple:
        """Write L1 and L2 seasonal threshold dicts to JSON files.

        Parameters
        ----------
        l1_seasonal_thresholds:
            Dict produced by ThresholdCalibrator.calibrate_seasonal().
        l2_seasonal_thresholds:
            Dict with per-season per-pump L2 warning/alarm thresholds.
        output_dir:
            Directory to write files. Defaults to baseline_path.parent if set,
            else log_dir.

        Returns
        -------
        (l1_path, l2_path) as Path objects.
        """
        required_seasons = {"winter", "spring", "summer", "autumn"}
        missing_l1 = required_seasons - set(l1_seasonal_thresholds.keys())
        missing_l2 = required_seasons - set(l2_seasonal_thresholds.keys())
        missing = missing_l1 | missing_l2
        if missing:
            raise ValueError(f"Seasonal threshold dicts missing season keys: {sorted(missing)}")

        if output_dir is not None:
            out_dir = Path(output_dir)
        elif self.baseline_path is not None:
            out_dir = self.baseline_path.parent
        else:
            out_dir = self.log_dir

        out_dir.mkdir(parents=True, exist_ok=True)

        l1_path = out_dir / "production_thresholds_seasonal.json"
        l2_path = out_dir / "production_thresholds_seasonal_l2.json"

        with l1_path.open("w", encoding="utf-8") as f:
            json.dump(l1_seasonal_thresholds, f, indent=2)
        with l2_path.open("w", encoding="utf-8") as f:
            json.dump(l2_seasonal_thresholds, f, indent=2)

        return l1_path, l2_path

    def generate_seasonal_report(self) -> str:
        stats = self.get_seasonal_stats()
        if not stats:
            return "# Seasonal Monitoring Report\n\nNo normal-day data available."

        lines = ["# Seasonal Monitoring Report", ""]
        for pump_id in sorted(stats.keys()):
            lines.append(f"## Pump {pump_id}")
            lines.append("")
            lines.append("| Season | L1 Mean | L1 Std | L2 Mean | L2 Std | N Days |")
            lines.append("|---|---:|---:|---:|---:|---:|")

            for season in ["winter", "spring", "summer", "autumn"]:
                season_stats = stats[pump_id].get(season)
                if season_stats is None:
                    lines.append(f"| {season} | - | - | - | - | 0 |")
                    continue

                l2_mean = "-" if season_stats["l2_mean"] is None else f"{season_stats['l2_mean']:.6f}"
                l2_std = "-" if season_stats["l2_std"] is None else f"{season_stats['l2_std']:.6f}"
                lines.append(
                    (
                        f"| {season} | {season_stats['l1_mean']:.6f} | {season_stats['l1_std']:.6f} | "
                        f"{l2_mean} | {l2_std} | {season_stats['n_days']} |"
                    )
                )

            lines.append("")

        return "\n".join(lines)


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble monitoring and drift detection")
    subparsers = parser.add_subparsers(dest="command")

    drift_parser = subparsers.add_parser("check-drift", help="Check drift status")
    drift_parser.add_argument("--pump-id", type=int, default=None)
    drift_parser.add_argument("--log-dir", type=Path, default=Path("monitoring_logs"))
    drift_parser.add_argument("--l1-thresholds", type=Path, required=True)
    drift_parser.add_argument("--l2-thresholds", type=Path, required=True)

    recal_parser = subparsers.add_parser("check-recalibration", help="Check if recalibration needed")
    recal_parser.add_argument("--state-dir", type=Path, default=Path("monitoring_logs"))
    recal_parser.add_argument("--post-maintenance", action="store_true")

    seasonal_parser = subparsers.add_parser("seasonal-report", help="Show seasonal distributions")
    seasonal_parser.add_argument("--log-dir", type=Path, default=Path("monitoring_logs"))
    seasonal_parser.add_argument("--pump-id", type=int, default=None)

    log_parser = subparsers.add_parser("log-day", help="Log a day's scores")
    log_parser.add_argument("--date", type=str, required=True)
    log_parser.add_argument("--pump-id", type=int, required=True)
    log_parser.add_argument("--l1-score", type=float, required=True)
    log_parser.add_argument("--l2-score", type=float, default=None)
    log_parser.add_argument("--status", type=str, required=True)
    log_parser.add_argument("--log-dir", type=Path, default=Path("monitoring_logs"))
    log_parser.add_argument("--l1-thresholds", type=Path, required=True)
    log_parser.add_argument("--l2-thresholds", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "check-drift":
        monitor = DriftMonitor(
            l1_thresholds_path=args.l1_thresholds,
            l2_thresholds_path=args.l2_thresholds,
            log_dir=args.log_dir,
        )
        alerts = monitor.check_drift(pump_id=args.pump_id)
        stats = monitor.get_rolling_stats(pump_id=args.pump_id)

        print("Rolling stats:")
        print(json.dumps(stats, indent=2))
        print("\nDrift alerts:")
        print(json.dumps([asdict(alert) for alert in alerts], indent=2))
        return

    if args.command == "check-recalibration":
        tracker = RecalibrationTracker(state_dir=args.state_dir)
        should_do, reason = tracker.should_recalibrate(post_maintenance=args.post_maintenance)
        print(json.dumps({"should_recalibrate": should_do, "reason": reason}, indent=2))
        return

    if args.command == "seasonal-report":
        seasonal = SeasonalTracker(log_dir=args.log_dir)
        if args.pump_id is not None:
            warnings = seasonal.check_seasonal_deviation(pump_id=args.pump_id)
            print(json.dumps({"pump_id": args.pump_id, "warnings": warnings}, indent=2))
        print(seasonal.generate_seasonal_report())
        return

    if args.command == "log-day":
        log_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        monitor = DriftMonitor(
            l1_thresholds_path=args.l1_thresholds,
            l2_thresholds_path=args.l2_thresholds,
            log_dir=args.log_dir,
        )
        summary = DailyScoreSummary(
            date=log_date.isoformat(),
            pump_id=args.pump_id,
            l1_mean_mahalanobis=args.l1_score,
            l2_mean_smoothed_mse=args.l2_score,
            season=SeasonalTracker.tag_season(log_date),
            day_status=args.status,
        )

        alerts = monitor.log_day(summary)
        print(json.dumps({"logged": asdict(summary), "alerts": [asdict(a) for a in alerts]}, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    cli_main()
