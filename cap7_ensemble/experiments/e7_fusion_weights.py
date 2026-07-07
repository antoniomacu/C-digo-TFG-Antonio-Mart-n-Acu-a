"""Experiment E7: Fusion Weight Optimization (RD-3).

Question:
    What are the optimal corroboration factor α and health weight w in the
    ensemble fusion layer, replacing the hardcoded constants in scoring.py?

Fusion formulas being tuned:
    severity     = max(s1, s2) + α * min(s1, s2)   [currently α = 0.3]
    health_ens   = w * min(h1, h2) + (1-w) * max(h1, h2)  [currently w = 0.6]

Where:
    s1 = mean_mahal / l1_alarm_threshold  (L1 normalized severity, clamped [0,2])
    s2 = mean_mse / l2_alarm_threshold    (L2 normalized severity, clamped [0,2])
    h1, h2 = health scores from compute_health_score()

Strategy:
    Phase 1 — Collect per-day (s1, s2, h1, h2) for all 801 normal and 1033
              abnormal days. This is the expensive inference step (done once).
    Phase 2 — Grid search over α ∈ [0.0, 0.6] step 0.05 and
              w ∈ [0.4, 0.8] step 0.05. No re-inference needed.
    Phase 3 — Report best params, delta vs current, heatmap.

Usage:
    From project root:
        uv run --project ensemble python experiments/e7_fusion_weights.py

    With explicit paths:
        uv run --project ensemble python experiments/e7_fusion_weights.py \\
            --train-path /path/to/new_data/train \\
            --test-path /path/to/new_data/test \\
            --l1-weights ensemble/cond_reg_v2/model/weights \\
            --l2-version bin/final_metrics \\
            --output experiments/E7_fusion_weights_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent  # experiments/
PROJECT_ROOT = SCRIPT_DIR.parent  # unsupervised_learning/
ENSEMBLE_DIR = PROJECT_ROOT / "ensemble"
BIN_DIR = PROJECT_ROOT / "bin"

L1_WEIGHTS_DEFAULT = ENSEMBLE_DIR / "cond_reg_v2" / "model" / "weights"
L2_VERSION_DEFAULT = BIN_DIR / "final_metrics"

NEW_DATA_TRAIN_DEFAULT = Path("<PATH_TO_DATA_DIR>/train")
NEW_DATA_TEST_DEFAULT = Path("<PATH_TO_DATA_DIR>/test")

OUTPUT_DEFAULT = SCRIPT_DIR / "E7_fusion_weights_results.json"

# Current production constants (baseline)
CURRENT_ALPHA = 0.3
CURRENT_HEALTH_WEIGHT = 0.6


# ---------------------------------------------------------------------------
# L1 model loader
# ---------------------------------------------------------------------------


def _import_l1_predictor(weights_dir: Path):
    """Load PumpPredictor (L1 digital twin), injecting ensemble on sys.path."""
    try:
        from cond_reg_v2.model.inference import PumpPredictor  # type: ignore[import]
    except ImportError:
        sys.path.insert(0, str(ENSEMBLE_DIR))
        from cond_reg_v2.model.inference import PumpPredictor  # type: ignore[import]
    return PumpPredictor(weights_dir=str(weights_dir))


# ---------------------------------------------------------------------------
# L2 model loader
# ---------------------------------------------------------------------------


def _import_l2_detector(version_dir: Path):
    """Load ProductionDetector (L2 VAE) from bin/model, evicting ensemble/model from cache first."""
    # ensemble/model shadows bin/model because ensemble/ is on sys.path via the project venv.
    # Fix: purge all 'model' and 'model.*' entries from sys.modules, prepend BIN_DIR, then import.
    to_purge = [k for k in sys.modules if k == "model" or k.startswith("model.")]
    for k in to_purge:
        del sys.modules[k]

    if str(BIN_DIR) not in sys.path:
        sys.path.insert(0, str(BIN_DIR))

    from model.inference import ProductionDetector  # type: ignore[import]

    return ProductionDetector(version_dir=str(version_dir))


# ---------------------------------------------------------------------------
# L1 thresholds helper
# ---------------------------------------------------------------------------


def _load_l1_thresholds(weights_dir: Path) -> dict:
    """Load production_thresholds.json for L1 (Mahalanobis)."""
    path = weights_dir / "production_thresholds.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _l1_alarm_threshold(thresholds: dict, pump_id: int) -> float:
    """Return L1 Mahalanobis alarm threshold for pump_id (per-pump or global)."""
    per_pump = thresholds.get("per_pump", {})
    key = str(pump_id)
    if key in per_pump:
        maha = per_pump[key].get("mahalanobis", {})
        return float(maha.get("alarm", thresholds["global"]["mahalanobis"]["alarm"]))
    return float(thresholds["global"]["mahalanobis"]["alarm"])


def _l1_training_mean(thresholds: dict, pump_id: int) -> float:
    """Return L1 Mahalanobis training mean for pump_id (for health score)."""
    per_pump = thresholds.get("per_pump", {})
    key = str(pump_id)
    if key in per_pump:
        maha = per_pump[key].get("mahalanobis", {})
        return float(maha.get("mean", thresholds["global"]["mahalanobis"]["mean"]))
    return float(thresholds["global"]["mahalanobis"]["mean"])


# ---------------------------------------------------------------------------
# L2 thresholds helper
# ---------------------------------------------------------------------------


def _load_l2_thresholds(version_dir: Path) -> dict:
    """Load production_thresholds.json for L2 (VAE MSE)."""
    path = version_dir / "production_thresholds.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _l2_alarm_threshold(thresholds: dict, pump_id: int) -> float:
    """Return L2 MSE alarm threshold for pump_id (per-pump or global)."""
    per_pump = thresholds.get("per_pump", {})
    key = str(pump_id)
    if key in per_pump:
        return float(per_pump[key].get("alarm", thresholds["global"]["alarm"]))
    return float(thresholds["global"]["alarm"])


def _l2_training_mean(thresholds: dict, pump_id: int) -> float:
    """Return L2 MSE training mean for pump_id (for health score)."""
    per_pump = thresholds.get("per_pump", {})
    key = str(pump_id)
    if key in per_pump:
        return float(per_pump[key].get("mean", thresholds["global"]["mean"]))
    return float(thresholds["global"]["mean"])


# ---------------------------------------------------------------------------
# Scoring helpers (replicate scoring.py formulas without import dependency)
# ---------------------------------------------------------------------------


def _normalize_severity(score: float, alarm_threshold: float) -> float:
    """Normalize score to [0, 2]. Mirrors scoring.normalize_severity()."""
    if alarm_threshold <= 0:
        return 0.0
    norm = score / alarm_threshold
    return float(np.clip(norm, 0.0, 2.0))


def _compute_health_score(
    score: float, training_mean: float, alarm_threshold: float
) -> float:
    """Inverted sigmoid health score 0-100. Mirrors scoring.compute_health_score()."""
    import math

    if alarm_threshold <= training_mean:
        return 100.0 if score <= training_mean else 0.0
    x = (score - training_mean) / (alarm_threshold - training_mean)
    exponent = 3.0 * (x - 0.5)
    exponent = max(-20.0, min(20.0, float(exponent)))
    health = 100.0 / (1.0 + math.exp(exponent))
    return float(np.clip(health, 0.0, 100.0))


# ---------------------------------------------------------------------------
# EMA smoothing (matches L2 production code)
# ---------------------------------------------------------------------------


def _ema_smooth(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average — matches ProductionDetector._ema_smooth()."""
    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


# ---------------------------------------------------------------------------
# Residual computation (L1) — adapted from e2_mahalanobis_calibration.py
# ---------------------------------------------------------------------------


def _compute_l1_residuals(
    predictor,
    feature_names: list[str],
    csv_path: Path,
) -> pd.DataFrame | None:
    """Compute residual vectors for one CSV file via L1 predictor.

    Returns DataFrame with columns [timestamp, pump_id, *feature_names] or None on error.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"    [SKIP L1] {csv_path.name}: read error: {exc}")
        return None

    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        print(f"    [SKIP L1] {csv_path.name}: missing columns {missing[:3]}...")
        return None

    if "timestamp" in df.columns:
        timestamps = df["timestamp"].copy()
    else:
        timestamps = pd.Series(pd.RangeIndex(len(df)).astype(str), name="timestamp")

    if "pump_id" in df.columns:
        pump_id_series = pd.to_numeric(df["pump_id"], errors="coerce")
    else:
        match = re.search(r"pump[_-]?(\d+)", csv_path.stem, flags=re.IGNORECASE)
        if match:
            pump_id_series = pd.Series(
                np.full(len(df), int(match.group(1))), dtype=float
            )
        else:
            print(f"    [SKIP L1] {csv_path.name}: cannot infer pump_id")
            return None

    try:
        predictions = predictor.predict(df)
    except Exception as exc:
        print(f"    [SKIP L1] {csv_path.name}: prediction error: {exc}")
        return None

    actual = df[feature_names].apply(pd.to_numeric, errors="coerce")
    predicted = predictions[feature_names].apply(pd.to_numeric, errors="coerce")
    residual = actual - predicted

    frame = pd.DataFrame({"timestamp": timestamps, "pump_id": pump_id_series})
    frame[feature_names] = residual.values
    frame = frame.dropna(subset=["pump_id"] + feature_names)
    frame["pump_id"] = frame["pump_id"].astype(int)
    return frame


# ---------------------------------------------------------------------------
# L1 Mahalanobis scoring on residuals
# ---------------------------------------------------------------------------


def _mahalanobis_scores(
    R: np.ndarray,
    mu: np.ndarray,
    sigma_inv: np.ndarray,
) -> np.ndarray:
    """Compute Mahalanobis distance per sample. R shape (N, D)."""
    centered = R - mu
    mahal_sq = np.einsum("ij,jk,ik->i", centered, sigma_inv, centered)
    mahal_sq = np.clip(mahal_sq, 0.0, None)
    return np.sqrt(mahal_sq)


# ---------------------------------------------------------------------------
# Phase 1: collect per-day scores
# ---------------------------------------------------------------------------


def collect_day_scores(
    csv_files: list[Path],
    label: int,
    l1_predictor,
    l1_feature_names: list[str],
    l1_mu: np.ndarray,
    l1_sigma_inv: np.ndarray,
    l1_thresholds: dict,
    l2_detector,
    l2_thresholds: dict,
) -> list[dict]:
    """Run both models on each CSV and return per-day score records.

    Each record contains:
        pump_id, date, s1, s2, h1, h2, label

    s1 = mean_mahal / l1_alarm_threshold  (clamped [0, 2])
    s2 = mean_smoothed_mse / l2_alarm_threshold  (clamped [0, 2])
    h1 = L1 health score
    h2 = L2 health score
    """
    records = []
    n_files = len(csv_files)

    for idx, csv_path in enumerate(csv_files):
        if (idx + 1) % 100 == 0 or idx == 0:
            lbl_name = "normal" if label == 0 else "abnormal"
            print(f"    [{lbl_name}] {idx + 1}/{n_files} — {csv_path.name}")

        # ── Infer pump_id and date from filename ──────────────────────────
        m = re.search(
            r"pump[_-]?(\d+)[_-](\d{4}-\d{2}-\d{2})", csv_path.stem, flags=re.IGNORECASE
        )
        if m:
            pump_id = int(m.group(1))
            date_str = m.group(2)
        else:
            # Try reading from file
            try:
                tmp_df = pd.read_csv(csv_path, nrows=1)
                pump_id = (
                    int(tmp_df["pump_id"].iloc[0]) if "pump_id" in tmp_df.columns else 1
                )
                date_str = (
                    str(pd.to_datetime(tmp_df["timestamp"].iloc[0]).date())
                    if "timestamp" in tmp_df.columns
                    else "unknown"
                )
            except Exception:
                print(f"    [SKIP] {csv_path.name}: cannot infer pump_id/date")
                continue

        # ── L1 scoring ────────────────────────────────────────────────────
        s1 = None
        h1 = None
        residuals_df = _compute_l1_residuals(l1_predictor, l1_feature_names, csv_path)
        if residuals_df is not None and len(residuals_df) > 0:
            R = residuals_df[l1_feature_names].to_numpy(dtype=float)
            mahal_per_ts = _mahalanobis_scores(R, l1_mu, l1_sigma_inv)
            mean_mahal = float(np.mean(mahal_per_ts))
            l1_alarm = _l1_alarm_threshold(l1_thresholds, pump_id)
            l1_mean = _l1_training_mean(l1_thresholds, pump_id)
            s1 = _normalize_severity(mean_mahal, l1_alarm)
            h1 = _compute_health_score(mean_mahal, l1_mean, l1_alarm)

        # ── L2 scoring ────────────────────────────────────────────────────
        s2 = None
        h2 = None
        if l2_detector is not None:
            try:
                l2_result = l2_detector.classify([str(csv_path)])
                pump_results = l2_result.get("pump_results", [])
                if pump_results:
                    pr = pump_results[0]
                    # day_error_mse is the mean over all windows (pre-smoothed at window level)
                    # For consistency with E2 use mean of raw per-window MSEs = day_error_mse
                    day_mse = float(pr["day_error_mse"])
                    pump_id_l2 = int(pr.get("pump_id", pump_id))
                    l2_alarm = _l2_alarm_threshold(l2_thresholds, pump_id_l2)
                    l2_mean = _l2_training_mean(l2_thresholds, pump_id_l2)
                    s2 = _normalize_severity(day_mse, l2_alarm)
                    h2 = _compute_health_score(day_mse, l2_mean, l2_alarm)
            except Exception as exc:
                print(f"    [L2 WARN] {csv_path.name}: {exc}")

        # ── Only keep records where at least L1 succeeded ─────────────────
        if s1 is None:
            continue

        records.append(
            {
                "pump_id": pump_id,
                "date": date_str,
                "s1": s1,
                "s2": s2,
                "h1": h1,
                "h2": h2,
                "label": label,
            }
        )

    return records


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def compute_clf_metrics(
    scores_normal: np.ndarray,
    scores_abnormal: np.ndarray,
) -> dict:
    """Compute AUC, F1, precision, recall at Youden's J optimal threshold."""
    labels = np.concatenate(
        [np.zeros(len(scores_normal)), np.ones(len(scores_abnormal))]
    )
    scores = np.concatenate([scores_normal, scores_abnormal])

    if len(np.unique(labels)) < 2:
        return {
            "auc_roc": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }

    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    opt_thresh = float(thresholds[optimal_idx])

    preds = (scores >= opt_thresh).astype(int)
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))

    return {
        "auc_roc": round(auc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "optimal_threshold": round(opt_thresh, 6),
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def grid_search(
    records_normal: list[dict],
    records_abnormal_opt: list[dict],
    records_abnormal_val: list[dict],
    l2_available: bool,
) -> tuple[list[dict], dict, dict, dict, dict]:
    """Grid search over α and w.

    Returns:
        heatmap_opt        — list of {alpha, health_weight, auc_roc} on opt set
        best_params        — {alpha, health_weight}
        best_metrics_opt   — metrics on opt set at best params
        best_metrics_val   — metrics on val set at best params
        current_metrics    — metrics at (alpha=0.3, health_weight=0.6) on opt set
    """
    alphas = np.round(np.arange(0.0, 0.61, 0.05), 2).tolist()
    weights = np.round(np.arange(0.4, 0.81, 0.05), 2).tolist()

    def build_ensemble_scores(
        records: list[dict], alpha: float, w: float
    ) -> np.ndarray:
        scores = []
        for r in records:
            s1 = r["s1"]
            s2 = (
                r["s2"] if r["s2"] is not None else s1
            )  # fallback to s1 when L2 unavailable
            ens_score = max(s1, s2) + alpha * min(s1, s2)
            scores.append(ens_score)
        return np.array(scores, dtype=float)

    normal_scores_cache: dict[tuple, np.ndarray] = {}

    def get_normal_scores(alpha: float, w: float) -> np.ndarray:
        key = (alpha, w)
        if key not in normal_scores_cache:
            normal_scores_cache[key] = build_ensemble_scores(records_normal, alpha, w)
        return normal_scores_cache[key]

    heatmap_opt: list[dict] = []
    best_auc = -1.0
    best_alpha = CURRENT_ALPHA
    best_w = CURRENT_HEALTH_WEIGHT
    current_metrics: dict = {}

    for alpha in alphas:
        for w in weights:
            norm_scores = get_normal_scores(alpha, w)
            abn_scores_opt = build_ensemble_scores(records_abnormal_opt, alpha, w)

            metrics_opt = compute_clf_metrics(norm_scores, abn_scores_opt)
            auc_opt = metrics_opt["auc_roc"]

            heatmap_opt.append({"alpha": alpha, "health_weight": w, "auc_roc": auc_opt})

            if (
                abs(alpha - CURRENT_ALPHA) < 1e-9
                and abs(w - CURRENT_HEALTH_WEIGHT) < 1e-9
            ):
                current_metrics = metrics_opt

            if auc_opt > best_auc:
                best_auc = auc_opt
                best_alpha = alpha
                best_w = w

    # Compute val-set metrics at best params
    norm_scores_best = build_ensemble_scores(records_normal, best_alpha, best_w)
    abn_scores_val = build_ensemble_scores(records_abnormal_val, best_alpha, best_w)
    best_metrics_opt_full = compute_clf_metrics(
        build_ensemble_scores(records_normal, best_alpha, best_w),
        build_ensemble_scores(records_abnormal_opt, best_alpha, best_w),
    )
    best_metrics_val = compute_clf_metrics(norm_scores_best, abn_scores_val)

    return (
        heatmap_opt,
        {"alpha": best_alpha, "health_weight": best_w},
        best_metrics_opt_full,
        best_metrics_val,
        current_metrics,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    train_path: Path,
    test_path: Path,
    l1_weights_dir: Path,
    l2_version_dir: Path,
    output_path: Path,
) -> dict:
    print("\n" + "=" * 65)
    print("E7: Fusion Weight Optimization (RD-3)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load models and thresholds
    # ------------------------------------------------------------------
    print("\n[1/4] Loading models ...")

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # L1
    l1_predictor = _import_l1_predictor(l1_weights_dir)
    l1_feature_names: list[str] = list(l1_predictor.output_columns)
    l1_thresholds = _load_l1_thresholds(l1_weights_dir)

    global_maha = l1_thresholds["global"]["mahalanobis"]
    mean_residual_vector = l1_thresholds["global"].get("mean_residual_vector")
    if not mean_residual_vector:
        # Rebuild from per-feature stats in global block
        pf = l1_thresholds["global"].get("per_feature", {})
        mean_residual_vector = [
            pf.get(f, {}).get("mean_residual", 0.0) for f in l1_feature_names
        ]
    l1_mu = np.asarray(mean_residual_vector, dtype=float)

    inv_cov = l1_thresholds["global"].get("inverse_covariance_matrix")
    if inv_cov:
        l1_sigma_inv = np.asarray(inv_cov, dtype=float)
    else:
        # Fall back to diagonal (variance only)
        pf = l1_thresholds["global"].get("per_feature", {})
        stds = [float(pf.get(f, {}).get("std_residual", 1.0)) for f in l1_feature_names]
        l1_sigma_inv = np.diag(
            1.0 / np.square(np.where(np.array(stds) > 1e-12, stds, 1e-12))
        )

    print(
        f"  L1 loaded — {len(l1_feature_names)} features, alarm={global_maha['alarm']:.3f}"
    )

    # L2
    l2_detector = None
    l2_thresholds: dict = {}
    l2_available = False
    try:
        l2_detector = _import_l2_detector(l2_version_dir)
        l2_thresholds = _load_l2_thresholds(l2_version_dir)
        l2_available = True
        print(
            f"  L2 loaded — version={l2_version_dir.name}, alarm={l2_thresholds['global']['alarm']:.6f}"
        )
    except Exception as exc:
        print(f"  L2 BLOCKED: {exc}")
        print("  Proceeding with L1-only scores (s2=s1, h2=h1 fallback).")

    # ------------------------------------------------------------------
    # 2. Collect per-day scores (Phase 1)
    # ------------------------------------------------------------------
    print("\n[2/4] Collecting per-day scores ...")

    normal_files = sorted(train_path.glob("*.csv"))
    abnormal_files = sorted(test_path.glob("*.csv"))

    print(f"  Normal days:   {len(normal_files)} CSVs in {train_path}")
    print(f"  Abnormal days: {len(abnormal_files)} CSVs in {test_path}")

    if not normal_files:
        raise FileNotFoundError(f"No CSV files found in {train_path}")
    if not abnormal_files:
        raise FileNotFoundError(f"No CSV files found in {test_path}")

    print("  Processing normal days ...")
    records_normal = collect_day_scores(
        normal_files,
        label=0,
        l1_predictor=l1_predictor,
        l1_feature_names=l1_feature_names,
        l1_mu=l1_mu,
        l1_sigma_inv=l1_sigma_inv,
        l1_thresholds=l1_thresholds,
        l2_detector=l2_detector,
        l2_thresholds=l2_thresholds,
    )

    print("  Processing abnormal days ...")
    records_abnormal_all = collect_day_scores(
        abnormal_files,
        label=1,
        l1_predictor=l1_predictor,
        l1_feature_names=l1_feature_names,
        l1_mu=l1_mu,
        l1_sigma_inv=l1_sigma_inv,
        l1_thresholds=l1_thresholds,
        l2_detector=l2_detector,
        l2_thresholds=l2_thresholds,
    )

    # 80/20 split of abnormal set (opt / val)
    rng = np.random.default_rng(42)
    n_abn = len(records_abnormal_all)
    n_opt = int(0.8 * n_abn)
    shuffle_idx = rng.permutation(n_abn)
    opt_idx = shuffle_idx[:n_opt]
    val_idx = shuffle_idx[n_opt:]

    records_abnormal_opt = [records_abnormal_all[i] for i in opt_idx]
    records_abnormal_val = [records_abnormal_all[i] for i in val_idx]

    l2_coverage = sum(
        1 for r in records_normal + records_abnormal_all if r["s2"] is not None
    )
    l2_total = len(records_normal) + len(records_abnormal_all)

    print(
        f"\n  Collected: {len(records_normal)} normal, {len(records_abnormal_all)} abnormal days"
    )
    print(
        f"  Abnormal split: {len(records_abnormal_opt)} opt / {len(records_abnormal_val)} val"
    )
    print(f"  L2 coverage: {l2_coverage}/{l2_total} days")

    # ------------------------------------------------------------------
    # 3. Grid search (Phase 2)
    # ------------------------------------------------------------------
    print("\n[3/4] Running grid search ...")

    heatmap_opt, best_params, best_metrics_opt, best_metrics_val, current_metrics = (
        grid_search(
            records_normal=records_normal,
            records_abnormal_opt=records_abnormal_opt,
            records_abnormal_val=records_abnormal_val,
            l2_available=l2_available,
        )
    )

    # ------------------------------------------------------------------
    # 4. Results
    # ------------------------------------------------------------------
    delta_auc_opt = round(
        best_metrics_opt["auc_roc"] - current_metrics.get("auc_roc", float("nan")), 4
    )
    delta_auc_val = round(
        best_metrics_val["auc_roc"] - current_metrics.get("auc_roc", float("nan")), 4
    )

    if delta_auc_opt > 0.005:
        verdict = "IMPROVEMENT"
        recommendation = (
            f"Update scoring.py constants to alpha={best_params['alpha']}, "
            f"health_weight={best_params['health_weight']}"
        )
    elif delta_auc_opt < -0.002:
        verdict = "NO_CHANGE"
        recommendation = (
            "Keep current constants (best found is worse or equal to current)"
        )
    elif abs(delta_auc_opt) <= 0.005:
        verdict = "MARGINAL"
        recommendation = (
            "Keep current constants — improvement is within noise margin (<0.005 AUC)"
        )
    else:
        verdict = "NO_CHANGE"
        recommendation = "Keep current constants"

    notes_parts = [
        f"L1 scores: {len(records_normal)} normal, {len(records_abnormal_all)} abnormal days processed.",
        f"L2 {'available — dual-model fusion' if l2_available else 'unavailable — L1-only fallback (s2=s1)'}.",
        f"L2 coverage: {l2_coverage}/{l2_total} days.",
        "Abnormal split seed=42: 80% opt / 20% val.",
        f"Grid: α∈[0.0,0.6] step=0.05 ({len(set(e['alpha'] for e in heatmap_opt))} values), "
        f"w∈[0.4,0.8] step=0.05 ({len(set(e['health_weight'] for e in heatmap_opt))} values).",
    ]

    results = {
        "experiment": "E7",
        "description": "RD-3 — Grid search over fusion corroboration factor and health weight",
        "status": "COMPLETE",
        "current_params": {
            "alpha": CURRENT_ALPHA,
            "health_weight": CURRENT_HEALTH_WEIGHT,
        },
        "current_metrics": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in current_metrics.items()
        },
        "best_params": best_params,
        "best_metrics_opt_set": best_metrics_opt,
        "best_metrics_val_set": best_metrics_val,
        "delta_auc_opt": delta_auc_opt,
        "delta_auc_val": delta_auc_val,
        "n_normal_days": len(records_normal),
        "n_abnormal_opt": len(records_abnormal_opt),
        "n_abnormal_val": len(records_abnormal_val),
        "l2_available": l2_available,
        "l2_coverage": f"{l2_coverage}/{l2_total}",
        "grid_heatmap": heatmap_opt,
        "verdict": verdict,
        "recommendation": recommendation,
        "notes": " ".join(notes_parts),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Summary print
    print("\n" + "=" * 65)
    print(f"  Current   α={CURRENT_ALPHA}, w={CURRENT_HEALTH_WEIGHT}:")
    print(
        f"    AUC={current_metrics.get('auc_roc', 'n/a')}, F1={current_metrics.get('f1', 'n/a')}"
    )
    print(f"  Best      α={best_params['alpha']}, w={best_params['health_weight']}:")
    print(f"    OPT AUC={best_metrics_opt['auc_roc']}, F1={best_metrics_opt['f1']}")
    print(f"    VAL AUC={best_metrics_val['auc_roc']}, F1={best_metrics_val['f1']}")
    print(f"  Δ AUC (opt) = {delta_auc_opt:+.4f}")
    print(f"  Δ AUC (val) = {delta_auc_val:+.4f}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  RECOMMENDATION: {recommendation}")
    print(f"\n  Results written to {output_path}")
    print("=" * 65 + "\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E7: Fusion weight optimization via grid search (RD-3)",
    )
    parser.add_argument(
        "--train-path",
        default=str(NEW_DATA_TRAIN_DEFAULT),
        help="Directory of normal (train) CSVs",
    )
    parser.add_argument(
        "--test-path",
        default=str(NEW_DATA_TEST_DEFAULT),
        help="Directory of abnormal (test) CSVs",
    )
    parser.add_argument(
        "--l1-weights",
        default=str(L1_WEIGHTS_DEFAULT),
        help="L1 weights directory (best_weights.pt, production_thresholds.json, norm_params.json)",
    )
    parser.add_argument(
        "--l2-version",
        default=str(L2_VERSION_DEFAULT),
        help="L2 version directory (model_weights.ckpt, production_thresholds.json, norm_params.json)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DEFAULT),
        help="Output JSON path",
    )
    args = parser.parse_args()

    run(
        train_path=Path(args.train_path),
        test_path=Path(args.test_path),
        l1_weights_dir=Path(args.l1_weights),
        l2_version_dir=Path(args.l2_version),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
