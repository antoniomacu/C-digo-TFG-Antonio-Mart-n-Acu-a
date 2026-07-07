"""Experiment E2: Mahalanobis Calibration for L1 Anomaly Scoring.

Question:
    Does replacing MSE scoring with Mahalanobis distance (Sigma_inv recomputed
    from the full new_data training set) improve val_ratio and AUC?

Key insight:
    MSE treats all 13 channels equally and is dominated by bearing vibration
    channels. Mahalanobis equalizes channel contributions by accounting for the
    full residual covariance structure across channels.

No retraining — production weights from ensemble/cond_reg_v2/model/weights/
are used as-is.

Usage:
    From project root:
        cd ensemble
        uv run python ../experiments/e2_mahalanobis_calibration.py

    Or with explicit paths:
        uv run python experiments/e2_mahalanobis_calibration.py \
            --train-path /path/to/new_data/train \
            --test-normal-path /path/to/new_data/train \
            --test-abnormal-path /path/to/new_data/test \
            --weights-dir ensemble/cond_reg_v2/model/weights \
            --output experiments/E2_mahalanobis_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
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
# Path resolution — script can be run from the ensemble/ directory or from
# the project root. Locate key directories relative to this file.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent  # experiments/
PROJECT_ROOT = SCRIPT_DIR.parent  # unsupervised_learning/
ENSEMBLE_DIR = PROJECT_ROOT / "ensemble"
WEIGHTS_DIR_DEFAULT = ENSEMBLE_DIR / "cond_reg_v2" / "model" / "weights"
THRESHOLDS_PATH = WEIGHTS_DIR_DEFAULT / "production_thresholds.json"

# New data lives in the supervised_learning sibling project
SUPERVISED_ROOT = PROJECT_ROOT.parent / "supervised_learning"
NEW_DATA_TRAIN_DEFAULT = SUPERVISED_ROOT / "new_data" / "train"
NEW_DATA_TEST_DEFAULT = SUPERVISED_ROOT / "new_data" / "test"

BASELINE = {
    "scoring": "MSE",
    "auc_roc": 0.9663,
    "f1": 0.9663,
    "precision": 0.991,
    "recall": 0.943,
    "val_ratio": 6.41,
}


def _import_predictor(weights_dir: Path):
    """Import PumpPredictor, injecting ensemble on sys.path if needed."""
    try:
        from cond_reg_v2.model.inference import PumpPredictor  # type: ignore[import]

        return PumpPredictor(weights_dir=str(weights_dir))
    except ImportError:
        # Running without installed package — add parent of cond_reg_v2 to path
        sys.path.insert(0, str(ENSEMBLE_DIR))
        from cond_reg_v2.model.inference import PumpPredictor  # type: ignore[import]

        return PumpPredictor(weights_dir=str(weights_dir))


def _load_csvs(directory: Path, label: str) -> list[Path]:
    """Return sorted list of CSV files in directory."""
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {directory} ({label}). "
            "Check that the path is correct and files exist."
        )
    print(f"  {label}: {len(files)} CSV files in {directory}")
    return files


def _compute_residuals(
    predictor,
    feature_names: list[str],
    csv_files: list[Path],
    label: str,
) -> pd.DataFrame:
    """Run predictor on each CSV file and collect residual vectors.

    Residual = actual_output - predicted_output (per sample, per channel).

    Returns:
        DataFrame with columns: timestamp, pump_id, + 13 residual columns
    """
    import re

    rows: list[pd.DataFrame] = []

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"  [SKIP] {csv_path.name}: read error: {exc}")
            continue

        # Verify required output columns exist (training-class labels)
        missing = [col for col in feature_names if col not in df.columns]
        if missing:
            print(f"  [SKIP] {csv_path.name}: missing output columns {missing}")
            continue

        # Resolve timestamps
        if "timestamp" in df.columns:
            timestamps = df["timestamp"].copy()
        else:
            timestamps = pd.Series(pd.RangeIndex(len(df)).astype(str), name="timestamp")

        # Resolve pump_id
        if "pump_id" in df.columns:
            pump_id_series = pd.to_numeric(df["pump_id"], errors="coerce")
        else:
            match = re.search(r"pump[_-]?(\d+)", csv_path.stem, flags=re.IGNORECASE)
            if match:
                pump_id_series = pd.Series(
                    np.full(len(df), int(match.group(1))), dtype=float
                )
            else:
                print(f"  [SKIP] {csv_path.name}: cannot infer pump_id")
                continue

        try:
            predictions = predictor.predict(df)
        except Exception as exc:
            print(f"  [SKIP] {csv_path.name}: prediction error: {exc}")
            continue

        actual = df[feature_names].apply(pd.to_numeric, errors="coerce")
        predicted = predictions[feature_names].apply(pd.to_numeric, errors="coerce")
        residual = actual - predicted

        frame = pd.DataFrame({"timestamp": timestamps, "pump_id": pump_id_series})
        frame[feature_names] = residual.values
        frame = frame.dropna(subset=["pump_id"] + feature_names)
        frame["pump_id"] = frame["pump_id"].astype(int)
        rows.append(frame)

    if not rows:
        raise RuntimeError(f"[{label}] All CSV files failed — no residuals computed.")

    result = pd.concat(rows, ignore_index=True)
    print(f"  {label}: {len(result):,} valid samples collected")
    return result


def _fit_covariance(residuals_df: pd.DataFrame, feature_names: list[str]) -> dict:
    """Compute mean residual, covariance, and inverse covariance.

    Ridge regularisation (1e-6 * I) is applied before inversion to handle
    near-singular covariance matrices (highly correlated sensor channels).

    Returns:
        {
            "mu": [13],
            "sigma": [[13,13]],
            "sigma_inv": [[13,13]],
            "condition_number": float,
            "n_samples": int,
            "ridge_applied": bool,
        }
    """
    REG_EPS = 1e-6

    R = residuals_df[feature_names].to_numpy(dtype=float)
    n_samples, n_features = R.shape

    mu = np.mean(R, axis=0)
    sigma = np.cov(R, rowvar=False)

    if sigma.ndim == 0:  # single feature (shouldn't happen here)
        sigma = np.array([[float(sigma)]])

    cond = float(np.linalg.cond(sigma))
    ridge_applied = cond > 1e10 or not np.isfinite(cond)

    sigma_reg = sigma + REG_EPS * np.eye(n_features, dtype=float)
    sigma_inv = np.linalg.pinv(sigma_reg)

    return {
        "mu": mu,
        "sigma": sigma,
        "sigma_inv": sigma_inv,
        "condition_number": cond if np.isfinite(cond) else float("inf"),
        "n_samples": n_samples,
        "ridge_applied": ridge_applied,
    }


def _mahalanobis_scores(
    residuals_df: pd.DataFrame,
    feature_names: list[str],
    mu: np.ndarray,
    sigma_inv: np.ndarray,
) -> np.ndarray:
    """Compute per-sample Mahalanobis distance.

    d_M = sqrt((r - mu)^T Sigma_inv (r - mu))
    """
    R = residuals_df[feature_names].to_numpy(dtype=float)
    centered = R - mu
    mahal_sq = np.einsum("ij,jk,ik->i", centered, sigma_inv, centered)
    mahal_sq = np.clip(mahal_sq, a_min=0.0, a_max=None)
    return np.sqrt(mahal_sq)


def _mse_scores(residuals_df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Compute per-sample MSE across all output channels (baseline scoring)."""
    R = residuals_df[feature_names].to_numpy(dtype=float)
    return np.mean(R**2, axis=1)


def _aggregate_by_day(
    scores: np.ndarray,
    timestamps: pd.Series,
    pump_ids: pd.Series,
) -> np.ndarray:
    """Aggregate per-sample scores to (pump, date) means — mirrors failure_detector.py."""
    df = pd.DataFrame(
        {
            "score": scores,
            "timestamp": pd.to_datetime(timestamps),
            "pump_id": pump_ids,
        }
    )
    df["date"] = df["timestamp"].dt.date
    return df.groupby(["pump_id", "date"])["score"].mean().to_numpy()


def _compute_clf_metrics(
    day_scores_normal: np.ndarray,
    day_scores_abnormal: np.ndarray,
) -> dict:
    """Compute AUC, F1, precision, recall using Youden's J optimal threshold.

    Mirrors FailureDetector.compute_classification_metrics() in failure_detector.py.
    """
    labels = np.concatenate(
        [np.zeros(len(day_scores_normal)), np.ones(len(day_scores_abnormal))]
    )
    scores = np.concatenate([day_scores_normal, day_scores_abnormal])

    auc = float(roc_auc_score(labels, scores))

    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    optimal_threshold = float(thresholds[optimal_idx])

    predictions = (scores >= optimal_threshold).astype(int)
    prec = float(precision_score(labels, predictions, zero_division=0))
    rec = float(recall_score(labels, predictions, zero_division=0))
    f1 = float(f1_score(labels, predictions))

    return {
        "auc_roc": round(auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "optimal_threshold": round(optimal_threshold, 6),
        "n_normal_days": int(len(day_scores_normal)),
        "n_abnormal_days": int(len(day_scores_abnormal)),
    }


def _update_production_thresholds(
    thresholds_path: Path,
    sigma_inv: np.ndarray,
    mu: np.ndarray,
    n_training_samples: int,
    condition_number: float,
) -> None:
    """Update production_thresholds.json with the new Sigma_inv and residual_mean.

    Preserves all existing keys; only overwrites:
        - global.inverse_covariance_matrix  (matches existing schema)
        - global.mean_residual_vector        (matches existing schema)
        - n_training_samples
    Also adds top-level "residual_mean" for E2 traceability.
    """
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    # Update global section (schema follows threshold_calibration.py convention)
    if "global" not in thresholds:
        thresholds["global"] = {}

    thresholds["global"]["inverse_covariance_matrix"] = sigma_inv.tolist()
    thresholds["global"]["mean_residual_vector"] = mu.tolist()
    thresholds["n_training_samples"] = n_training_samples

    # Top-level convenience keys for E2 traceability
    thresholds["residual_mean"] = mu.tolist()
    thresholds["e2_condition_number"] = condition_number
    thresholds["e2_note"] = (
        "inverse_covariance_matrix and residual_mean recomputed from full "
        "new_data training set by E2 experiment (e2_mahalanobis_calibration.py)."
    )

    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    print(f"  Updated {thresholds_path}")


def _verdict(delta_auc: float, delta_val_ratio: float) -> str:
    """Classify the result: IMPROVEMENT, REGRESSION, or NEUTRAL."""
    # Improvement: both metrics improve, or one improves substantially
    if delta_auc > 0.002 and delta_val_ratio > 0.1:
        return "IMPROVEMENT"
    if delta_auc < -0.002 and delta_val_ratio < -0.1:
        return "REGRESSION"
    return "NEUTRAL"


def run(
    train_path: Path,
    test_normal_path: Path,
    test_abnormal_path: Path,
    weights_dir: Path,
    thresholds_path: Path,
    output_path: Path,
) -> dict:
    """Main experiment pipeline. Returns the full results dict."""
    print("\n" + "=" * 60)
    print("E2: Mahalanobis Calibration for L1 Anomaly Scoring")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("\n[1/6] Loading production model...")
    predictor = _import_predictor(weights_dir)
    feature_names: list[str] = predictor.output_columns
    print(f"  Output features ({len(feature_names)}): {feature_names}")

    # ------------------------------------------------------------------
    # 2. Collect training residuals (normal operating data only)
    # ------------------------------------------------------------------
    print(f"\n[2/6] Computing training residuals on {train_path} ...")
    train_files = _load_csvs(train_path, "train-normal")
    train_residuals = _compute_residuals(predictor, feature_names, train_files, "train-normal")

    # ------------------------------------------------------------------
    # 3. Fit covariance on training residuals
    # ------------------------------------------------------------------
    print("\n[3/6] Fitting residual covariance matrix...")
    cov_result = _fit_covariance(train_residuals, feature_names)
    mu = cov_result["mu"]
    sigma_inv = cov_result["sigma_inv"]
    n_train = cov_result["n_samples"]
    cond = cov_result["condition_number"]

    print(f"  Training samples: {n_train:,}")
    print(f"  Condition number of Sigma: {cond:.3e}")
    if cov_result["ridge_applied"]:
        print("  Ridge regularisation applied (1e-6 * I) — Sigma was ill-conditioned")
    print(f"  mu  (first 3): {mu[:3].round(4)}")

    # ------------------------------------------------------------------
    # 4. Update production_thresholds.json
    # ------------------------------------------------------------------
    print(f"\n[4/6] Updating {thresholds_path} ...")
    _update_production_thresholds(thresholds_path, sigma_inv, mu, n_train, cond)

    # ------------------------------------------------------------------
    # 5. Compute test residuals (normal and abnormal)
    # ------------------------------------------------------------------
    print(f"\n[5/6] Computing test residuals...")
    print(f"  Normal test data: {test_normal_path}")
    test_normal_files = _load_csvs(test_normal_path, "test-normal")
    normal_residuals = _compute_residuals(
        predictor, feature_names, test_normal_files, "test-normal"
    )

    print(f"  Abnormal test data: {test_abnormal_path}")
    test_abnormal_files = _load_csvs(test_abnormal_path, "test-abnormal")
    abnormal_residuals = _compute_residuals(
        predictor, feature_names, test_abnormal_files, "test-abnormal"
    )

    # ------------------------------------------------------------------
    # 6. Score with Mahalanobis distance & aggregate to day-level
    # ------------------------------------------------------------------
    print("\n[6/6] Computing Mahalanobis scores and classification metrics...")
    mahal_normal = _mahalanobis_scores(normal_residuals, feature_names, mu, sigma_inv)
    mahal_abnormal = _mahalanobis_scores(abnormal_residuals, feature_names, mu, sigma_inv)

    day_mahal_normal = _aggregate_by_day(
        mahal_normal,
        normal_residuals["timestamp"],
        normal_residuals["pump_id"],
    )
    day_mahal_abnormal = _aggregate_by_day(
        mahal_abnormal,
        abnormal_residuals["timestamp"],
        abnormal_residuals["pump_id"],
    )

    val_ratio_mahal = float(
        np.mean(day_mahal_abnormal) / (np.mean(day_mahal_normal) + 1e-12)
    )

    clf = _compute_clf_metrics(day_mahal_normal, day_mahal_abnormal)

    print(f"\n  Mahalanobis — normal  days: {clf['n_normal_days']}")
    print(f"  Mahalanobis — abnormal days: {clf['n_abnormal_days']}")
    print(f"  Mean Mahal (normal):   {np.mean(day_mahal_normal):.4f}")
    print(f"  Mean Mahal (abnormal): {np.mean(day_mahal_abnormal):.4f}")
    print(f"  val_ratio:  {val_ratio_mahal:.4f}  (baseline MSE: {BASELINE['val_ratio']})")
    print(f"  AUC-ROC:    {clf['auc_roc']:.4f}  (baseline: {BASELINE['auc_roc']})")
    print(f"  F1:         {clf['f1']:.4f}  (baseline: {BASELINE['f1']})")
    print(f"  Precision:  {clf['precision']:.4f}")
    print(f"  Recall:     {clf['recall']:.4f}")

    # ------------------------------------------------------------------
    # Build results dict
    # ------------------------------------------------------------------
    delta_auc = round(clf["auc_roc"] - BASELINE["auc_roc"], 4)
    delta_val_ratio = round(val_ratio_mahal - BASELINE["val_ratio"], 4)
    verdict = _verdict(delta_auc, delta_val_ratio)

    results = {
        "experiment": "E2",
        "description": "Mahalanobis calibration for L1 anomaly scoring",
        "status": "COMPLETED",
        "baseline": BASELINE,
        "mahalanobis": {
            "n_training_samples": n_train,
            "condition_number": round(cond, 3) if np.isfinite(cond) else None,
            "ridge_applied": cov_result["ridge_applied"],
            "n_normal_days": clf["n_normal_days"],
            "n_abnormal_days": clf["n_abnormal_days"],
            "mean_mahal_normal": round(float(np.mean(day_mahal_normal)), 6),
            "mean_mahal_abnormal": round(float(np.mean(day_mahal_abnormal)), 6),
            "auc_roc": clf["auc_roc"],
            "f1": clf["f1"],
            "precision": clf["precision"],
            "recall": clf["recall"],
            "val_ratio": round(val_ratio_mahal, 4),
            "delta_auc": delta_auc,
            "delta_val_ratio": delta_val_ratio,
            "optimal_threshold": clf["optimal_threshold"],
        },
        "verdict": verdict,
        "notes": (
            f"Sigma_inv recomputed on {n_train:,} samples from new_data/train. "
            f"Condition number: {cond:.2e}. "
            f"AUC delta: {delta_auc:+.4f}, val_ratio delta: {delta_val_ratio:+.4f}. "
            "Day-level aggregation used (same as FailureDetector.compute_classification_metrics). "
            "Test-normal = new_data/train; Test-abnormal = new_data/test (anomalous pumps)."
        ),
    }

    # ------------------------------------------------------------------
    # Write results JSON
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results written to {output_path}")
    print(f"\n  VERDICT: {verdict}")
    print("=" * 60 + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2: Mahalanobis calibration for L1 anomaly scoring"
    )
    parser.add_argument(
        "--train-path",
        default=str(NEW_DATA_TRAIN_DEFAULT),
        help="Directory of normal training CSVs (default: supervised_learning/new_data/train)",
    )
    parser.add_argument(
        "--test-normal-path",
        default=str(NEW_DATA_TRAIN_DEFAULT),
        help=(
            "Directory of test-normal CSVs. Defaults to train path — the train split "
            "serves as held-out normal reference when no separate test-normal set exists."
        ),
    )
    parser.add_argument(
        "--test-abnormal-path",
        default=str(NEW_DATA_TEST_DEFAULT),
        help="Directory of test-abnormal CSVs (default: supervised_learning/new_data/test)",
    )
    parser.add_argument(
        "--weights-dir",
        default=str(WEIGHTS_DIR_DEFAULT),
        help="Directory with best_weights.pt and norm_params.json",
    )
    parser.add_argument(
        "--thresholds",
        default=str(THRESHOLDS_PATH),
        help="Path to production_thresholds.json to update",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "E2_mahalanobis_results.json"),
        help="Output JSON path for experiment results",
    )
    args = parser.parse_args()

    run(
        train_path=Path(args.train_path),
        test_normal_path=Path(args.test_normal_path),
        test_abnormal_path=Path(args.test_abnormal_path),
        weights_dir=Path(args.weights_dir),
        thresholds_path=Path(args.thresholds),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
