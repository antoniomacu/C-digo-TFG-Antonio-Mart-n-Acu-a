"""Experiment E3 — ELBO Anomaly Scoring for the Temporal CVAE (L1 model).

Question
--------
Does adding the KLD component to the reconstruction-based anomaly score
improve failure detection, especially for gradual thermal drift cases?

Hypothesis
----------
Gradual drift may not cause large reconstruction errors but pushes the
posterior q(z|x) away from the prior N(0,I), producing high KLD.
Adding lambda_kld * KLD to the score may catch these cases.

ELBO anomaly score:
    score_i = recon_loss_i + lambda_kld * KLD_i
    KLD_i   = -0.5 * sum_j(1 + logvar_j - mu_j^2 - exp(logvar_j))

No retraining. Production weights from ensemble/cond_reg_v2/model/weights/.

Usage
-----
    cd <PATH_TO_PROJECT>
    uv run python experiments/e3_elbo_scoring.py [--data-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup — make the workspace root importable regardless of cwd.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
ENSEMBLE_DIR = WORKSPACE_ROOT / "ensemble"

sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(ENSEMBLE_DIR))

from ensemble.cond_reg_v2.model.models import TemporalCVAE  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEIGHTS_DIR = ENSEMBLE_DIR / "cond_reg_v2" / "model" / "weights"
RESULTS_PATH = SCRIPT_DIR / "E3_elbo_results.json"

# data dir candidates — resolved at runtime
DATA_DIR_CANDIDATES: list[Path] = [
    Path("<PATH_TO_DATA_DIR>"),
    WORKSPACE_ROOT / "new_data",
    WORKSPACE_ROOT / "data",
]

LAMBDA_GRID: list[float] = [0.0, 0.01, 0.05, 0.10, 0.50]
MC_K = 50  # Monte Carlo encoder samples for lambda=0.05
MC_LAMBDA = 0.05  # lambda at which MC is evaluated

INPUT_VARS = [
    "Ambient temperature",
    "Main HTF Pump Speed",
    "Main HTF Pump Inlet Temperature",
    "pump_id_1",
    "pump_id_2",
    "pump_id_3",
    "pump_id_4",
]

OUTPUT_VARS = [
    "Main HTF Pump Current Consumption",
    "Main HTF Pump Flow",
    "Main HTF Pump Outlet Pressure",
    "Main HTF Pump NDE Outboard bearing",
    "Main HTF Pump NDE Inboard bearing",
    "Main HTF Pump DE bearing",
    "Main HTF Pump Motor bearing Temp 1",
    "Main HTF Pump Motor bearing Temp 2",
    "Main HTF Pump Motor U winding Temp 1",
    "Main HTF Pump Motor U winding Temp 2",
    "Main HTF Pump Motor U winding Temp 3",
    "Main HTF Pump DE Side Bearing vibration",
    "Main HTF Pump NDE Side Bearing vibration",
]

PAST_HISTORY = 3

# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

try:
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    logger.warning("scikit-learn not available; AUC will be computed manually.")


def _auc_roc_manual(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Trapezoid AUC-ROC without sklearn dependency."""
    thresholds = np.unique(scores)[::-1]
    tpr_list = [0.0]
    fpr_list = [0.0]
    pos = y_true.sum()
    neg = len(y_true) - pos
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tpr_list.append(tp / max(pos, 1))
        fpr_list.append(fp / max(neg, 1))
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    return float(np.trapz(tpr_list, fpr_list))


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    normal_scores: np.ndarray,
    abnormal_scores: np.ndarray,
) -> dict[str, float]:
    """Compute AUC-ROC, F1, precision, recall and val_ratio from day-level scores."""
    if SKLEARN_OK:
        auc = float(roc_auc_score(y_true, scores))
    else:
        auc = _auc_roc_manual(y_true, scores)

    # Optimal threshold: maximise F1 over the score distribution.
    thresholds = np.unique(scores)
    best_f1 = -1.0
    best_thr = thresholds[0]
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        if SKLEARN_OK:
            f1 = float(f1_score(y_true, preds, zero_division=0))
        else:
            tp = int(((preds == 1) & (y_true == 1)).sum())
            fp = int(((preds == 1) & (y_true == 0)).sum())
            fn = int(((preds == 0) & (y_true == 1)).sum())
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    preds_opt = (scores >= best_thr).astype(int)
    if SKLEARN_OK:
        prec = float(precision_score(y_true, preds_opt, zero_division=0))
        rec = float(recall_score(y_true, preds_opt, zero_division=0))
    else:
        tp = int(((preds_opt == 1) & (y_true == 1)).sum())
        fp = int(((preds_opt == 1) & (y_true == 0)).sum())
        fn = int(((preds_opt == 0) & (y_true == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)

    # val_ratio: mean abnormal score / mean normal score (separation metric).
    mean_normal = float(np.mean(normal_scores)) if len(normal_scores) > 0 else 1e-8
    mean_abnormal = float(np.mean(abnormal_scores)) if len(abnormal_scores) > 0 else 0.0
    val_ratio = mean_abnormal / max(mean_normal, 1e-12)

    return {
        "auc_roc": round(auc, 4),
        "f1": round(best_f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "val_ratio": round(val_ratio, 4),
        "optimal_threshold": float(best_thr),
        "mean_normal_score": round(mean_normal, 6),
        "mean_abnormal_score": round(mean_abnormal, 6),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _resolve_data_dir(override: str | None = None) -> Path:
    if override is not None:
        p = Path(override).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified data dir not found: {p}")

    for candidate in DATA_DIR_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "No data directory found. Tried:\n  "
        + "\n  ".join(str(c) for c in DATA_DIR_CANDIDATES)
        + "\nSet --data-dir to the folder containing train/ and test/ sub-dirs."
    )


def _parse_pump_from_filename(path: Path) -> int | None:
    match = re.match(r"pump_(\d+)_", path.name)
    return int(match.group(1)) if match else None


def _resample_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if len(df.index) < 3:
        return df.sort_index()
    diffs = pd.Series(df.index).sort_values().diff().dropna().dt.total_seconds() / 60.0
    if diffs.empty or float(diffs.median()) >= 3.0:
        return df.sort_index()
    return df.resample("5min").first().dropna(how="all").sort_index()


def _apply_speed_filter(df: pd.DataFrame) -> pd.DataFrame:
    speed_col = "Main HTF Pump Speed"
    if speed_col not in df.columns:
        return df
    speed_series = pd.to_numeric(df[speed_col], errors="coerce")
    peak_speed = float(speed_series.max()) if not speed_series.empty else float("nan")
    if not pd.notna(peak_speed) or peak_speed <= 0.0:
        return df
    stable_threshold = peak_speed * 0.90
    stable_mask = (speed_series >= stable_threshold).fillna(False)
    stable_indices = df.index[stable_mask]
    if len(stable_indices) < 10:
        return df
    return df.loc[stable_indices[0] : stable_indices[-1]]


def _load_day(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {csv_path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if df.empty:
        return df
    df = _resample_if_needed(df)
    df = _apply_speed_filter(df)
    return df


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(weights_dir: Path) -> TemporalCVAE:
    """Load TemporalCVAE from best_weights.pt or model_weights.ckpt."""
    best_weights = weights_dir / "best_weights.pt"
    ckpt = weights_dir / "model_weights.ckpt"

    if best_weights.exists():
        checkpoint = torch.load(best_weights, map_location="cpu", weights_only=False)
        params = checkpoint["params"]
        # Reconstruct the LightningModule shell (no training data needed).
        model = TemporalCVAE(**params)
        model.temporal_embedding.load_state_dict(
            checkpoint["temporal_embedding_state_dict"]
        )
        model.temporal_attention.load_state_dict(
            checkpoint["temporal_attention_state_dict"]
        )
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        print(f"  Loaded best_weights.pt from {weights_dir}")
        return model.eval()

    if ckpt.exists():
        model = TemporalCVAE.load_from_checkpoint(ckpt, map_location="cpu")
        print(f"  Loaded model_weights.ckpt from {weights_dir}")
        return model.eval()

    raise FileNotFoundError(
        f"No weights found in {weights_dir}. "
        "Expected best_weights.pt or model_weights.ckpt."
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def load_norm_params(weights_dir: Path) -> dict:
    norm_path = weights_dir / "norm_params.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"norm_params.json not found in {weights_dir}")
    with open(norm_path, encoding="utf-8") as f:
        return json.load(f)


def normalize_col(series: pd.Series, col: str, norm_params: dict) -> pd.Series:
    """Min-max normalise a single column using stored training statistics."""
    if col not in norm_params:
        return series.fillna(0.0)
    stats = norm_params[col]
    lo, hi = stats["min"], stats["max"]
    denom = hi - lo
    if abs(denom) < 1e-12:
        return pd.Series(0.0, index=series.index)
    return ((series - lo) / denom).fillna(0.0)


# ---------------------------------------------------------------------------
# Feature engineering: build windows and target tensors from a loaded day DF
# ---------------------------------------------------------------------------


def build_tensors(
    df: pd.DataFrame,
    pump_id: int,
    norm_params: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Build (x_windows, y_targets) tensors from a single-day DataFrame.

    Returns None if the day has insufficient rows.
    """
    if len(df) < 5:
        return None

    # One-hot encode pump_id
    for pid in (1, 2, 3, 4):
        df = df.copy()
        df[f"pump_id_{pid}"] = int(pump_id == pid)

    # Check all expected columns are present
    missing_inputs = [c for c in INPUT_VARS if c not in df.columns]
    missing_outputs = [c for c in OUTPUT_VARS if c not in df.columns]
    if missing_inputs or missing_outputs:
        logger.warning(
            "Day missing columns — inputs: %s, outputs: %s",
            missing_inputs,
            missing_outputs,
        )
        return None

    # Normalise inputs
    norm_input_df = pd.DataFrame(index=df.index)
    for col in INPUT_VARS:
        norm_input_df[col] = normalize_col(df[col].astype(float), col, norm_params)
    norm_input_values = norm_input_df[INPUT_VARS].to_numpy(dtype=np.float32)

    # Build sliding windows of shape [N, PAST_HISTORY, n_input] -> flatten to [N, PAST_HISTORY*n_input]
    n = len(norm_input_values)
    windows = []
    for i in range(n):
        if i < PAST_HISTORY - 1:
            n_pad = PAST_HISTORY - 1 - i
            pad = np.tile(norm_input_values[0], (n_pad, 1))
            window = np.vstack([pad, norm_input_values[: i + 1]])
        else:
            window = norm_input_values[i - PAST_HISTORY + 1 : i + 1]
        windows.append(window.reshape(-1))

    x_tensor = torch.tensor(np.asarray(windows), dtype=torch.float32, device=device)

    # Normalise outputs for loss computation (model was trained on normalised targets)
    norm_output_df = pd.DataFrame(index=df.index)
    for col in OUTPUT_VARS:
        norm_output_df[col] = normalize_col(df[col].astype(float), col, norm_params)
    y_values = norm_output_df[OUTPUT_VARS].to_numpy(dtype=np.float32)
    y_tensor = torch.tensor(y_values, dtype=torch.float32, device=device)

    return x_tensor, y_tensor


# ---------------------------------------------------------------------------
# Score aggregation: day-level score = mean timestep score
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_day_elbo(
    model: TemporalCVAE,
    x: torch.Tensor,
    y: torch.Tensor,
    lambda_kld: float,
) -> float:
    """Return mean ELBO anomaly score over all timesteps of a day."""
    scores = model.anomaly_score_elbo(x, y, lambda_kld=lambda_kld)
    return float(scores.mean().item())


@torch.no_grad()
def score_day_elbo_mc(
    model: TemporalCVAE,
    x: torch.Tensor,
    y: torch.Tensor,
    lambda_kld: float,
    k: int,
) -> float:
    """
    Monte Carlo ELBO score: run the encoder K times with noise, average scores.

    Forces dropout active by putting model in training mode for the encoder
    only (stdev noise in reparameterize is the stochasticity source here).
    """
    # TemporalCVAE.reparameterize uses self.training to decide whether to add
    # noise. We need train mode temporarily so the sampler draws K distinct z.
    model.train()
    accumulated = torch.zeros(x.shape[0], device=x.device)
    for _ in range(k):
        x_reshaped = x.reshape(x.shape[0], model.past_history, model.n_input)
        embedded = model.temporal_embedding(x_reshaped)
        pooled, _ = model.temporal_attention(embedded)
        mu, logvar, _ = model.encoder(pooled)
        z = model.reparameterize(mu, logvar)
        recon = model.decoder(z)
        recon_loss_per_sample = F.mse_loss(recon, y, reduction="none").mean(dim=1)
        logvar_clamped = logvar.clamp(-30.0, 20.0)
        kld_per_sample = -0.5 * (
            1.0 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()
        ).sum(dim=1)
        accumulated += recon_loss_per_sample + lambda_kld * kld_per_sample

    model.eval()
    return float((accumulated / k).mean().item())


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    model: TemporalCVAE,
    norm_params: dict,
    data_dir: Path,
    device: torch.device,
    max_files_per_class: int | None = None,
) -> tuple[
    dict[str, list[float]],  # normal day scores keyed by lambda str
    dict[str, list[float]],  # abnormal day scores keyed by lambda str
    list[str],  # normal day names
    list[str],  # abnormal day names
    dict[str, float],  # MC scores for lambda=0.05 normal
    dict[str, float],  # MC scores for lambda=0.05 abnormal
]:
    """Iterate over all days, collect ELBO scores for each lambda."""
    normal_dir = data_dir / "train"
    abnormal_dir = data_dir / "test"

    normal_files = sorted(normal_dir.glob("*.csv"))
    abnormal_files = sorted(abnormal_dir.glob("*.csv"))

    if max_files_per_class is not None:
        normal_files = normal_files[:max_files_per_class]
        abnormal_files = abnormal_files[:max_files_per_class]

    print(f"  Normal files : {len(normal_files)}")
    print(f"  Abnormal files: {len(abnormal_files)}")

    lambda_keys = [str(lam) for lam in LAMBDA_GRID]

    normal_scores: dict[str, list[float]] = {k: [] for k in lambda_keys}
    abnormal_scores: dict[str, list[float]] = {k: [] for k in lambda_keys}
    normal_names: list[str] = []
    abnormal_names: list[str] = []

    mc_normal: list[float] = []
    mc_abnormal: list[float] = []

    def _process_file(csv_path: Path, label: str) -> None:
        pump_id = _parse_pump_from_filename(csv_path)
        if pump_id is None:
            logger.warning("Cannot parse pump_id from %s, skipping.", csv_path.name)
            return

        try:
            df = _load_day(csv_path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", csv_path.name, exc)
            return

        tensors = build_tensors(df, pump_id, norm_params, device)
        if tensors is None:
            logger.warning(
                "Skipped %s (insufficient rows after preprocessing).", csv_path.name
            )
            return

        x, y = tensors

        for lam in LAMBDA_GRID:
            key = str(lam)
            score = score_day_elbo(model, x, y, lambda_kld=lam)
            if label == "normal":
                normal_scores[key].append(score)
            else:
                abnormal_scores[key].append(score)

        # Monte Carlo pass for MC_LAMBDA
        mc_score = score_day_elbo_mc(model, x, y, lambda_kld=MC_LAMBDA, k=MC_K)
        if label == "normal":
            mc_normal.append(mc_score)
            normal_names.append(csv_path.name)
        else:
            mc_abnormal.append(mc_score)
            abnormal_names.append(csv_path.name)

    print("  Scoring normal days...")
    for i, csv_path in enumerate(normal_files):
        _process_file(csv_path, "normal")
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(normal_files)} normal done")

    print("  Scoring abnormal days...")
    for i, csv_path in enumerate(abnormal_files):
        _process_file(csv_path, "abnormal")
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(abnormal_files)} abnormal done")

    return (
        normal_scores,
        abnormal_scores,
        normal_names,
        abnormal_names,
        {str(i): v for i, v in enumerate(mc_normal)},
        {str(i): v for i, v in enumerate(mc_abnormal)},
    )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def find_cases_caught_by_elbo_not_mse(
    normal_scores: dict[str, list[float]],
    abnormal_scores: dict[str, list[float]],
    normal_names: list[str],
    abnormal_names: list[str],
) -> list[str]:
    """
    Identify abnormal days detected by any lambda >= 0.01 but missed by lambda=0.0.

    Uses the optimal threshold from lambda=0.0 (pure MSE) to label each day,
    then checks which days flip from missed -> detected at higher lambdas.
    """
    mse_normal = np.array(normal_scores["0.0"])
    mse_abnormal = np.array(abnormal_scores["0.0"])

    if len(mse_normal) == 0 or len(mse_abnormal) == 0:
        return []

    # Determine threshold for MSE (pure reconstruction): maximise F1.
    all_scores_mse = np.concatenate([mse_normal, mse_abnormal])
    all_labels = np.array([0] * len(mse_normal) + [1] * len(mse_abnormal))

    best_f1, best_thr = -1.0, 0.0
    for thr in np.unique(all_scores_mse):
        preds = (all_scores_mse >= thr).astype(int)
        tp = int(((preds == 1) & (all_labels == 1)).sum())
        fp = int(((preds == 1) & (all_labels == 0)).sum())
        fn = int(((preds == 0) & (all_labels == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    # Missed by MSE = abnormal days with score < best_thr at lambda=0.0
    missed_by_mse: set[int] = set()
    for idx, score in enumerate(mse_abnormal):
        if score < best_thr:
            missed_by_mse.add(idx)

    if not missed_by_mse:
        return []

    caught_by_elbo: list[str] = []
    for lam in LAMBDA_GRID:
        if lam < 0.01:
            continue
        lam_key = str(lam)
        elbo_abnormal = np.array(abnormal_scores[lam_key])
        # Recompute optimal threshold for this lambda.
        elbo_normal = np.array(normal_scores[lam_key])
        all_scores_lam = np.concatenate([elbo_normal, elbo_abnormal])
        best_f1_lam, best_thr_lam = -1.0, 0.0
        for thr in np.unique(all_scores_lam):
            preds = (all_scores_lam >= thr).astype(int)
            tp = int(((preds == 1) & (all_labels == 1)).sum())
            fp = int(((preds == 1) & (all_labels == 0)).sum())
            fn = int(((preds == 0) & (all_labels == 1)).sum())
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1_lam:
                best_f1_lam, best_thr_lam = f1, thr

        for idx in missed_by_mse:
            if idx < len(elbo_abnormal) and elbo_abnormal[idx] >= best_thr_lam:
                name = (
                    abnormal_names[idx] if idx < len(abnormal_names) else f"idx_{idx}"
                )
                entry = f"{name} (caught at lambda={lam})"
                if entry not in caught_by_elbo:
                    caught_by_elbo.append(entry)

    return caught_by_elbo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E3: ELBO anomaly scoring grid search on the Temporal CVAE"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to data directory (must contain train/ and test/ sub-dirs). "
        "Auto-detected if not provided.",
    )
    parser.add_argument(
        "--weights-dir",
        default=str(WEIGHTS_DIR),
        help=f"Path to model weights directory. Default: {WEIGHTS_DIR}",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files per class (for quick smoke tests).",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_PATH),
        help=f"Path for results JSON. Default: {RESULTS_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    print("\n=== Experiment E3: ELBO Anomaly Scoring ===")

    # --- Resolve paths ---
    weights_dir = Path(args.weights_dir)
    try:
        data_dir = _resolve_data_dir(args.data_dir)
        print(f"  Data dir    : {data_dir}")
    except FileNotFoundError as exc:
        print(f"\n[BLOCKED] {exc}")
        blocked_result = {
            "experiment": "E3",
            "status": "BLOCKED",
            "blocker": str(exc),
            "description": "ELBO anomaly scoring — KLD component added to reconstruction loss",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(blocked_result, f, indent=2)
        print(f"  Blocked status written to {output_path}")
        return

    print(f"  Weights dir : {weights_dir}")
    print(f"  Lambda grid : {LAMBDA_GRID}")
    print(f"  MC samples  : K={MC_K} at lambda={MC_LAMBDA}")

    # --- Resolve device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device      : {device}")

    # --- Load model ---
    print("\nLoading model...")
    try:
        model = load_model(weights_dir)
        model = model.to(device)
    except Exception as exc:
        blocked_result = {
            "experiment": "E3",
            "status": "BLOCKED",
            "blocker": f"Model load failed: {exc}",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(blocked_result, f, indent=2)
        print(f"  BLOCKED: {exc}")
        return

    # --- Load norm params ---
    print("Loading normalisation parameters...")
    try:
        norm_params = load_norm_params(weights_dir)
    except Exception as exc:
        blocked_result = {
            "experiment": "E3",
            "status": "BLOCKED",
            "blocker": f"Norm params load failed: {exc}",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(blocked_result, f, indent=2)
        print(f"  BLOCKED: {exc}")
        return

    # --- Evaluate ---
    print("\nScoring all days...")
    (
        normal_scores,
        abnormal_scores,
        normal_names,
        abnormal_names,
        _mc_normal_dict,
        _mc_abnormal_dict,
    ) = evaluate(
        model=model,
        norm_params=norm_params,
        data_dir=data_dir,
        device=device,
        max_files_per_class=args.max_files,
    )

    n_normal = len(normal_names)
    n_abnormal = len(abnormal_names)
    print(f"\n  Evaluated: {n_normal} normal days, {n_abnormal} abnormal days")

    if n_normal == 0 or n_abnormal == 0:
        blocked_result = {
            "experiment": "E3",
            "status": "BLOCKED",
            "blocker": f"Insufficient data: {n_normal} normal, {n_abnormal} abnormal days scored.",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(blocked_result, f, indent=2)
        print(f"  BLOCKED: insufficient data. Written to {output_path}")
        return

    # --- Compute metrics for each lambda ---
    print("\nComputing metrics for each lambda...")
    all_labels = np.array([0] * n_normal + [1] * n_abnormal)

    grid_results: list[dict[str, Any]] = []
    for lam in LAMBDA_GRID:
        key = str(lam)
        all_scores = np.array(normal_scores[key] + abnormal_scores[key])
        metrics = compute_metrics(
            y_true=all_labels,
            scores=all_scores,
            normal_scores=np.array(normal_scores[key]),
            abnormal_scores=np.array(abnormal_scores[key]),
        )
        entry: dict[str, Any] = {"lambda_kld": lam, **metrics}
        grid_results.append(entry)
        print(
            f"  lambda={lam:5.2f}  AUC={metrics['auc_roc']:.4f}  F1={metrics['f1']:.4f}  val_ratio={metrics['val_ratio']:.4f}"
        )

    # --- MC K=50 for lambda=0.05 ---
    mc_normal_scores_list = list(_mc_normal_dict.values())
    mc_abnormal_scores_list = list(_mc_abnormal_dict.values())
    mc_all_scores = np.array(mc_normal_scores_list + mc_abnormal_scores_list)
    mc_metrics = compute_metrics(
        y_true=all_labels,
        scores=mc_all_scores,
        normal_scores=np.array(mc_normal_scores_list),
        abnormal_scores=np.array(mc_abnormal_scores_list),
    )
    mc_result = {"lambda_kld": MC_LAMBDA, "k_mc": MC_K, **mc_metrics}
    print(
        f"  MC K={MC_K}  lambda={MC_LAMBDA}  AUC={mc_metrics['auc_roc']:.4f}  "
        f"F1={mc_metrics['f1']:.4f}  val_ratio={mc_metrics['val_ratio']:.4f}"
    )

    # --- Best lambda by AUC-ROC ---
    best_entry = max(grid_results, key=lambda r: r["auc_roc"])
    best_lambda = best_entry["lambda_kld"]
    print(f"\n  Best lambda by AUC: {best_lambda} (AUC={best_entry['auc_roc']:.4f})")

    # --- Cases caught by ELBO but missed by MSE ---
    print("\nIdentifying cases caught by ELBO but missed by MSE...")
    caught_cases = find_cases_caught_by_elbo_not_mse(
        normal_scores, abnormal_scores, normal_names, abnormal_names
    )
    print(f"  Cases: {len(caught_cases)}")
    for c in caught_cases[:10]:
        print(f"    {c}")

    # --- Verdict ---
    baseline_auc = 0.9663
    baseline_f1 = 0.9663
    best_auc = best_entry["auc_roc"]
    mse_entry = next(r for r in grid_results if r["lambda_kld"] == 0.0)

    delta_auc_vs_baseline = best_auc - baseline_auc
    delta_auc_vs_mse = best_auc - mse_entry["auc_roc"]

    if delta_auc_vs_baseline > 0.001:
        verdict = "IMPROVEMENT"
    elif delta_auc_vs_baseline < -0.001:
        verdict = "REGRESSION"
    else:
        verdict = "NEUTRAL"

    notes = (
        f"Best lambda={best_lambda} achieves AUC={best_auc:.4f} vs baseline "
        f"AUC={baseline_auc:.4f} (delta={delta_auc_vs_baseline:+.4f}). "
        f"Pure-MSE (lambda=0.0) AUC={mse_entry['auc_roc']:.4f}. "
        f"ELBO vs pure-MSE delta: {delta_auc_vs_mse:+.4f}. "
        f"MC K=50 AUC={mc_metrics['auc_roc']:.4f}. "
        f"Caught {len(caught_cases)} days by ELBO that MSE missed."
    )

    # --- Build results dict ---
    results: dict[str, Any] = {
        "experiment": "E3",
        "description": "ELBO anomaly scoring — KLD component added to reconstruction loss",
        "status": "COMPLETE",
        "n_normal_days": n_normal,
        "n_abnormal_days": n_abnormal,
        "baseline_mse": {
            "auc_roc": baseline_auc,
            "f1": baseline_f1,
            "val_ratio": 6.41,
        },
        "grid_search": grid_results,
        "mc_k50": mc_result,
        "best_lambda": best_lambda,
        "cases_caught_by_elbo_not_mse": caught_cases,
        "verdict": verdict,
        "notes": notes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {output_path}")
    print(f"Verdict: {verdict}")
    print(f"Notes  : {notes}")
    print("\n=== E3 Complete ===")


if __name__ == "__main__":
    main()
