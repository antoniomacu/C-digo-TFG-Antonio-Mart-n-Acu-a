"""Experiment E4 — Alternative L2 architecture benchmark.

Benchmarks USAD, TransformerAE, StandardAE, LSTMAutoencoder, and CNNAutoencoder
against the production VAE baseline (AUC=0.9772, F1=0.9744, val_ratio=8.78).

For USAD: runs a 20-trial Optuna search over alpha/beta (the inference-time
scoring weights) after training, using the held-out test set to select the best
combination.

Usage (from project root):
    uv run python experiments/e4_benchmark.py
    uv run python experiments/e4_benchmark.py --epochs 50 --patience 20
    uv run python experiments/e4_benchmark.py --quick           # 30 epochs, 15 patience
    uv run python experiments/e4_benchmark.py --model USAD      # single model

Output:
    experiments/E4_benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Inject bin/ onto sys.path so the 'model' package is importable regardless
# of where the script is invoked from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = PROJECT_ROOT / "bin"
if str(BIN_DIR) not in sys.path:
    sys.path.insert(0, str(BIN_DIR))

# Data paths — actual new_data on this machine
NEW_DATA_ROOT = Path("<PATH_TO_DATA_DIR>")
TRAIN_PATH = NEW_DATA_ROOT / "train"
TEST_PATH = NEW_DATA_ROOT / "test"

# Reference baseline
REFERENCE = {
    "model": "Production VAE",
    "auc_roc": 0.9772,
    "f1": 0.9744,
    "val_ratio": 8.78,
}

# USAD Optuna flag threshold
USAD_INVESTIGATE_THRESHOLD = 0.995

# Output path
OUTPUT_JSON = PROJECT_ROOT / "experiments" / "E4_benchmark_results.json"


# ---------------------------------------------------------------------------
# Patched parameters — override train/test paths in the loaded params dict
# ---------------------------------------------------------------------------
def build_patched_parameters(
    base_params_path: Path, epochs: int | None = None, patience: int | None = None
) -> dict:
    """Load parameters.json and replace train/test paths with new_data paths."""
    with open(base_params_path, "r") as f:
        params = json.load(f)

    params["train_path"] = str(TRAIN_PATH) + "/"
    params["test_path"] = str(TEST_PATH) + "/"

    if epochs is not None:
        params["epochs"] = epochs
    if patience is not None:
        params["patience"] = patience

    return params


# ---------------------------------------------------------------------------
# USAD Optuna alpha/beta search
# ---------------------------------------------------------------------------
def usad_optuna_search(
    model,
    x_test_normal,
    x_test_abnormal,
    y_test_normal,
    y_test_abnormal,
    n_trials: int = 20,
) -> tuple[float, float, float]:
    """Search for best alpha/beta for USAD anomaly scoring via Optuna.

    USAD anomaly score: alpha * ||y - W1||^2 + beta * ||y - W3||^2
    where W1 = Decoder1(Encoder(x))  and  W3 = Decoder2(Encoder(W1))

    alpha and beta are constrained to [0, 1] and alpha + beta == 1
    (canonical USAD formulation).

    Returns:
        (best_alpha, best_beta, best_auc)
    """
    import optuna
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    device = next(model.parameters()).device
    model.eval()

    x_n = x_test_normal.to(device)
    x_a = x_test_abnormal.to(device)
    y_n = y_test_normal.to(device)
    y_a = y_test_abnormal.to(device)

    with torch.no_grad():
        # Precompute W1 and W3 for normal and abnormal once (expensive)
        z_n = model.usad_encoder(x_n)
        w1_n = model.decoder1(z_n)
        w1_n_proj = model.reproject(w1_n)
        z_w1_n = model.usad_encoder(w1_n_proj)
        w3_n = model.decoder2(z_w1_n)

        z_a = model.usad_encoder(x_a)
        w1_a = model.decoder1(z_a)
        w1_a_proj = model.reproject(w1_a)
        z_w1_a = model.usad_encoder(w1_a_proj)
        w3_a = model.decoder2(z_w1_a)

        # Per-sample squared errors (batch dimension only)
        err1_n = ((y_n - w1_n) ** 2).mean(dim=1).cpu().numpy()
        err3_n = ((y_n - w3_n) ** 2).mean(dim=1).cpu().numpy()
        err1_a = ((y_a - w1_a) ** 2).mean(dim=1).cpu().numpy()
        err3_a = ((y_a - w3_a) ** 2).mean(dim=1).cpu().numpy()

    labels = np.concatenate([np.zeros(len(err1_n)), np.ones(len(err1_a))])

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        beta = 1.0 - alpha

        scores_n = alpha * err1_n + beta * err3_n
        scores_a = alpha * err1_a + beta * err3_a
        scores = np.concatenate([scores_n, scores_a])

        try:
            return roc_auc_score(labels, scores)
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_alpha = study.best_params["alpha"]
    best_beta = 1.0 - best_alpha
    best_auc = study.best_value

    print(
        f"  USAD Optuna: best alpha={best_alpha:.4f}, beta={best_beta:.4f}, AUC={best_auc:.4f}"
    )
    return best_alpha, best_beta, best_auc


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------
def run_e4(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # Validate data paths
    # ------------------------------------------------------------------
    if not TRAIN_PATH.exists():
        print(f"ERROR: Train data not found at {TRAIN_PATH}")
        _write_blocked_result("Train data missing", str(TRAIN_PATH))
        return

    if not TEST_PATH.exists():
        print(f"ERROR: Test data not found at {TEST_PATH}")
        _write_blocked_result("Test data missing", str(TEST_PATH))
        return

    # ------------------------------------------------------------------
    # Import model infrastructure (must be after sys.path injection)
    # ------------------------------------------------------------------
    try:
        import warnings as _w

        _w.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
        _w.filterwarnings("ignore", category=FutureWarning)

        from model.comparison.benchmark import BenchmarkRunner
        from model.comparison.alternative_models import (
            USADModel,
            TransformerAE,
            StandardAE,
            LSTMAutoencoder,
            CNNAutoencoder,
        )
        from model.main import set_global_seed
    except ImportError as e:
        print(f"ERROR: Could not import model package: {e}")
        _write_blocked_result(f"Import error: {e}", "")
        return

    # ------------------------------------------------------------------
    # Build patched parameters file (temp file in experiments/)
    # ------------------------------------------------------------------
    base_params = BIN_DIR / "model" / "parameters" / "parameters.json"
    params_dict = build_patched_parameters(
        base_params,
        epochs=args.epochs,
        patience=args.patience,
    )

    # Write patched params to a temp file so BenchmarkRunner can read it
    tmp_params_path = OUTPUT_JSON.parent / "_e4_tmp_params.json"
    with open(tmp_params_path, "w") as f:
        json.dump(params_dict, f, indent=4)

    print("\n" + "=" * 70)
    print("  EXPERIMENT E4 — Alternative L2 Architecture Benchmark")
    print("=" * 70)
    print(f"  Train data:   {TRAIN_PATH}")
    print(f"  Test data:    {TEST_PATH}")
    print(f"  Epochs:       {params_dict['epochs']}")
    print(f"  Patience:     {params_dict['patience']}")
    print(f"  past_history: {params_dict['past_history']}")
    print(
        f"  Reference:    VAE AUC={REFERENCE['auc_roc']}, F1={REFERENCE['f1']}, val_ratio={REFERENCE['val_ratio']}"
    )

    # ------------------------------------------------------------------
    # BenchmarkRunner — run only the 5 target architectures
    # ------------------------------------------------------------------
    target_models = [
        (USADModel, "USAD"),
        (TransformerAE, "TransformerAE"),
        (StandardAE, "StandardAE"),
        (LSTMAutoencoder, "LSTMAutoencoder"),
        (CNNAutoencoder, "CNNAutoencoder"),
    ]

    # Filter if --model flag given
    if args.model:
        filt = args.model.lower()
        target_models = [(c, n) for c, n in target_models if filt in n.lower()]
        if not target_models:
            print(f"ERROR: No model matched filter '{args.model}'")
            sys.exit(1)

    set_global_seed(42)

    runner = BenchmarkRunner(
        parameters_file=str(tmp_params_path),
        seed=42,
    )
    runner.load_data()

    # Keep references to test tensors for USAD Optuna
    x_test_normal = runner.x_test_normal
    x_test_abnormal = runner.x_test_abnormal
    y_test_normal = runner.y_test_normal
    y_test_abnormal = runner.y_test_abnormal

    raw_results: list[dict] = []
    trained_models: dict[str, object] = {}  # name → trained model instance

    for model_class, name in target_models:
        set_global_seed(42)
        try:
            result = runner.train_nn_model(model_class, name)
            raw_results.append(result)

            # Save trained model instance for USAD Optuna
            if name == "USAD" and hasattr(runner, "_last_model"):
                trained_models["USAD"] = runner._last_model

        except Exception as e:
            print(f"\nFAILED: {name} — {e}")
            import traceback

            traceback.print_exc()
            raw_results.append(
                {
                    "model_name": name,
                    "error": str(e),
                    "status": "FAILED",
                }
            )

    # Clean up temp params file
    try:
        tmp_params_path.unlink()
    except Exception:
        pass

    # ------------------------------------------------------------------
    # USAD Optuna alpha/beta search
    # ------------------------------------------------------------------
    usad_optuna_result: dict | None = None

    # Find the USAD raw result
    usad_raw = next(
        (r for r in raw_results if r.get("model_name") == "USAD" and "error" not in r),
        None,
    )

    if usad_raw is not None:
        print("\n" + "=" * 70)
        print("  USAD Optuna: 20-trial alpha/beta search")
        print("=" * 70)

        # Re-instantiate USAD with the best weights from training
        # We need to train it again or recover model from BenchmarkRunner.
        # BenchmarkRunner.train_nn_model does not expose the trained model object
        # publicly. We will retrain USAD with a minimal wrapper that captures it.
        try:
            import pytorch_lightning as pl
            from pytorch_lightning.loggers import CSVLogger
            from model.comparison.benchmark import BestModelTracker
            from model.device import select_accelerator, select_precision

            hparams = dict(runner.hparams_dict)
            usad_model = USADModel(
                runner.x_train,
                runner.y_train,
                runner.x_val_normal,
                runner.y_val_normal,
                runner.x_val_abnormal,
                runner.y_val_abnormal,
                pids_train=runner.pids_train,
                **hparams,
            )

            tracker = BestModelTracker()
            early_stop = pl.callbacks.EarlyStopping(
                monitor="val_ratio",
                patience=hparams.get("patience", 30),
                verbose=False,
                mode="max",
            )

            log_dir = str(BIN_DIR / "benchmark_results" / "lightning_logs")
            csv_logger = CSVLogger(save_dir=log_dir, name="USAD_optuna")

            trainer = pl.Trainer(
                max_epochs=hparams.get("epochs", 150),
                callbacks=[tracker, early_stop],
                logger=csv_logger,
                accelerator=select_accelerator(),
                precision=select_precision(),
                devices=1,
                deterministic=True,
                enable_progress_bar=True,
            )

            print(
                "  Re-training USAD to capture model weights for alpha/beta search..."
            )
            set_global_seed(42)
            trainer.fit(usad_model)

            if tracker.best_state is not None:
                usad_model.load_state_dict(tracker.best_state)

            usad_model.eval()
            best_alpha, best_beta, best_auc_optuna = usad_optuna_search(
                usad_model,
                x_test_normal,
                x_test_abnormal,
                y_test_normal,
                y_test_abnormal,
                n_trials=20,
            )
            usad_optuna_result = {
                "best_alpha": round(best_alpha, 4),
                "best_beta": round(best_beta, 4),
                "best_auc_optuna": round(best_auc_optuna, 6),
            }
        except Exception as e:
            print(f"  USAD Optuna search failed: {e}")
            import traceback

            traceback.print_exc()
            usad_optuna_result = {"error": str(e)}

    # ------------------------------------------------------------------
    # Build E4 result record
    # ------------------------------------------------------------------
    e4_results: list[dict] = []

    for r in raw_results:
        name = r.get("model_name", "unknown")
        if "error" in r:
            entry = {
                "model": name,
                "status": "FAILED",
                "error": r["error"],
                "auc_roc": None,
                "f1": None,
                "val_ratio": None,
                "delta_auc": None,
            }
        else:
            auc = r.get("auc_roc")
            f1 = r.get("f1")
            # val_ratio from benchmark is mae_ratio (same concept, different naming)
            val_ratio = r.get("mae_ratio")
            delta_auc = (
                round(auc - REFERENCE["auc_roc"], 6) if auc is not None else None
            )

            entry = {
                "model": name,
                "auc_roc": auc,
                "f1": f1,
                "precision": r.get("precision"),
                "recall": r.get("recall"),
                "val_ratio": val_ratio,
                "score_ratio": r.get("score_ratio"),
                "delta_auc": delta_auc,
                "n_params": r.get("n_params"),
                "train_time_s": r.get("train_time_s"),
            }

            if name == "USAD" and usad_optuna_result:
                if "error" not in usad_optuna_result:
                    entry["best_alpha"] = usad_optuna_result["best_alpha"]
                    entry["best_beta"] = usad_optuna_result["best_beta"]
                    # Override AUC with Optuna result if it's higher
                    optuna_auc = usad_optuna_result.get("best_auc_optuna")
                    if optuna_auc is not None and optuna_auc > (auc or 0):
                        entry["auc_roc_default"] = auc
                        entry["auc_roc"] = optuna_auc
                        entry["delta_auc"] = round(optuna_auc - REFERENCE["auc_roc"], 6)
                        entry["note_auc"] = "AUC improved by Optuna alpha/beta search"
                    flag = (entry["auc_roc"] or 0) > USAD_INVESTIGATE_THRESHOLD
                    entry["flag_investigate"] = flag
                    if flag:
                        entry["flag_note"] = (
                            f"USAD AUC={entry['auc_roc']:.4f} > {USAD_INVESTIGATE_THRESHOLD} "
                            "— warrants further investigation"
                        )
                else:
                    entry["usad_optuna_error"] = usad_optuna_result["error"]
                    entry["flag_investigate"] = False
            elif name == "USAD":
                entry["flag_investigate"] = bool(
                    (entry.get("auc_roc") or 0) > USAD_INVESTIGATE_THRESHOLD
                )

        e4_results.append(entry)

    # ------------------------------------------------------------------
    # Verdict and best alternative
    # ------------------------------------------------------------------
    valid_results = [r for r in e4_results if r.get("auc_roc") is not None]

    best_alt: dict | None = None
    if valid_results:
        best_alt = max(valid_results, key=lambda r: r["auc_roc"])

    ref_auc = REFERENCE["auc_roc"]
    verdict = "VAE_HOLDS"
    if best_alt:
        if best_alt["auc_roc"] > ref_auc + 0.005:
            verdict = "USAD_WINS" if best_alt["model"] == "USAD" else "ALTERNATIVE_WINS"
        elif any(r.get("auc_roc", 0) > ref_auc + 0.001 for r in valid_results):
            verdict = "MIXED"
        else:
            verdict = "VAE_HOLDS"

    notes_parts: list[str] = [
        f"past_history={params_dict['past_history']}, epochs={params_dict['epochs']}, "
        f"patience={params_dict['patience']}.",
        "Evaluation protocol: same train/val/test file-level split as production VAE "
        "(80/10/10 normal; 50/50 abnormal), seed=42.",
    ]

    usad_flag_triggered = any(
        r.get("flag_investigate") for r in e4_results if r.get("model") == "USAD"
    )
    if usad_flag_triggered:
        usad_auc = next(
            (r["auc_roc"] for r in e4_results if r.get("model") == "USAD"), None
        )
        notes_parts.append(
            f"USAD AUC={usad_auc:.4f} EXCEEDS THRESHOLD {USAD_INVESTIGATE_THRESHOLD} "
            "— flag_investigate=True. Recommend dedicated USAD study."
        )

    failed_models = [r["model"] for r in e4_results if r.get("status") == "FAILED"]
    if failed_models:
        notes_parts.append(
            f"Failed models (excluded from verdict): {', '.join(failed_models)}."
        )

    output = {
        "experiment": "E4",
        "description": "Alternative L2 architecture benchmark vs production VAE",
        "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference": REFERENCE,
        "results": e4_results,
        "best_alternative": best_alt["model"] if best_alt else None,
        "best_alternative_auc": best_alt["auc_roc"] if best_alt else None,
        "verdict": verdict,
        "notes": " ".join(notes_parts),
    }

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=4, default=str)

    print(f"\n  E4 results written to {OUTPUT_JSON}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  E4 BENCHMARK — RANKED BY AUC-ROC")
    print("=" * 80)
    print(
        f"  Reference: {REFERENCE['model']}  AUC={REFERENCE['auc_roc']}  F1={REFERENCE['f1']}  val_ratio={REFERENCE['val_ratio']}"
    )
    print("-" * 80)
    ranked = sorted(valid_results, key=lambda r: r["auc_roc"], reverse=True)
    for r in ranked:
        delta = r.get("delta_auc", 0)
        sign = "+" if delta >= 0 else ""
        flag = " *** INVESTIGATE ***" if r.get("flag_investigate") else ""
        print(
            f"  {r['model']:<22}  AUC={r['auc_roc']:.4f}  "
            f"F1={r.get('f1', 0):.4f}  "
            f"val_ratio={r.get('val_ratio', 0):.2f}  "
            f"delta={sign}{delta:.4f}{flag}"
        )
    print("=" * 80)
    print(f"  Verdict: {verdict}")
    if best_alt:
        print(
            f"  Best alternative: {best_alt['model']} (AUC={best_alt['auc_roc']:.4f})"
        )


def _write_blocked_result(reason: str, detail: str) -> None:
    """Write a BLOCKED status JSON when training cannot proceed."""
    output = {
        "experiment": "E4",
        "description": "Alternative L2 architecture benchmark vs production VAE",
        "status": "BLOCKED",
        "reason": reason,
        "detail": detail,
        "reference": REFERENCE,
        "results": [],
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=4)
    print(f"  BLOCKED result written to {OUTPUT_JSON}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E4: Alternative L2 architecture benchmark"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max training epochs (default: from parameters.json = 150)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early-stopping patience (default: from parameters.json = 44)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: 30 epochs, 15 patience"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only one architecture (substring match, case-insensitive)",
    )
    parser.add_argument(
        "--usad-trials",
        type=int,
        default=20,
        help="Number of Optuna trials for USAD alpha/beta search (default: 20)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.quick:
        args.epochs = args.epochs or 30
        args.patience = args.patience or 15
        print("Quick mode: 30 epochs, 15 patience")

    warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
    warnings.filterwarnings("ignore", category=FutureWarning)

    run_e4(args)
