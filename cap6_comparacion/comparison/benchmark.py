"""Benchmark script — train all anomaly detection models and compare them.

Trains 7 neural network models + 4 classical ML models on the same data
splits, then evaluates each one with identical metrics for fair comparison.

Usage:
    cd bin
    python -m model.benchmark                  # full benchmark (default params)
    python -m model.benchmark --epochs 50      # quick benchmark with fewer epochs
    python -m model.benchmark --quick          # fast mode (30 epochs, patience 15)

Output:
    benchmark_results/results.json   — full results including ROC curve data
    benchmark_results/summary.csv    — summary table for easy comparison
    Console: ranked comparison table
"""

import argparse
import csv
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score,
)

from .. import PARAMETERS_DIR
from ..preprocessing import Preprocessor
from ..failure_detector import FailureDetector
from ..models import VAE
from .alternative_models import (
    StandardAE, SparseAE, DenoisingAE,
    LSTMAutoencoder, CNNAutoencoder, TransformerAE,
    USADModel,
)
from .classical_models import (
    IsolationForestModel, OneClassSVMModel,
    LOFModel, PCAReconstructionModel,
)
from ..device import select_accelerator, select_precision
from ..main import set_global_seed


RESULTS_DIR = Path('benchmark_results')


# ============================================================================
# CALLBACK — track best model in memory (avoids checkpoint I/O issues)
# ============================================================================

class BestModelTracker(pl.Callback):
    """Track the best model state_dict in memory based on val_ratio."""

    def __init__(self):
        super().__init__()
        self.best_ratio = -float('inf')
        self.best_state = None

    def on_train_epoch_end(self, trainer, pl_module):
        ratio = trainer.callback_metrics.get('val_ratio')
        if ratio is not None and float(ratio) > self.best_ratio:
            self.best_ratio = float(ratio)
            self.best_state = {
                k: v.cpu().clone() for k, v in pl_module.state_dict().items()
            }


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkRunner:
    """Orchestrates training and evaluation of all anomaly detection models."""

    def __init__(self, parameters_file=None, seed=42, epochs=None, patience=None):
        self.seed = seed

        if parameters_file is None:
            parameters_file = str(PARAMETERS_DIR / 'parameters.json')

        with open(parameters_file, 'r') as f:
            raw_params = json.load(f)

        # Apply the same parameter upgrades used by FailureDetector
        self.hparams_dict = FailureDetector._upgrade_parameters(raw_params)

        # Allow CLI overrides
        if epochs is not None:
            self.hparams_dict['epochs'] = epochs
        if patience is not None:
            self.hparams_dict['patience'] = patience

        from argparse import Namespace
        self.hparams = Namespace(**self.hparams_dict)
        self.accelerator = select_accelerator()
        self.precision = select_precision()

        self.results: list[dict] = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self):
        """Load and preprocess data using the same pipeline as FailureDetector."""
        print("\n" + "=" * 60)
        print("LOADING AND PREPROCESSING DATA")
        print("=" * 60)

        preprocessor = Preprocessor(
            self.hparams, seed=self.seed,
            train_split=0.8, val_normal_split=0.1, abnormal_val_split=0.5,
        )

        (x_train_n, y_train_n, _, pids_train_n,
         x_val_normal_n, y_val_normal_n, _, _,
         x_test_normal_n, y_test_normal_n, _, _,
         x_val_abnormal_n, y_val_abnormal_n, _, _,
         x_test_abnormal_n, y_test_abnormal_n, _, _,
         ) = preprocessor.build_dataset(train=True)

        # Flatten for models that expect 2D input
        self.x_train = torch.tensor(x_train_n, dtype=torch.float32).view(x_train_n.shape[0], -1)
        self.y_train = torch.tensor(y_train_n, dtype=torch.float32).view(y_train_n.shape[0], -1)

        self.x_val_normal = torch.tensor(x_val_normal_n, dtype=torch.float32).view(x_val_normal_n.shape[0], -1)
        self.y_val_normal = torch.tensor(y_val_normal_n, dtype=torch.float32).view(y_val_normal_n.shape[0], -1)

        self.x_val_abnormal = torch.tensor(x_val_abnormal_n, dtype=torch.float32).view(x_val_abnormal_n.shape[0], -1)
        self.y_val_abnormal = torch.tensor(y_val_abnormal_n, dtype=torch.float32).view(y_val_abnormal_n.shape[0], -1)

        # Test set — used exclusively for final metric reporting (no leakage)
        self.x_test_normal = torch.tensor(x_test_normal_n, dtype=torch.float32).view(x_test_normal_n.shape[0], -1)
        self.y_test_normal = torch.tensor(y_test_normal_n, dtype=torch.float32).view(y_test_normal_n.shape[0], -1)

        self.x_test_abnormal = torch.tensor(x_test_abnormal_n, dtype=torch.float32).view(x_test_abnormal_n.shape[0], -1)
        self.y_test_abnormal = torch.tensor(y_test_abnormal_n, dtype=torch.float32).view(y_test_abnormal_n.shape[0], -1)

        # Numpy versions for classical models
        self.x_train_np = self.x_train.numpy()
        self.y_train_np = self.y_train.numpy()

        # Store pump IDs for balanced sampling
        self.pids_train = pids_train_n
        self.x_val_normal_np = self.x_val_normal.numpy()
        self.y_val_normal_np = self.y_val_normal.numpy()
        self.x_val_abnormal_np = self.x_val_abnormal.numpy()
        self.y_val_abnormal_np = self.y_val_abnormal.numpy()
        self.x_test_normal_np = self.x_test_normal.numpy()
        self.x_test_abnormal_np = self.x_test_abnormal.numpy()
        self.y_test_normal_np = self.y_test_normal.numpy()
        self.y_test_abnormal_np = self.y_test_abnormal.numpy()

        print(f"\n  Data loaded successfully:")
        print(f"    Train:         {self.x_train.shape}")
        print(f"    Val normal:    {self.x_val_normal.shape}  (model selection)")
        print(f"    Val abnormal:  {self.x_val_abnormal.shape}  (model selection)")
        print(f"    Test normal:   {self.x_test_normal.shape}  (final evaluation)")
        print(f"    Test abnormal: {self.x_test_abnormal.shape}  (final evaluation)")

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------
    def _compute_metrics(self, scores_normal: np.ndarray,
                         scores_abnormal: np.ndarray) -> dict:
        """Compute comparison metrics from per-sample anomaly scores."""
        labels = np.concatenate([
            np.zeros(len(scores_normal)),
            np.ones(len(scores_abnormal)),
        ])
        scores = np.concatenate([scores_normal, scores_abnormal])

        # AUC-ROC
        auc = roc_auc_score(labels, scores)

        # ROC curve points (for plotting)
        fpr, tpr, thresholds = roc_curve(labels, scores)

        # Optimal threshold — Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = int(np.argmax(j_scores))
        optimal_threshold = float(thresholds[optimal_idx])

        # Binary predictions at optimal threshold
        predictions = (scores >= optimal_threshold).astype(int)
        f1 = f1_score(labels, predictions)
        prec = precision_score(labels, predictions, zero_division=0)
        rec = recall_score(labels, predictions, zero_division=0)

        # Mean scores
        mean_normal = float(np.mean(scores_normal))
        mean_abnormal = float(np.mean(scores_abnormal))
        score_ratio = mean_abnormal / (mean_normal + 1e-12)

        return {
            'auc_roc': round(float(auc), 6),
            'f1': round(float(f1), 6),
            'precision': round(float(prec), 6),
            'recall': round(float(rec), 6),
            'mean_score_normal': round(mean_normal, 6),
            'mean_score_abnormal': round(mean_abnormal, 6),
            'score_ratio': round(float(score_ratio), 4),
            'optimal_threshold': round(optimal_threshold, 6),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
        }

    # ------------------------------------------------------------------
    # Neural network model training
    # ------------------------------------------------------------------
    def train_nn_model(self, model_class, model_name: str,
                       extra_hparams: dict | None = None) -> dict:
        """Train a PyTorch Lightning model and evaluate it."""
        print(f"\n{'=' * 60}")
        print(f"  TRAINING: {model_name}")
        print(f"{'=' * 60}")

        hparams = dict(self.hparams_dict)
        if extra_hparams:
            hparams.update(extra_hparams)

        t_start = time.time()

        # Create model
        model = model_class(
            self.x_train, self.y_train,
            self.x_val_normal, self.y_val_normal,
            self.x_val_abnormal, self.y_val_abnormal,
            pids_train=self.pids_train,
            **hparams,
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Callbacks
        tracker = BestModelTracker()
        early_stop = pl.callbacks.EarlyStopping(
            monitor='val_ratio',
            patience=hparams.get('patience', 30),
            verbose=False,
            mode='max',
        )

        # Logger
        log_dir = str(RESULTS_DIR / 'lightning_logs')
        csv_logger = CSVLogger(save_dir=log_dir, name=model_name.replace(' ', '_'))

        # Trainer
        trainer = pl.Trainer(
            max_epochs=hparams.get('epochs', 150),
            callbacks=[tracker, early_stop],
            logger=csv_logger,
            accelerator=self.accelerator,
            precision=self.precision,
            devices=1,
            deterministic=True,
            enable_progress_bar=True,
        )

        trainer.fit(model)
        train_time = time.time() - t_start

        # Restore best model weights
        if tracker.best_state is not None:
            model.load_state_dict(tracker.best_state)
            print(f"  Restored best model (val_ratio: {tracker.best_ratio:.4f})")

        # Evaluate
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            # Use held-out TEST set for final metrics — completely independent
            # from the val set used during training / model selection
            x_n = self.x_test_normal.to(device)
            x_a = self.x_test_abnormal.to(device)
            y_n = self.y_test_normal.to(device)
            y_a = self.y_test_abnormal.to(device)

            # Use reconstruct() if available, otherwise unpack forward tuple
            if hasattr(model, 'reconstruct'):
                recon_n = model.reconstruct(x_n)
                recon_a = model.reconstruct(x_a)
            else:
                out_n = model(x_n)
                recon_n = out_n[0] if isinstance(out_n, tuple) else out_n
                out_a = model(x_a)
                recon_a = out_a[0] if isinstance(out_a, tuple) else out_a

            # Per-sample MAE as anomaly score
            scores_normal = (recon_n - y_n).abs().mean(dim=1).cpu().numpy()
            scores_abnormal = (recon_a - y_a).abs().mean(dim=1).cpu().numpy()

            # Overall MAE
            mae_normal = float(F.l1_loss(recon_n, y_n).cpu())
            mae_abnormal = float(F.l1_loss(recon_a, y_a).cpu())

        metrics = self._compute_metrics(scores_normal, scores_abnormal)

        result = {
            'model_name': model_name,
            'model_type': 'neural_network',
            'n_params': n_params,
            'train_time_s': round(train_time, 1),
            'mae_normal': round(mae_normal, 6),
            'mae_abnormal': round(mae_abnormal, 6),
            'mae_ratio': round(mae_abnormal / (mae_normal + 1e-12), 4),
            **{k: v for k, v in metrics.items() if k not in ('fpr', 'tpr')},
            'roc_curve': {'fpr': metrics['fpr'], 'tpr': metrics['tpr']},
        }

        print(f"\n  Results for {model_name}:")
        print(f"    Parameters:  {n_params:,}")
        print(f"    Train time:  {train_time:.1f}s")
        print(f"    MAE Ratio:   {result['mae_ratio']:.4f}")
        print(f"    AUC-ROC:     {result['auc_roc']:.4f}")
        print(f"    F1:          {result['f1']:.4f}")

        self.results.append(result)
        return result

    # ------------------------------------------------------------------
    # Classical model training
    # ------------------------------------------------------------------
    def train_classical_model(self, model_instance) -> dict:
        """Train a classical ML model and evaluate it."""
        name = model_instance.name
        print(f"\n{'=' * 60}")
        print(f"  TRAINING: {name}")
        print(f"{'=' * 60}")

        t_start = time.time()

        # PCA should reconstruct output variables (y) only — same feature
        # space that NN models use — for a fair apples-to-apples comparison.
        if isinstance(model_instance, PCAReconstructionModel):
            model_instance.fit(self.y_train_np)
            train_time = time.time() - t_start
            scores_normal = model_instance.anomaly_score(self.y_test_normal_np)
            scores_abnormal = model_instance.anomaly_score(self.y_test_abnormal_np)
        else:
            model_instance.fit(self.x_train_np)
            train_time = time.time() - t_start
            # Use held-out TEST set for final metrics
            scores_normal = model_instance.anomaly_score(self.x_test_normal_np)
            scores_abnormal = model_instance.anomaly_score(self.x_test_abnormal_np)

        metrics = self._compute_metrics(scores_normal, scores_abnormal)

        result = {
            'model_name': name,
            'model_type': 'classical_ml',
            'n_params': 'N/A',
            'train_time_s': round(train_time, 1),
            'mae_normal': None,
            'mae_abnormal': None,
            'mae_ratio': None,
            **{k: v for k, v in metrics.items() if k not in ('fpr', 'tpr')},
            'roc_curve': {'fpr': metrics['fpr'], 'tpr': metrics['tpr']},
        }

        # PCA has reconstruction-based scores → can compute MAE ratio
        if isinstance(model_instance, PCAReconstructionModel):
            result['mae_normal'] = round(float(np.mean(scores_normal)), 6)
            result['mae_abnormal'] = round(float(np.mean(scores_abnormal)), 6)
            result['mae_ratio'] = round(
                result['mae_abnormal'] / (result['mae_normal'] + 1e-12), 4
            )

        print(f"\n  Results for {name}:")
        print(f"    Train time:  {train_time:.1f}s")
        print(f"    Score Ratio: {result['score_ratio']:.4f}")
        print(f"    AUC-ROC:     {result['auc_roc']:.4f}")
        print(f"    F1:          {result['f1']:.4f}")

        self.results.append(result)
        return result

    # ------------------------------------------------------------------
    # Full benchmark
    # ------------------------------------------------------------------
    def run(self, model_filter: str = None, append: bool = False):
        """Run the full benchmark across all models.

        Args:
            model_filter: If set, only run models whose name contains this
                          substring (case-insensitive).
            append: If True, load existing results.json and append new results
                    (replacing entries with the same model name).
        """
        set_global_seed(self.seed)
        self.load_data()

        # Load previous results when appending
        if append:
            json_path = RESULTS_DIR / 'results.json'
            if json_path.exists():
                import json as _json
                with open(json_path) as _f:
                    self.results = _json.load(_f)
                print(f"  Loaded {len(self.results)} existing results (append mode)")

        # ── Neural Network Models ──────────────────────────────────────
        nn_models = [
            (VAE,              "VAE (Baseline)"),
            (StandardAE,       "Standard AE"),
            (SparseAE,         "Sparse AE"),
            (DenoisingAE,      "Denoising AE"),
            (LSTMAutoencoder,  "LSTM Autoencoder"),
            (CNNAutoencoder,   "CNN Autoencoder"),
            (TransformerAE,    "Transformer AE"),
            (USADModel,        "USAD"),
        ]

        # ── Classical ML Models ────────────────────────────────────────
        classical_models = [
            IsolationForestModel(),
            OneClassSVMModel(),
            LOFModel(),
            PCAReconstructionModel(),
        ]

        # Apply model filter if specified
        if model_filter:
            filt = model_filter.lower()
            nn_models = [(c, n) for c, n in nn_models if filt in n.lower()]
            classical_models = [m for m in classical_models if filt in m.name.lower()]

        total = len(nn_models) + len(classical_models)
        idx = 0

        print("\n" + "=" * 60)
        print("PHASE 1: NEURAL NETWORK MODELS")
        print("=" * 60)

        for model_class, name in nn_models:
            idx += 1
            print(f"\n[{idx}/{total}]", end="")
            try:
                set_global_seed(self.seed)
                self.train_nn_model(model_class, name)
            except Exception as e:
                print(f"\n  FAILED: {name} — {e}")
                self.results.append({
                    'model_name': name,
                    'model_type': 'neural_network',
                    'error': str(e),
                })

        print("\n" + "=" * 60)
        print("PHASE 2: CLASSICAL ML MODELS")
        print("=" * 60)

        for model_instance in classical_models:
            idx += 1
            print(f"\n[{idx}/{total}]", end="")
            try:
                self.train_classical_model(model_instance)
            except Exception as e:
                print(f"\n  FAILED: {model_instance.name} — {e}")
                self.results.append({
                    'model_name': model_instance.name,
                    'model_type': 'classical_ml',
                    'error': str(e),
                })

        self.save_results()
        self.print_summary()

        return self.results

    # ------------------------------------------------------------------
    # Results persistence
    # ------------------------------------------------------------------
    def save_results(self):
        """Save benchmark results to JSON and CSV."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Deduplicate: keep last entry for each model name
        seen = {}
        for r in self.results:
            seen[r['model_name']] = r
        self.results = list(seen.values())

        # Full JSON (including ROC curve data)
        json_path = RESULTS_DIR / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
        print(f"\n  Full results saved to {json_path}")

        # Summary CSV (without ROC curve arrays)
        csv_path = RESULTS_DIR / 'summary.csv'
        fieldnames = [
            'model_name', 'model_type', 'n_params', 'train_time_s',
            'mae_normal', 'mae_abnormal', 'mae_ratio',
            'auc_roc', 'f1', 'precision', 'recall',
            'score_ratio', 'optimal_threshold',
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in self.results:
                if 'error' not in r:
                    writer.writerow(r)
        print(f"  Summary CSV saved to {csv_path}")

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    def print_summary(self):
        """Print a ranked summary table to the console."""
        valid = [r for r in self.results if 'error' not in r]
        valid.sort(key=lambda r: r.get('auc_roc', 0), reverse=True)

        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS — RANKED BY AUC-ROC")
        print("=" * 100)

        header = (
            f"{'Rank':<5} {'Model':<22} {'Type':<12} {'AUC-ROC':<10} "
            f"{'F1':<8} {'MAE Ratio':<12} {'Score Ratio':<13} "
            f"{'Params':<12} {'Time(s)':<8}"
        )
        print(header)
        print("-" * len(header))

        for i, r in enumerate(valid, 1):
            mae_r = f"{r['mae_ratio']:.4f}" if r.get('mae_ratio') is not None else "N/A"
            n_p = f"{r['n_params']:,}" if isinstance(r.get('n_params'), int) else str(r.get('n_params', 'N/A'))
            m_type = 'NN' if r['model_type'] == 'neural_network' else 'Classical'
            print(
                f"{i:<5} {r['model_name']:<22} {m_type:<12} "
                f"{r['auc_roc']:<10.4f} {r['f1']:<8.4f} "
                f"{mae_r:<12} {r['score_ratio']:<13.4f} "
                f"{n_p:<12} {r['train_time_s']:<8.1f}"
            )

        failed = [r for r in self.results if 'error' in r]
        if failed:
            print(f"\n  ({len(failed)} model(s) failed — see results.json for details)")

        if valid:
            best = valid[0]
            print(f"\n  BEST MODEL: {best['model_name']} (AUC-ROC: {best['auc_roc']:.4f})")

        print("=" * 100)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark anomaly detection models')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs (default: from parameters.json)')
    parser.add_argument('--patience', type=int, default=None,
                        help='Override early-stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 30 epochs, patience 15')
    parser.add_argument('--model', type=str, default=None,
                        help='Run only models whose name contains this string (case-insensitive)')
    parser.add_argument('--append', action='store_true',
                        help='Append results to existing results.json instead of overwriting')
    parser.add_argument('--parameters', type=str, default=None,
                        help='Path to parameters JSON (overrides default parameters.json)')
    args = parser.parse_args()

    epochs = args.epochs
    patience = args.patience

    if args.quick:
        epochs = epochs or 30
        patience = patience or 15
        print("Quick mode: 30 epochs, patience 15")

    # Suppress noisy warnings during benchmark
    warnings.filterwarnings('ignore', category=UserWarning, module='pytorch_lightning')

    runner = BenchmarkRunner(
        parameters_file=args.parameters,
        seed=args.seed,
        epochs=epochs,
        patience=patience,
    )
    runner.run(model_filter=args.model, append=args.append)


if __name__ == '__main__':
    main()
