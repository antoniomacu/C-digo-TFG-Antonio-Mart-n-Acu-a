# Fine-tuning — Optuna Bayesian (TPE) Search over VAE hyperparameters
#
# Adapted from VAE_finetun.ipynb for production use.
# Uses the same model architecture (models.py) and preprocessing (preprocessing.py)
# as the main training pipeline to ensure consistency.

import json
import time
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score

import optuna
from optuna.samplers import TPESampler
import warnings
import logging

from .preprocessing import Preprocessor
from .models import Encoder, Decoder
from .device import select_accelerator, select_precision


# Default search spaces — can be overridden at construction
#   Architecture: VAE (Encoder → latent → Decoder), loss = mse + β·KLD
#   Optimizer: AdamW + CosineAnnealingWarmRestarts(T_0=25)
#
# Categorical params are discrete by nature (architecture choices).
# Continuous params (loss/optimizer) use TPE with float ranges for fine resolution.
DEFAULT_CATEGORICAL_SPACE = {
    "past_history": [12, 18, 24, 36],
    "latent_dim": [16, 24, 32, 48, 64],
    "layer_sizes": [
        "256,128,64",
        "512,256,128",
        "512,256,128,64",
    ],
    "batch_size": [32, 64, 128],
    "batch_norm": [True, False],
}

# Continuous ranges: (low, high, log_scale)
# log_scale=True for params spanning orders of magnitude (kld_beta, lr, weight_decay).
DEFAULT_CONTINUOUS_SPACE = {
    "stdev":        (0.01, 0.25, False),
    "kld_beta":     (1e-5, 5e-2, True),
    "lr":           (1e-4, 1e-2, True),
    "weight_decay": (1e-6, 5e-3, True),
}

# LSTM-VAE search spaces (RD-5)
LSTM_VAE_CATEGORICAL_SPACE = {
    "past_history": [12, 18, 24, 36],
    "latent_dim": [16, 24, 32, 48, 64],
    "lstm_hidden": [64, 128, 256],
    "lstm_layers": [1, 2, 3],
    "batch_size": [32, 64, 128],
}

LSTM_VAE_CONTINUOUS_SPACE = {
    "stdev":              (0.01, 0.25, False),
    "kld_beta":           (1e-5, 5e-2, True),
    "lr":                 (1e-4, 1e-2, True),
    "weight_decay":       (1e-6, 5e-3, True),
    "lstm_dropout":       (0.0, 0.3, False),
    "kld_warmup_epochs":  (20, 80, False),
}

# TCN-VAE search spaces (RD-5)
TCN_VAE_CATEGORICAL_SPACE = {
    "past_history": [12, 18, 24, 36],
    "latent_dim": [16, 24, 32, 48, 64],
    "tcn_channels": ["64,128", "64,128,256", "128,256,512"],
    "tcn_kernel_size": [3, 5, 7],
    "batch_size": [32, 64, 128],
}

TCN_VAE_CONTINUOUS_SPACE = {
    "stdev":              (0.01, 0.25, False),
    "kld_beta":           (1e-5, 5e-2, True),
    "lr":                 (1e-4, 1e-2, True),
    "weight_decay":       (1e-6, 5e-3, True),
    "tcn_dropout":        (0.0, 0.3, False),
    "kld_warmup_epochs":  (20, 80, False),
}


class HyperparameterTuner:
    """Optuna Bayesian (TPE) search over VAE hyperparameters.

    Workflow
    --------
    1.  Pre-compute windowed datasets for every ``past_history`` value
        in the categorical space (one-time cost; cached in memory).
    2.  For each Optuna trial, sample categorical params from discrete
        lists and continuous params (stdev, kld_beta, lr, weight_decay)
        from float ranges via ``suggest_float``.  Train the model for
        ``grid_epochs`` (with early stopping) and evaluate the
        anomaly-separation *ratio* (MAE_abnormal / MAE_normal).
    3.  Report the top configurations and return the best parameters as
        a dictionary that can be written back to ``parameters.json``.
    """

    def __init__(
        self,
        parameters_file: str,
        categorical_space: dict | None = None,
        continuous_space: dict | None = None,
        n_trials: int = 300,
        data_fraction: float = 1.0,
        grid_epochs: int | None = None,
        grid_patience: int | None = None,
        seed: int = 42,
        train_split: float = 0.8,
        val_normal_split: float = 0.1,
        abnormal_val_split: float = 0.5,
        save_results: bool = True,
    ):
        """
        Args:
            parameters_file: Path to the base parameters JSON.
            categorical_space: Dict mapping param names to lists of discrete
                               candidate values.  Defaults to
                               ``DEFAULT_CATEGORICAL_SPACE``.
            continuous_space: Dict mapping param names to ``(low, high, log)``
                              tuples for ``suggest_float``.  Defaults to
                              ``DEFAULT_CONTINUOUS_SPACE``.
            n_trials: Number of Optuna trials to run (default 300).
            data_fraction: Fraction of training data used per trial (1.0 = all).
            grid_epochs: Maximum epochs per trial.  Defaults to the ``epochs``
                         value from the base parameters file (same as production).
            grid_patience: Early-stopping patience per trial.  Defaults to the
                           ``patience`` value from the base parameters file.
            seed: Random seed.
            train_split / val_normal_split / abnormal_val_split: Split fractions.
            save_results: Whether to save the results DataFrame to CSV.
        """
        # Load base parameters
        with open(parameters_file, "r") as f:
            self.base_params: dict = json.load(f)

        self.categorical_space = categorical_space or DEFAULT_CATEGORICAL_SPACE
        self.continuous_space = continuous_space or DEFAULT_CONTINUOUS_SPACE
        self.n_trials = n_trials
        self.data_fraction = data_fraction
        self.grid_epochs = grid_epochs if grid_epochs is not None else self.base_params.get("epochs", 150)
        self.grid_patience = grid_patience if grid_patience is not None else self.base_params.get("patience", 44)
        self.seed = seed
        self.save_results = save_results
        self.rng = np.random.RandomState(seed)

        # Preprocessor (shared across all trials)
        from argparse import Namespace
        self.base_hparams = Namespace(**self.base_params)
        self.preprocessor = Preprocessor(
            self.base_hparams,
            seed=seed,
            train_split=train_split,
            val_normal_split=val_normal_split,
            abnormal_val_split=abnormal_val_split,
        )

        self.windowed_datasets: dict = {}
        self.accelerator = select_accelerator()
        self.precision = select_precision()

        # Resolve the torch device once (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._amp_device_type = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
            self._amp_device_type = "cpu"   # MPS does not support autocast
        else:
            self._device = torch.device("cpu")
            self._amp_device_type = "cpu"

        # CUDA-specific optimisations (safe no-ops on other backends)
        if self._device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("medium")  # TF32 on Ampere+

    # ------------------------------------------------------------------
    # Dataset pre-computation
    # ------------------------------------------------------------------
    def _precompute_datasets(self):
        """Load raw data once, then window and cache for every past_history value.
        
        Splits are done at FILE level (each CSV = one pump-day) to prevent
        data leakage — same approach as Preprocessor.build_dataset().
        """

        print("Pre-computing windowed datasets for all past_history values...")
        print("(This runs once — all trials will reuse these cached datasets)\n")

        t_start = time.time()

        # ===== SPLIT FILES FIRST (before loading any data) =====
        all_train_files = self.preprocessor._get_file_list(
            self.base_hparams.train_path, is_training=True
        )
        train_files, temp_files = train_test_split(
            all_train_files, test_size=0.2, random_state=self.seed
        )
        val_files, _ = train_test_split(
            temp_files, test_size=0.5, random_state=self.seed
        )

        all_abnormal_files = self.preprocessor._get_file_list(
            self.base_hparams.test_path, is_training=False
        )
        val_abn_files, _ = train_test_split(
            all_abnormal_files, test_size=0.5, random_state=self.seed
        )

        print(f"\n  File-level split (no leakage):")
        print(f"    Training files:       {len(train_files)}")
        print(f"    Validation files:     {len(val_files)}")
        print(f"    Val abnormal files:   {len(val_abn_files)}")

        # ===== LOAD EACH SPLIT INDEPENDENTLY =====
        print("\n  Loading training files...")
        df_train_raw = self.preprocessor._load_files(
            self.base_hparams.train_path, train_files
        )
        self.norm_params_cached = self.preprocessor.get_normalization_params(df_train_raw)

        df_train_norm = self.preprocessor.normalize_data(
            df_train_raw, self.norm_params_cached, self.base_hparams.norm_method
        )
        df_train_norm = self.preprocessor.rebuild_pump_id(df_train_norm)

        print("  Loading validation (normal) files...")
        df_val_raw = self.preprocessor._load_files(
            self.base_hparams.train_path, val_files
        )
        df_val_norm = self.preprocessor.normalize_data(
            df_val_raw, self.norm_params_cached, self.base_hparams.norm_method
        )
        df_val_norm = self.preprocessor.rebuild_pump_id(df_val_norm)

        print("  Loading validation (abnormal) files...")
        df_abnormal_raw = self.preprocessor._load_files(
            self.base_hparams.test_path, val_abn_files
        )
        df_abnormal_norm = self.preprocessor.normalize_data(
            df_abnormal_raw, self.norm_params_cached, self.base_hparams.norm_method
        )
        df_abnormal_norm = self.preprocessor.rebuild_pump_id(df_abnormal_norm)

        past_history_values = self.categorical_space.get(
            "past_history", [self.base_hparams.past_history]
        )

        for ph in past_history_values:
            print(f"\n  Windowing past_history={ph}...")

            x_train, y_train, _, pids_train = self.preprocessor.build_preprocessing_window(
                df_train_norm.copy(), ph
            )
            x_val_n, y_val_n, ts_val_n, pids_val_n = self.preprocessor.build_preprocessing_window(
                df_val_norm.copy(), ph
            )
            x_val_a, y_val_a, ts_val_a, pids_val_a = self.preprocessor.build_preprocessing_window(
                df_abnormal_norm.copy(), ph
            )

            # Optionally subsample training data (data_fraction < 1.0)
            n_total = x_train.shape[0]
            if self.data_fraction < 1.0:
                n_subset = max(1, int(n_total * self.data_fraction))
                subset_idx = self.rng.choice(n_total, size=n_subset, replace=False)
                x_tr_use = x_train[subset_idx]
                y_tr_use = y_train[subset_idx]
                pids_use = np.array(pids_train)[subset_idx]
            else:
                n_subset = n_total
                x_tr_use = x_train
                y_tr_use = y_train
                pids_use = np.array(pids_train)

            # Build per-sample weights for pump-balanced sampling
            unique_pids, pid_counts = np.unique(pids_use, return_counts=True)
            weight_map = {pid: 1.0 / cnt for pid, cnt in zip(unique_pids, pid_counts)}
            sample_weights = torch.tensor(
                [weight_map[p] for p in pids_use], dtype=torch.float32,
            )

            # Build tensors and move them to GPU immediately so every
            # trial trains from device-resident data (no CPU→GPU copies).
            self.windowed_datasets[ph] = {
                "x_train": torch.tensor(x_tr_use, dtype=torch.float32).view(n_subset, -1).to(self._device),
                "y_train": torch.tensor(y_tr_use, dtype=torch.float32).view(n_subset, -1).to(self._device),
                "sample_weights": sample_weights.to(self._device),  # pump-balanced weights
                "x_val_normal": torch.tensor(x_val_n, dtype=torch.float32).view(x_val_n.shape[0], -1).to(self._device),
                "y_val_normal": torch.tensor(y_val_n, dtype=torch.float32).view(y_val_n.shape[0], -1).to(self._device),
                "x_val_abnormal": torch.tensor(x_val_a, dtype=torch.float32).view(x_val_a.shape[0], -1).to(self._device),
                "y_val_abnormal": torch.tensor(y_val_a, dtype=torch.float32).view(y_val_a.shape[0], -1).to(self._device),
                "ts_val_normal": ts_val_n,
                "pids_val_normal": np.array(pids_val_n),
                "ts_val_abnormal": ts_val_a,
                "pids_val_abnormal": np.array(pids_val_a),
            }

            print(
                f"    ✓ ph={ph}: train={n_subset} ({self.data_fraction*100:.0f}% of {n_total}), "
                f"val_normal={x_val_n.shape[0]}, val_abnormal={x_val_a.shape[0]}"
            )

        # Clean up to free memory
        del df_train_raw, df_train_norm, df_val_raw, df_val_norm
        del df_abnormal_raw, df_abnormal_norm

        elapsed = time.time() - t_start
        print(f"\n✓ All datasets pre-computed in {elapsed:.1f}s")
        print(f"✓ TPE search: {self.n_trials} trials planned")
        print(f"✓ Training regime per trial: max {self.grid_epochs} epochs, patience {self.grid_patience}")
        print(f"✓ Data fraction: {self.data_fraction*100:.0f}% of training data")
        print(f"✓ Memory cached: {len(self.windowed_datasets)} windowed datasets (tensors ready)")

    # ------------------------------------------------------------------
    # Raw PyTorch training (bypasses Lightning for ~50x speedup)
    # ------------------------------------------------------------------
    def _train_vae_raw(self, trial_params: dict, x_train: torch.Tensor,
                       y_train: torch.Tensor,
                       sample_weights: torch.Tensor | None = None) -> tuple:
        """Train a VAE using a raw PyTorch loop.

        Replicates the same training logic as the Lightning VAE:
        - AdamW optimizer with weight_decay and eps=1e-4
        - CosineAnnealingWarmRestarts scheduler (T_0=25)
        - mse_loss (L2) + kld_beta * KLD
        - Reparameterization with stdev scaling
        - Mixed-precision (autocast + GradScaler)
        - Early stopping on total_loss (patience = grid_patience)

        Returns (encoder, decoder, epochs_trained).
        """
        device = x_train.device
        encoder = Encoder(**trial_params).to(device)
        decoder = Decoder(**trial_params).to(device)

        params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.AdamW(
            params, lr=trial_params["lr"],
            weight_decay=trial_params.get("weight_decay", 1e-5),
            eps=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=25, T_mult=1, eta_min=1e-9, last_epoch=-1,
        )

        batch_size = trial_params["batch_size"]
        kld_beta = trial_params["kld_beta"]
        kld_warmup_epochs = trial_params.get("kld_warmup_epochs", 30)
        stdev = trial_params["stdev"]
        n = x_train.shape[0]

        amp_dtype = self._amp_device_type
        use_amp = self.precision == "16-mixed" and amp_dtype == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)

        best_loss = float("inf")
        patience_counter = 0
        epochs_trained = 0

        for epoch in range(self.grid_epochs):
            encoder.train()
            decoder.train()

            # Pump-balanced sampling: draw indices proportional to weights
            if sample_weights is not None:
                perm = torch.multinomial(sample_weights, n, replacement=True)
            else:
                perm = torch.randperm(n, device=device)

            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = x_train[idx]
                yb = y_train[idx]

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(amp_dtype, enabled=use_amp):
                    mu, logvar, h = encoder(xb)
                    std = torch.exp(0.5 * logvar)
                    z = mu + (torch.randn_like(std) * stdev) * std
                    recon = decoder(z)

                    recon_loss = F.mse_loss(recon, yb, reduction="mean")
                    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                    # KLD warmup: linearly ramp beta over first N epochs
                    beta = kld_beta * min(1.0, epoch / max(kld_warmup_epochs, 1))
                    loss = recon_loss + beta * kld

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            epochs_trained = epoch + 1

            # Early stopping on mean training loss
            avg_loss = epoch_loss / max(n_batches, 1)
            if not np.isfinite(avg_loss):
                break  # NaN/Inf → stop early
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.grid_patience:
                    break

        return encoder, decoder, epochs_trained

    # ------------------------------------------------------------------
    # Day-level classification metrics (same logic as FailureDetector)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_day_level_f1(errors_normal, ts_normal, pids_normal,
                              errors_abnormal, ts_abnormal, pids_abnormal):
        """Compute day-level F1, AUC-ROC from per-sample reconstruction errors.

        Aggregates samples by (pump_id, date) → mean error per operational day,
        then finds the optimal threshold via Youden's J statistic on the ROC curve.

        Returns dict with f1_score, auc_roc, precision, recall.
        """
        def aggregate_by_day(errors, timestamps, pump_ids):
            df = pd.DataFrame({
                'error': errors,
                'timestamp': timestamps,
                'pump_id': pump_ids,
            })
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            return df.groupby(['pump_id', 'date'])['error'].mean().values

        day_errors_n = aggregate_by_day(errors_normal, ts_normal, pids_normal)
        day_errors_a = aggregate_by_day(errors_abnormal, ts_abnormal, pids_abnormal)

        labels = np.concatenate([np.zeros(len(day_errors_n)), np.ones(len(day_errors_a))])
        scores = np.concatenate([day_errors_n, day_errors_a])

        auc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        optimal_idx = int(np.argmax(tpr - fpr))
        optimal_threshold = float(thresholds[optimal_idx])

        predictions = (scores >= optimal_threshold).astype(int)
        prec = precision_score(labels, predictions, zero_division=0)
        rec = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        return {
            'f1_score': float(f1),
            'auc_roc': float(auc),
            'precision': float(prec),
            'recall': float(rec),
        }

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------
    def _objective(self, trial: optuna.Trial) -> float:
        """Train one model and return the negative F1-score.

        Optuna *minimises* the objective, so we return ``-f1_score`` to
        effectively maximise the day-level F1.

        Uses a raw PyTorch training loop instead of Lightning to avoid
        ~50x framework overhead per trial (1s vs 54s for 50 epochs on
        this model size).
        """
        # Sample from search spaces
        trial_params = self.base_params.copy()

        # Categorical params
        for key, values in self.categorical_space.items():
            trial_params[key] = trial.suggest_categorical(key, values)

        # Continuous params (stdev, kld_beta, lr, weight_decay)
        for key, (low, high, log_scale) in self.continuous_space.items():
            trial_params[key] = trial.suggest_float(key, low, high, log=log_scale)

        # batch_size is now searched as a categorical hyperparameter,
        # so it's co-optimised with lr, weight_decay, etc.

        try:
            t0 = time.time()
            past_history = trial_params["past_history"]

            ds = self.windowed_datasets[past_history]
            x_train_t = ds["x_train"]
            y_train_t = ds["y_train"]
            weights_t = ds.get("sample_weights")
            x_val_n_t = ds["x_val_normal"]
            y_val_n_t = ds["y_val_normal"]
            x_val_a_t = ds["x_val_abnormal"]
            y_val_a_t = ds["y_val_abnormal"]

            # Train with raw PyTorch loop (bypasses Lightning overhead)
            encoder, decoder, epochs_trained = self._train_vae_raw(
                trial_params, x_train_t, y_train_t, sample_weights=weights_t,
            )

            # Evaluate on validation sets
            encoder.eval()
            decoder.eval()
            amp_dtype = self._amp_device_type
            use_amp = self.precision == "16-mixed" and amp_dtype == "cuda"
            with torch.no_grad(), torch.amp.autocast(amp_dtype, enabled=use_amp):
                mu_n, logvar_n, _ = encoder(x_val_n_t)
                pred_normal = decoder(mu_n)  # eval: use mu directly (no noise)
                mu_a, logvar_a, _ = encoder(x_val_a_t)
                pred_abnormal = decoder(mu_a)

                mse_normal = F.mse_loss(pred_normal, y_val_n_t).item()
                mse_abnormal = F.mse_loss(pred_abnormal, y_val_a_t).item()

                # Per-sample errors for day-level classification metrics
                errors_n = ((pred_normal - y_val_n_t) ** 2).mean(dim=1).cpu().numpy()
                errors_a = ((pred_abnormal - y_val_a_t) ** 2).mean(dim=1).cpu().numpy()

            ratio = mse_abnormal / (mse_normal + 1e-8)

            # Day-level F1 / AUC-ROC (same methodology as FailureDetector)
            clf = self._compute_day_level_f1(
                errors_n, ds["ts_val_normal"], ds["pids_val_normal"],
                errors_a, ds["ts_val_abnormal"], ds["pids_val_abnormal"],
            )
            elapsed = time.time() - t0

            # Store metadata for later analysis
            trial.set_user_attr("mse_normal", mse_normal)
            trial.set_user_attr("mse_abnormal", mse_abnormal)
            trial.set_user_attr("ratio", ratio)
            trial.set_user_attr("f1_score", clf["f1_score"])
            trial.set_user_attr("auc_roc", clf["auc_roc"])
            trial.set_user_attr("precision", clf["precision"])
            trial.set_user_attr("recall", clf["recall"])
            trial.set_user_attr("epochs_trained", epochs_trained)

            # Compact per-trial summary
            all_keys = list(self.categorical_space) + list(self.continuous_space)
            param_str = ", ".join(
                f"{k}={trial_params[k]:.6g}" if isinstance(trial_params[k], float)
                else f"{k}={trial_params[k]}"
                for k in all_keys
            )
            print(
                f"  Trial {trial.number + 1:3d}/{self.n_trials} ({elapsed:.1f}s) | "
                f"{param_str} | "
                f"F1={clf['f1_score']:.3f} AUC={clf['auc_roc']:.3f} "
                f"ratio={ratio:.1f}x"
            )

            # We want to MAXIMISE F1, so return its negative
            return -clf["f1_score"]

        except Exception as e:
            print(f"  Trial {trial.number + 1:3d}/{self.n_trials} FAILED: {e}")
            return float("inf")
        finally:
            # Free GPU memory between trials to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Execute the TPE search and return the best parameters dict.

        Returns:
            dict: A copy of the base parameters updated with the best
            hyperparameters found during tuning.

        Side-effects:
            - Saves ``tpe_search_results.csv`` with all trials ranked by F1
            - Saves ``best_params.json`` with the winning configuration
            - Saves ``best_weights.pt`` (encoder + decoder state dicts)
            - Saves ``norm_params.json`` used during tuning
        """
        # Suppress verbose output during search
        warnings.filterwarnings("ignore")
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._precompute_datasets()

        print("\n" + "=" * 70)
        print("STARTING OPTUNA TPE SEARCH")
        print("=" * 70)

        sampler = TPESampler(
            seed=self.seed,
            multivariate=True,
            n_startup_trials=20,
        )
        study = optuna.create_study(
            study_name="vae_tpe_search",
            direction="minimize",  # we return -f1_score
            sampler=sampler,
        )

        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("lightning_logs", "tuning_trials", run_stamp)
        os.makedirs(results_dir, exist_ok=True)

        t_start = time.time()
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
        )
        t_total = time.time() - t_start

        print(f"\n{'=' * 70}")
        print(f"TPE SEARCH COMPLETE — {self.n_trials} trials in {t_total / 60:.1f} min")
        print(f"{'=' * 70}")

        # ---- Build results table ----
        results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value != float("inf"):
                row = {"trial": trial.number}
                row.update(trial.params)
                row["f1_score"] = trial.user_attrs.get("f1_score")
                row["auc_roc"] = trial.user_attrs.get("auc_roc")
                row["precision"] = trial.user_attrs.get("precision")
                row["recall"] = trial.user_attrs.get("recall")
                row["mse_normal"] = trial.user_attrs.get("mse_normal")
                row["mse_abnormal"] = trial.user_attrs.get("mse_abnormal")
                row["ratio"] = trial.user_attrs.get("ratio")
                row["epochs"] = trial.user_attrs.get("epochs_trained")
                results.append(row)

        df_results = pd.DataFrame(results)

        if df_results.empty:
            print("\n⚠ No successful trials. Returning base parameters unchanged.")
            return self.base_params.copy()

        # Rank by F1 (primary), then AUC-ROC (tiebreaker)
        df_results = df_results.sort_values(
            by=["f1_score", "auc_roc", "ratio"],
            ascending=[False, False, False],
        )

        # Top 15
        print(f"\n{'=' * 100}")
        print("TOP 15 CONFIGURATIONS (ranked by F1-score)")
        print(f"{'=' * 100}")
        display_cols = (
            ["trial"] + list(self.categorical_space) + list(self.continuous_space) +
            ["f1_score", "auc_roc", "precision", "recall", "ratio", "epochs"]
        )
        print(df_results[display_cols].head(15).to_string(index=False))

        # Best trial
        best_row = df_results.iloc[0]
        best_trial_num = int(best_row["trial"])
        print(f"\n{'=' * 100}")
        print("BEST TRIAL (highest F1-score)")
        print(f"{'=' * 100}")
        print(f"  Trial number: {best_trial_num}")
        print(f"  F1-score:     {best_row['f1_score']:.4f}")
        print(f"  AUC-ROC:      {best_row['auc_roc']:.4f}")
        print(f"  Precision:    {best_row['precision']:.4f}")
        print(f"  Recall:       {best_row['recall']:.4f}")
        print(f"  Ratio:        {best_row['ratio']:.2f}x")
        print(f"\n  Best parameters:")
        all_keys = list(self.categorical_space) + list(self.continuous_space)
        for key in all_keys:
            print(f"    {key}: {best_row[key]}")

        # Build updated params dict
        best_params = self.base_params.copy()
        for key in all_keys:
            val = best_row[key]
            # Convert numpy types to native Python for JSON serialisation
            if hasattr(val, "item"):
                val = val.item()
            best_params[key] = val

        # Print recommended JSON
        print(f"\n{'=' * 100}")
        print("RECOMMENDED parameters.json UPDATE")
        print(f"{'=' * 100}")
        print(json.dumps(best_params, indent=4))

        # ---- Persist results ----
        if self.save_results:
            os.makedirs(results_dir, exist_ok=True)

            # 1. Full results CSV (all trials, ranked by F1)
            csv_path = os.path.join(results_dir, "tpe_search_results.csv")
            df_results.to_csv(csv_path, index=False)
            print(f"\n✓ Full results saved to {csv_path}")

            # 2. Best parameters JSON
            best_params_path = os.path.join(results_dir, "best_params.json")
            with open(best_params_path, "w") as f:
                json.dump(best_params, f, indent=4)
            print(f"✓ Best parameters saved to {best_params_path}")

            # 3. Normalization params (needed to use the weights later)
            if self.norm_params_cached:
                norm_path = os.path.join(results_dir, "norm_params.json")
                with open(norm_path, "w") as f:
                    json.dump(self.norm_params_cached, f, indent=4)
                print(f"✓ Normalization params saved to {norm_path}")

            # 4. Retrain the best trial and save encoder/decoder weights
            print(f"\n  Retraining best trial (#{best_trial_num}) to save weights...")
            best_trial_obj = study.trials[best_trial_num]
            retrain_params = self.base_params.copy()
            retrain_params.update(best_trial_obj.params)

            ph = retrain_params["past_history"]
            ds = self.windowed_datasets[ph]
            encoder, decoder, epochs = self._train_vae_raw(
                retrain_params, ds["x_train"], ds["y_train"],
                sample_weights=ds.get("sample_weights"),
            )

            weights_path = os.path.join(results_dir, "best_weights.pt")
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "params": best_params,
                "epochs_trained": epochs,
                "f1_score": best_row["f1_score"],
                "auc_roc": best_row["auc_roc"],
            }, weights_path)
            print(f"✓ Best model weights saved to {weights_path}")

        # Clean up norm_params.json written by preprocessor during precompute
        if os.path.exists("norm_params.json"):
            os.remove("norm_params.json")

        return best_params