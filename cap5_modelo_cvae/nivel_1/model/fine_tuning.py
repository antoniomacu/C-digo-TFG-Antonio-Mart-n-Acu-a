import json
import logging
import os
import time
import warnings
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from optuna.exceptions import TrialPruned
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast

from .device import select_accelerator, should_pin_memory
from .models import TemporalCVAE
from .preprocessing import Preprocessor


DEFAULT_CATEGORICAL_SPACE = {
    "past_history": [3, 5],
    "latent_dim": [8, 16, 24],
    "layer_sizes": ["128,64", "256,128,64", "256,128", "512,256,128"],
    "batch_size": [32, 64, 128],
    "loss_fn": ["huber", "mse", "smooth_l1"],
}

DEFAULT_CONTINUOUS_SPACE = {
    "stdev": (0.01, 0.5, True),
    "kld_beta": (0.0001, 0.01, True),
    "lr": (1e-4, 0.01, True),
    "weight_decay": (1e-6, 1e-3, True),
    "dropout": (0.0, 0.3, False),
}


class HyperparameterTuner:
    """Optuna TPE tuner for TemporalCVAE (cond_reg_v2)."""

    PAST_HISTORY_CHOICES = [3, 5]

    @staticmethod
    def _resolve_config_path(path_value: str, base_dir: Path) -> str:
        """Resolve config paths relative to parameters.json directory."""
        path = Path(path_value)
        if path.is_absolute():
            return str(path)
        return str((base_dir / path).resolve())

    def __init__(
        self,
        parameters_file,
        categorical_space=None,
        continuous_space=None,
        n_trials=300,
        data_fraction=1.0,
        grid_epochs=None,
        grid_patience=None,
        seed=42,
        save_results=True,
    ):
        parameters_path = Path(parameters_file).resolve()
        with open(parameters_path, "r", encoding="utf-8") as f:
            self.base_params = json.load(f)

        params_base_dir = parameters_path.parent
        for key in ("train_path", "test_path", "norm_path"):
            if key in self.base_params and isinstance(self.base_params[key], str):
                self.base_params[key] = self._resolve_config_path(
                    self.base_params[key], params_base_dir
                )

        self.categorical_space = categorical_space or DEFAULT_CATEGORICAL_SPACE
        self.continuous_space = continuous_space or DEFAULT_CONTINUOUS_SPACE
        self.n_trials = int(n_trials)
        self.data_fraction = float(data_fraction)
        self.grid_epochs = int(
            grid_epochs if grid_epochs is not None else self.base_params.get("epochs", 150)
        )
        self.grid_patience = int(
            grid_patience if grid_patience is not None else self.base_params.get("patience", 44)
        )
        self.seed = int(seed)
        self.save_results = bool(save_results)
        self.rng = np.random.RandomState(self.seed)

        self.base_hparams = Namespace(**self.base_params)
        self.preprocessor = Preprocessor(
            self.base_hparams,
            seed=self.seed,
            train_split=0.8,
            val_normal_split=0.1,
            abnormal_val_split=0.5,
        )

        self.accelerator = select_accelerator()
        self.pin_memory = should_pin_memory()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Mixed precision is only enabled on CUDA per requirements.
        self.use_amp = self.device.type == "cuda"

        self.windowed_datasets = {}
        self.norm_params_cached = None
        self.output_dir = None

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("medium")

    def _suggest_hparams(self, trial):
        hparams = self.base_params.copy()

        for key, values in self.categorical_space.items():
            hparams[key] = trial.suggest_categorical(key, values)

        for key, (low, high, log_scale) in self.continuous_space.items():
            hparams[key] = trial.suggest_float(key, low, high, log=log_scale)

        return hparams

    def _precompute_datasets(self):
        print("Pre-computing datasets for all past_history values...")
        print("(File-level splits: 80/10/10 normal + 50/50 abnormal val/test)")

        t_start = time.time()

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
        val_abnormal_files, _ = train_test_split(
            all_abnormal_files, test_size=0.5, random_state=self.seed
        )

        print("Loading split files once...")
        df_train_raw = self.preprocessor._load_files(self.base_hparams.train_path, train_files)
        self.norm_params_cached = self.preprocessor.get_normalization_params(
            df_train_raw,
            save_path=str(self.output_dir / "norm_params.json"),
        )

        df_train_norm = self.preprocessor.normalize_data(
            df_train_raw, self.norm_params_cached, self.base_hparams.norm_method
        )
        df_train_norm = self.preprocessor.rebuild_pump_id(df_train_norm)

        df_val_raw = self.preprocessor._load_files(self.base_hparams.train_path, val_files)
        df_val_norm = self.preprocessor.normalize_data(
            df_val_raw, self.norm_params_cached, self.base_hparams.norm_method
        )
        df_val_norm = self.preprocessor.rebuild_pump_id(df_val_norm)

        df_abnormal_raw = self.preprocessor._load_files(
            self.base_hparams.test_path, val_abnormal_files
        )
        df_abnormal_norm = self.preprocessor.normalize_data(
            df_abnormal_raw, self.norm_params_cached, self.base_hparams.norm_method
        )
        df_abnormal_norm = self.preprocessor.rebuild_pump_id(df_abnormal_norm)

        past_history_values = self.categorical_space.get("past_history", self.PAST_HISTORY_CHOICES)

        for ph in past_history_values:
            x_train, y_train, _, pids_train = self.preprocessor.build_preprocessing_window(
                df_train_norm.copy(), ph
            )
            x_val_normal, y_val_normal, ts_val_normal, pids_val_normal = self.preprocessor.build_preprocessing_window(
                df_val_norm.copy(), ph
            )
            x_val_abnormal, y_val_abnormal, ts_val_abnormal, pids_val_abnormal = self.preprocessor.build_preprocessing_window(
                df_abnormal_norm.copy(), ph
            )

            n_total = x_train.shape[0]
            if self.data_fraction < 1.0:
                n_subset = max(1, int(n_total * self.data_fraction))
                subset_idx = self.rng.choice(n_total, size=n_subset, replace=False)
                x_train_use = x_train[subset_idx]
                y_train_use = y_train[subset_idx]
                pids_train_use = np.asarray(pids_train)[subset_idx]
            else:
                n_subset = n_total
                x_train_use = x_train
                y_train_use = y_train
                pids_train_use = np.asarray(pids_train)

            unique_pids, pid_counts = np.unique(pids_train_use, return_counts=True)
            weight_map = {pid: 1.0 / count for pid, count in zip(unique_pids, pid_counts)}
            sample_weights = torch.tensor(
                [weight_map[pid] for pid in pids_train_use],
                dtype=torch.float32,
                device=self.device,
            )

            x_train_t = torch.tensor(x_train_use, dtype=torch.float32, device=self.device).view(n_subset, -1)
            y_train_t = torch.tensor(y_train_use, dtype=torch.float32, device=self.device).view(n_subset, -1)
            x_val_normal_t = torch.tensor(x_val_normal, dtype=torch.float32, device=self.device).view(
                x_val_normal.shape[0], -1
            )
            y_val_normal_t = torch.tensor(y_val_normal, dtype=torch.float32, device=self.device).view(
                y_val_normal.shape[0], -1
            )
            x_val_abnormal_t = torch.tensor(x_val_abnormal, dtype=torch.float32, device=self.device).view(
                x_val_abnormal.shape[0], -1
            )
            y_val_abnormal_t = torch.tensor(y_val_abnormal, dtype=torch.float32, device=self.device).view(
                y_val_abnormal.shape[0], -1
            )

            self.windowed_datasets[ph] = {
                "x_train": x_train_t,
                "y_train": y_train_t,
                "sample_weights": sample_weights,
                "x_val_normal": x_val_normal_t,
                "y_val_normal": y_val_normal_t,
                "x_val_abnormal": x_val_abnormal_t,
                "y_val_abnormal": y_val_abnormal_t,
                "ts_val_normal": ts_val_normal,
                "pids_val_normal": np.asarray(pids_val_normal),
                "ts_val_abnormal": ts_val_abnormal,
                "pids_val_abnormal": np.asarray(pids_val_abnormal),
                "norm_params": self.norm_params_cached,
            }

            print(
                f"  ph={ph}: train={x_train_t.shape[0]} ({self.data_fraction*100:.0f}% of {n_total}), "
                f"val_normal={x_val_normal_t.shape[0]}, "
                f"val_abnormal={x_val_abnormal_t.shape[0]}"
            )

        del df_train_raw, df_train_norm, df_val_raw, df_val_norm, df_abnormal_raw, df_abnormal_norm

        print(
            f"Dataset cache ready in {time.time() - t_start:.1f}s on "
            f"{self.accelerator} (pin_memory={self.pin_memory})"
        )
        print(f"TPE search: {self.n_trials} trials")
        print(f"Training per trial: max {self.grid_epochs} epochs, patience {self.grid_patience}")

    def _loss_reconstruction(self, recon, target, loss_fn):
        loss_name = str(loss_fn).lower()
        if loss_name == "huber":
            return F.smooth_l1_loss(recon, target)
        if loss_name == "mse":
            return F.mse_loss(recon, target)
        if loss_name == "smooth_l1":
            return F.smooth_l1_loss(recon, target)
        raise ValueError(f"Unsupported loss_fn='{loss_fn}'")

    def _train_raw(
        self,
        trial,
        model,
        x_train,
        y_train,
        x_val_normal,
        y_val_normal,
        x_val_abnormal,
        y_val_abnormal,
        hparams,
        max_epochs=80,
        patience=20,
        enable_pruning=True,
        sample_weights=None,
    ):
        all_params = list(model.parameters())

        optimizer = torch.optim.AdamW(
            all_params,
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
            eps=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=25, T_mult=1, eta_min=1e-9, last_epoch=-1
        )

        scaler = GradScaler(enabled=self.use_amp)

        n_train = x_train.shape[0]
        batch_size = int(hparams["batch_size"])

        best_val_ratio = float("-inf")
        best_epoch = -1
        epochs_no_improve = 0

        kld_beta = float(hparams["kld_beta"])
        stdev = float(hparams["stdev"])
        kld_warmup_epochs = int(hparams.get("kld_warmup_epochs", 30))

        for epoch in range(max_epochs):
            model.train()

            if sample_weights is not None:
                permutation = torch.multinomial(sample_weights, n_train, replacement=True)
            else:
                permutation = torch.randperm(n_train, device=self.device)

            for start in range(0, n_train, batch_size):
                idx = permutation[start : start + batch_size]
                x_batch = x_train[idx]
                y_batch = y_train[idx]

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=self.use_amp):
                    recon, mu, logvar, _ = model(x_batch)

                    recon_loss = self._loss_reconstruction(recon, y_batch, hparams["loss_fn"])
                    kld = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
                    warmup_factor = min(1.0, epoch / max(kld_warmup_epochs, 1))
                    loss = recon_loss + kld_beta * warmup_factor * kld

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()

            model.eval()

            with torch.no_grad(), autocast(enabled=self.use_amp):
                recon_normal = model.reconstruct(x_val_normal)
                mse_normal = F.mse_loss(recon_normal, y_val_normal).item()

                recon_abnormal = model.reconstruct(x_val_abnormal)
                mse_abnormal = F.mse_loss(recon_abnormal, y_val_abnormal).item()

            val_ratio = mse_abnormal / (mse_normal + 1e-8)

            if val_ratio > best_val_ratio:
                best_val_ratio = float(val_ratio)
                best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if enable_pruning and trial is not None:
                trial.report(float(val_ratio), epoch)
                if trial.should_prune():
                    raise TrialPruned()

            if epochs_no_improve >= patience:
                break

        return best_val_ratio, best_epoch

    @staticmethod
    def _compute_day_level_metrics(
        errors_normal,
        ts_normal,
        pids_normal,
        errors_abnormal,
        ts_abnormal,
        pids_abnormal,
    ):
        def aggregate_day(errors, timestamps, pump_ids):
            df = pd.DataFrame(
                {
                    "error": errors,
                    "timestamp": timestamps,
                    "pump_id": pump_ids,
                }
            )
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            return df.groupby(["pump_id", "date"])["error"].mean().values

        day_errors_n = aggregate_day(errors_normal, ts_normal, pids_normal)
        day_errors_a = aggregate_day(errors_abnormal, ts_abnormal, pids_abnormal)

        labels = np.concatenate([np.zeros(len(day_errors_n)), np.ones(len(day_errors_a))])
        scores = np.concatenate([day_errors_n, day_errors_a])

        auc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        best_idx = int(np.argmax(tpr - fpr))
        optimal_threshold = float(thresholds[best_idx])

        predictions = (scores >= optimal_threshold).astype(int)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        return {
            "f1_score": float(f1),
            "auc_roc": float(auc),
            "precision": float(precision),
            "recall": float(recall),
            "optimal_threshold": optimal_threshold,
            "n_normal_days": int(len(day_errors_n)),
            "n_abnormal_days": int(len(day_errors_a)),
        }

    def _objective(self, trial):
        try:
            hparams = self._suggest_hparams(trial)
            past_history = int(hparams["past_history"])
            ds = self.windowed_datasets[past_history]

            x_train = ds["x_train"]
            y_train = ds["y_train"]
            x_val_normal = ds["x_val_normal"]
            y_val_normal = ds["y_val_normal"]
            x_val_abnormal = ds["x_val_abnormal"]
            y_val_abnormal = ds["y_val_abnormal"]

            model = TemporalCVAE(**hparams).to(self.device)

            best_val_ratio, best_epoch = self._train_raw(
                trial,
                model,
                x_train,
                y_train,
                x_val_normal,
                y_val_normal,
                x_val_abnormal,
                y_val_abnormal,
                hparams,
                max_epochs=self.grid_epochs,
                patience=self.grid_patience,
                enable_pruning=True,
                sample_weights=ds.get("sample_weights"),
            )

            with torch.no_grad(), autocast(enabled=self.use_amp):
                model.eval()
                pred_normal = model.reconstruct(x_val_normal)
                pred_abnormal = model.reconstruct(x_val_abnormal)

                mse_normal = F.mse_loss(pred_normal, y_val_normal).item()
                mse_abnormal = F.mse_loss(pred_abnormal, y_val_abnormal).item()

                errors_n = ((pred_normal - y_val_normal) ** 2).mean(dim=1).detach().cpu().numpy()
                errors_a = ((pred_abnormal - y_val_abnormal) ** 2).mean(dim=1).detach().cpu().numpy()

            val_ratio = mse_abnormal / (mse_normal + 1e-8)
            clf_metrics = self._compute_day_level_metrics(
                errors_n,
                ds["ts_val_normal"],
                ds["pids_val_normal"],
                errors_a,
                ds["ts_val_abnormal"],
                ds["pids_val_abnormal"],
            )

            trial.set_user_attr("val_ratio", float(best_val_ratio))
            trial.set_user_attr("epochs_trained", int(best_epoch))
            trial.set_user_attr("mse_normal", float(mse_normal))
            trial.set_user_attr("mse_abnormal", float(mse_abnormal))
            trial.set_user_attr("f1_score", float(clf_metrics["f1_score"]))
            trial.set_user_attr("auc_roc", float(clf_metrics["auc_roc"]))
            trial.set_user_attr("precision", float(clf_metrics["precision"]))
            trial.set_user_attr("recall", float(clf_metrics["recall"]))
            trial.set_user_attr("optimal_threshold", float(clf_metrics["optimal_threshold"]))

            print(
                f"Trial {trial.number + 1}/{self.n_trials} | "
                f"params={trial.params} | "
                f"F1={clf_metrics['f1_score']:.4f} "
                f"AUC={clf_metrics['auc_roc']:.4f} "
                f"ratio={val_ratio:.2f}x",
                flush=True,
            )
            return float(-clf_metrics["f1_score"])

        except TrialPruned:
            raise
        except Exception as exc:
            print(f"Trial {trial.number + 1}/{self.n_trials} FAILED: {exc}", flush=True)
            return float("inf")
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _retrain_best(self, best_params, datasets):
        past_history = int(best_params["past_history"])
        ds = datasets[past_history]

        x_train = ds["x_train"]
        y_train = ds["y_train"]
        x_val_normal = ds["x_val_normal"]
        y_val_normal = ds["y_val_normal"]
        x_val_abnormal = ds["x_val_abnormal"]
        y_val_abnormal = ds["y_val_abnormal"]
        norm_params = ds["norm_params"]

        model = TemporalCVAE(**best_params).to(self.device)

        full_epochs = int(best_params.get("epochs", self.base_params.get("epochs", 150)))
        full_patience = int(best_params.get("patience", self.base_params.get("patience", 44)))

        best_val_ratio, epochs_trained = self._train_raw(
            None,
            model,
            x_train,
            y_train,
            x_val_normal,
            y_val_normal,
            x_val_abnormal,
            y_val_abnormal,
            best_params,
            max_epochs=full_epochs,
            patience=full_patience,
            enable_pruning=False,
            sample_weights=ds.get("sample_weights"),
        )

        ckpt = {
            "temporal_embedding_state_dict": model.temporal_embedding.state_dict(),
            "temporal_attention_state_dict": model.temporal_attention.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "decoder_state_dict": model.decoder.state_dict(),
            "params": best_params,
            "epochs_trained": int(epochs_trained),
            "val_loss": float(best_val_ratio),
            "val_ratio": float(best_val_ratio),
        }
        torch.save(ckpt, self.output_dir / "best_weights.pt")

        with open(self.output_dir / "norm_params.json", "w", encoding="utf-8") as f:
            json.dump(norm_params, f, indent=4)

        return best_val_ratio

    def run(self):
        warnings.filterwarnings("ignore")
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        output_root = Path(__file__).resolve().parents[1] / "best_results" / "tuning_trials"
        self.output_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._precompute_datasets()

        sampler = TPESampler(
            seed=self.seed,
            multivariate=True,
            n_startup_trials=min(20, max(5, self.n_trials // 10)),
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)

        print("\nStarting Optuna TPE search...")
        study.optimize(self._objective, n_trials=self.n_trials)

        rows = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE and np.isfinite(t.value):
                row = {"trial": t.number}
                row.update(t.params)
                row["f1_score"] = t.user_attrs.get("f1_score")
                row["auc_roc"] = t.user_attrs.get("auc_roc")
                row["precision"] = t.user_attrs.get("precision")
                row["recall"] = t.user_attrs.get("recall")
                row["val_ratio"] = t.user_attrs.get("val_ratio")
                row["mse_normal"] = t.user_attrs.get("mse_normal")
                row["mse_abnormal"] = t.user_attrs.get("mse_abnormal")
                row["epochs"] = t.user_attrs.get("epochs_trained")
                rows.append(row)

        if not rows:
            print("No completed trials. Returning base parameters.")
            return self.base_params.copy()

        df_results = pd.DataFrame(rows).sort_values(
            ["f1_score", "auc_roc", "val_ratio"],
            ascending=[False, False, False],
        )

        print("\nTop 15 trials (ranked by F1, AUC, ratio):")
        cols = [
            "trial",
            *list(self.categorical_space.keys()),
            *list(self.continuous_space.keys()),
            "f1_score",
            "auc_roc",
            "precision",
            "recall",
            "val_ratio",
            "epochs",
        ]
        print(df_results[cols].head(15).to_string(index=False))

        if self.save_results:
            df_results.to_csv(self.output_dir / "tpe_search_results.csv", index=False)

        best_trial_num = int(df_results.iloc[0]["trial"])
        best_trial = study.trials[best_trial_num]
        best_params = self.base_params.copy()
        best_params.update(best_trial.params)

        with open(self.output_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=4)

        best_retrain_ratio = self._retrain_best(best_params, self.windowed_datasets)
        print(
            f"\nBest trial #{best_trial.number}: "
            f"F1={df_results.iloc[0]['f1_score']:.4f} "
            f"AUC={df_results.iloc[0]['auc_roc']:.4f} "
            f"val_ratio={df_results.iloc[0]['val_ratio']:.6f} | "
            f"retrain_val_ratio={best_retrain_ratio:.6f}"
        )
        print(f"Saved tuning artifacts to: {self.output_dir}")

        # Keep this cleanup to avoid stale local files from previous pipeline behavior.
        local_norm = Path("norm_params.json")
        if local_norm.exists():
            os.remove(local_norm)

        return best_params
