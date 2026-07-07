"""Failure detector training orchestrator for cond_reg_v2 (TemporalCVAE)."""

import argparse
import json
import os
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from .device import select_accelerator, select_precision
from .models import TemporalCVAE
from .preprocessing import Preprocessor


class FailureDetector:
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
        seed=42,
        train_split=0.8,
        val_normal_split=0.1,
        abnormal_val_split=0.5,
    ):
        """Initialize detector, load hyperparameters, and configure preprocessor."""
        parameters_path = Path(parameters_file).resolve()
        with open(parameters_path, "r") as file:
            self.hparams2 = json.load(file)

        # Resolve data paths relative to parameters.json so CLI works from any cwd.
        params_base_dir = parameters_path.parent
        for key in ("train_path", "test_path", "norm_path"):
            if key in self.hparams2 and isinstance(self.hparams2[key], str):
                self.hparams2[key] = self._resolve_config_path(
                    self.hparams2[key], params_base_dir
                )

        # Ensure normalization output directory exists when using relative config path.
        norm_dir = Path(self.hparams2.get("norm_path", "")).parent
        if str(norm_dir):
            norm_dir.mkdir(parents=True, exist_ok=True)

        
            self.hparams = Namespace(**self.hparams2)

        self.preprocessor = Preprocessor(
            self.hparams,
            seed=seed,
            train_split=train_split,
            val_normal_split=val_normal_split,
            abnormal_val_split=abnormal_val_split,
        )
        self.seed = seed

    def train(self):
        """Run full training pipeline with validation export and threshold calibration."""
        (
            x_train_n,
            y_train_n,
            ts_train_n,
            pids_train_n,
            x_val_normal_n,
            y_val_normal_n,
            ts_val_normal_n,
            pids_val_normal_n,
            x_test_normal_n,
            y_test_normal_n,
            ts_test_normal_n,
            pids_test_normal_n,
            x_val_abnormal_n,
            y_val_abnormal_n,
            ts_val_abnormal_n,
            pids_val_abnormal_n,
            x_test_abnormal_n,
            y_test_abnormal_n,
            ts_test_abnormal_n,
            pids_test_abnormal_n,
        ) = self.preprocessor.build_dataset(train=True)

        print("\n" + "=" * 50)
        print("DATASET SHAPES")
        print("=" * 50)
        print(f"Training (normal):        x={x_train_n.shape}, y={y_train_n.shape}")
        print(f"Validation (normal):      x={x_val_normal_n.shape}, y={y_val_normal_n.shape}")
        print(f"Validation (abnormal):    x={x_val_abnormal_n.shape}, y={y_val_abnormal_n.shape}")
        print(f"Test (normal):            x={x_test_normal_n.shape}, y={y_test_normal_n.shape} [HELD OUT]")
        print(f"Test (abnormal):          x={x_test_abnormal_n.shape}, y={y_test_abnormal_n.shape} [HELD OUT]")
        print("=" * 50 + "\n")

        # TemporalCVAE consumes flattened x windows, while y remains current-step [N, n_output].
        x_train = torch.tensor(x_train_n, dtype=torch.float32).view(x_train_n.shape[0], -1)
        y_train = torch.tensor(y_train_n, dtype=torch.float32)

        x_val_normal = torch.tensor(x_val_normal_n, dtype=torch.float32).view(
            x_val_normal_n.shape[0], -1
        )
        y_val_normal = torch.tensor(y_val_normal_n, dtype=torch.float32)

        x_val_abnormal = torch.tensor(x_val_abnormal_n, dtype=torch.float32).view(
            x_val_abnormal_n.shape[0], -1
        )
        y_val_abnormal = torch.tensor(y_val_abnormal_n, dtype=torch.float32)

        # Safety squeeze in case downstream preprocessing returns [N, 1, n_output].
        if y_train.ndim == 3 and y_train.shape[1] == 1:
            y_train = y_train.squeeze(1)
        if y_val_normal.ndim == 3 and y_val_normal.shape[1] == 1:
            y_val_normal = y_val_normal.squeeze(1)
        if y_val_abnormal.ndim == 3 and y_val_abnormal.shape[1] == 1:
            y_val_abnormal = y_val_abnormal.squeeze(1)

        self.ts_val_normal = ts_val_normal_n
        self.pids_val_normal = pids_val_normal_n
        self.ts_val_abnormal = ts_val_abnormal_n
        self.pids_val_abnormal = pids_val_abnormal_n

        # Keep test split strictly held-out for explicit test() evaluation.
        self.x_test_normal = torch.tensor(x_test_normal_n, dtype=torch.float32).view(
            x_test_normal_n.shape[0], -1
        )
        self.y_test_normal = torch.tensor(y_test_normal_n, dtype=torch.float32)
        self.ts_test_normal = ts_test_normal_n
        self.pids_test_normal = pids_test_normal_n

        self.x_test_abnormal = torch.tensor(x_test_abnormal_n, dtype=torch.float32).view(
            x_test_abnormal_n.shape[0], -1
        )
        self.y_test_abnormal = torch.tensor(y_test_abnormal_n, dtype=torch.float32)
        self.ts_test_abnormal = ts_test_abnormal_n
        self.pids_test_abnormal = pids_test_abnormal_n

        if self.y_test_normal.ndim == 3 and self.y_test_normal.shape[1] == 1:
            self.y_test_normal = self.y_test_normal.squeeze(1)
        if self.y_test_abnormal.ndim == 3 and self.y_test_abnormal.shape[1] == 1:
            self.y_test_abnormal = self.y_test_abnormal.squeeze(1)

        self.model = TemporalCVAE(
            x_train,
            y_train,
            x_val_normal,
            y_val_normal,
            x_val_abnormal,
            y_val_abnormal,
            pids_train=pids_train_n,
            **self.hparams2,
        )

        csv_logger = CSVLogger(save_dir="lightning_logs", name="metrics")
        self.version_dir = csv_logger.log_dir

        ckpt_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.version_dir,
            filename="model_weights",
            save_top_k=1,
            monitor="val_ratio",
            mode="max",
        )

        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor="val_ratio",
            patience=self.hparams.patience,
            verbose=True,
            mode="max",
        )

        accelerator = select_accelerator()
        precision = select_precision()
        trainer = pl.Trainer(
            max_epochs=self.hparams.epochs,
            callbacks=[ckpt_callback, early_stop_callback],
            logger=csv_logger,
            accelerator=accelerator,
            precision=precision,
            devices=1,
            deterministic=True,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )

        print("Training model...")
        trainer.fit(self.model)

        best_ckpt_path = ckpt_callback.best_model_path
        print(f"\n✓ Loading best checkpoint: {best_ckpt_path}")
        print(f"  Best val_ratio during training: {ckpt_callback.best_model_score:.3f}")

        self.model = TemporalCVAE.load_from_checkpoint(best_ckpt_path)

        # Restore tensors used by training-time hooks/exports.
        self.model.x_train = x_train
        self.model.y_train = y_train
        self.model.x_val_normal = x_val_normal
        self.model.y_val_normal = y_val_normal
        self.model.x_val_abnormal = x_val_abnormal
        self.model.y_val_abnormal = y_val_abnormal

        new_norm_params_path = os.path.join(self.version_dir, "norm_params.json")
        os.makedirs(self.version_dir, exist_ok=True)
        if getattr(self.preprocessor, "norm_params_cached", None):
            with open(new_norm_params_path, "w") as f:
                json.dump(self.preprocessor.norm_params_cached, f, indent=4)
        elif os.path.exists("norm_params.json"):
            shutil.move("norm_params.json", new_norm_params_path)

        print("\n" + "=" * 50)
        print("EXPORTING VALIDATION PREDICTIONS")
        print("=" * 50)

        print("\n1. Validation (Normal data - should have LOW reconstruction error):")
        mse_val_normal = self.export_predictions_to_csv(
            self.version_dir,
            x_val_normal,
            y_val_normal,
            self.ts_val_normal,
            self.pids_val_normal,
            prefix="val_normal_",
        )

        print("\n2. Validation (Abnormal data - should have HIGH reconstruction error):")
        mse_val_abnormal = self.export_predictions_to_csv(
            self.version_dir,
            x_val_abnormal,
            y_val_abnormal,
            self.ts_val_abnormal,
            self.pids_val_abnormal,
            prefix="val_abnormal_",
        )

        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            x_val_n = x_val_normal.to(device)
            recon_normal = self._reconstruct(x_val_n)
            y_val_n = y_val_normal.to(device)
            mse_normal_norm = F.mse_loss(recon_normal, y_val_n).item()

            x_val_a = x_val_abnormal.to(device)
            recon_abnormal = self._reconstruct(x_val_a)
            y_val_a = y_val_abnormal.to(device)
            mse_abnormal_norm = F.mse_loss(recon_abnormal, y_val_a).item()

            ratio_normalized = mse_abnormal_norm / (mse_normal_norm + 1e-8)

        print("\n" + "=" * 50)
        print("TRAINING PHASE COMPLETE")
        print("=" * 50)
        print("\n--- VALIDATION PERFORMANCE (Normalized - for model comparison) ---")
        print(f"  MSE Normal:   {mse_normal_norm:.6f}")
        print(f"  MSE Abnormal: {mse_abnormal_norm:.6f}")
        print(
            "  Ratio: "
            f"{ratio_normalized:.2f}x  ← Should match best checkpoint ({ckpt_callback.best_model_score:.2f})"
        )

        print("\n--- VALIDATION PERFORMANCE (Denormalized - original units) ---")
        print(f"  MSE Normal:   {mse_val_normal:.6f}")
        print(f"  MSE Abnormal: {mse_val_abnormal:.6f}")
        print(f"  Ratio: {mse_val_abnormal / max(mse_val_normal, 1e-8):.2f}x")
        print("\n  → Higher ratio = better anomaly separation")

        val_clf_metrics = self.compute_classification_metrics(
            x_val_normal,
            y_val_normal,
            self.ts_val_normal,
            self.pids_val_normal,
            x_val_abnormal,
            y_val_abnormal,
            self.ts_val_abnormal,
            self.pids_val_abnormal,
        )
        print("\n--- VALIDATION CLASSIFICATION METRICS (day-level) ---")
        print(f"  Normal days:  {val_clf_metrics['n_normal_days']}")
        print(f"  Abnormal days: {val_clf_metrics['n_abnormal_days']}")
        print(f"  AUC-ROC:    {val_clf_metrics['auc_roc']:.4f}")
        print(f"  Precision:  {val_clf_metrics['precision']:.4f}")
        print(f"  Recall:     {val_clf_metrics['recall']:.4f}")
        print(f"  F1-Score:   {val_clf_metrics['f1_score']:.4f}")

        self.write_summary_json(prefix="val", clf_metrics=val_clf_metrics)

        self.compute_production_thresholds(
            x_train,
            y_train,
            pids_train_n,
            val_clf_metrics["optimal_threshold"],
            ts_train=ts_train_n,
        )

        print("\n📌 To evaluate on TEST set, call fd.test() after training")
        print("=" * 50)

    def export_predictions_to_csv(
        self,
        version_dir,
        x_data,
        y_data,
        timestamps,
        pump_ids,
        prefix="",
    ):
        """Export reconstruction predictions and per-sensor MSE to CSV."""
        output_csv_path = f"{prefix}predictions.csv"
        performance_csv_path = f"{prefix}performance_metrics.csv"

        self.model.eval()

        device = next(self.model.parameters()).device
        with torch.no_grad():
            x_tensor = x_data.clone().detach().float().to(device)
            y_real_tensor = y_data.clone().detach().float()
            predictions = self.model(x_tensor)

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        new_norm_params_path = os.path.join(version_dir, "norm_params.json")
        norm_params = None
        if os.path.exists(new_norm_params_path):
            with open(new_norm_params_path, "r") as file:
                norm_params = json.load(file)
        elif getattr(self.preprocessor, "norm_params_cached", None):
            norm_params = self.preprocessor.norm_params_cached
            os.makedirs(version_dir, exist_ok=True)
            with open(new_norm_params_path, "w") as f:
                json.dump(norm_params, f, indent=4)
        elif os.path.exists(self.hparams.norm_path):
            with open(self.hparams.norm_path, "r") as file:
                norm_params = json.load(file)
            os.makedirs(version_dir, exist_ok=True)
            with open(new_norm_params_path, "w") as f:
                json.dump(norm_params, f, indent=4)
        else:
            raise FileNotFoundError(
                f"Normalization params not found. Expected '{new_norm_params_path}' "
                f"or '{self.hparams.norm_path}'."
            )

        pred_np = predictions.cpu().numpy()
        real_np = y_real_tensor.cpu().numpy()

        df_preds = pd.DataFrame(pred_np, columns=self.hparams.output_variables)
        df_real = pd.DataFrame(real_np, columns=self.hparams.output_variables)

        df_preds = self.preprocessor.denormalize_data(
            df_preds, norm_params, self.hparams.norm_method
        )
        df_real = self.preprocessor.denormalize_data(
            df_real, norm_params, self.hparams.norm_method
        )

        df_preds.insert(0, "timestamp", timestamps)
        df_preds.insert(1, "pump_id", pump_ids)
        df_real.insert(0, "timestamp", timestamps)
        df_real.insert(1, "pump_id", pump_ids)

        output_cols = self.hparams.output_variables
        mse_per_sensor = ((df_real[output_cols] - df_preds[output_cols]) ** 2).mean()

        metrics_df = pd.DataFrame(
            {
                "Variable": self.hparams.output_variables,
                "MSE_Error": mse_per_sensor.values,
                "Real_Mean": df_real[output_cols].mean().values,
            }
        )

        df_preds.to_csv(output_csv_path, index=False)
        metrics_df.to_csv(performance_csv_path, index=False)

        print(f"  {prefix}predictions saved to {output_csv_path}")
        print(f"  {prefix}performance metrics saved to {performance_csv_path}")

        shutil.move(output_csv_path, os.path.join(version_dir, output_csv_path))
        shutil.move(performance_csv_path, os.path.join(version_dir, performance_csv_path))

        return mse_per_sensor.mean()

    def compute_classification_metrics(
        self,
        x_normal,
        y_normal,
        ts_normal,
        pids_normal,
        x_abnormal,
        y_abnormal,
        ts_abnormal,
        pids_abnormal,
    ):
        """Compute day-level classification metrics using reconstruction MSE."""
        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            x_n = x_normal.to(device)
            recon_n = self._reconstruct(x_n)
            errors_n = ((recon_n - y_normal.to(device)) ** 2).mean(dim=1).cpu().numpy()

            x_a = x_abnormal.to(device)
            recon_a = self._reconstruct(x_a)
            errors_a = ((recon_a - y_abnormal.to(device)) ** 2).mean(dim=1).cpu().numpy()

        def aggregate_by_day(errors, timestamps, pump_ids):
            df = pd.DataFrame(
                {
                    "error": errors,
                    "timestamp": timestamps,
                    "pump_id": pump_ids,
                }
            )
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            return df.groupby(["pump_id", "date"])["error"].mean().values

        day_errors_normal = aggregate_by_day(errors_n, ts_normal, pids_normal)
        day_errors_abnormal = aggregate_by_day(errors_a, ts_abnormal, pids_abnormal)

        labels = np.concatenate([np.zeros(len(day_errors_normal)), np.ones(len(day_errors_abnormal))])
        scores = np.concatenate([day_errors_normal, day_errors_abnormal])

        auc = roc_auc_score(labels, scores)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        optimal_idx = int(np.argmax(j_scores))
        optimal_threshold = float(thresholds[optimal_idx])

        predictions = (scores >= optimal_threshold).astype(int)
        prec = precision_score(labels, predictions, zero_division=0)
        rec = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions)

        return {
            "auc_roc": round(float(auc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "optimal_threshold": round(optimal_threshold, 6),
            "n_normal_days": int(len(day_errors_normal)),
            "n_abnormal_days": int(len(day_errors_abnormal)),
        }

    @staticmethod
    def _ema_smooth(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Exponential moving average for threshold calibration."""
        smoothed = np.empty_like(values)
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    def compute_production_thresholds(
        self,
        x_train,
        y_train,
        pids_train,
        youden_threshold: float | None = None,
        ts_train=None,
    ):
        """Compute and save production thresholds for anomaly detection."""
        print("\n" + "=" * 50)
        print("COMPUTING PRODUCTION THRESHOLDS")
        print("=" * 50)

        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            x = x_train.to(device)
            recon = self._reconstruct(x)
            per_sample_mse = ((recon - y_train.to(device)) ** 2).mean(dim=1).cpu().numpy()

        pids_arr = np.asarray(pids_train)

        smoothing_alpha = 0.3
        smoothed_per_sample = np.empty_like(per_sample_mse)
        for pid in np.unique(pids_arr):
            mask = pids_arr == pid
            pump_errors = per_sample_mse[mask]
            smoothed_per_sample[mask] = self._ema_smooth(pump_errors, alpha=smoothing_alpha)

        global_mean = float(np.mean(per_sample_mse))
        global_std = float(np.std(per_sample_mse))
        global_p95 = float(np.percentile(per_sample_mse, 95))
        global_p99 = float(np.percentile(per_sample_mse, 99))

        global_thresholds = {
            "warning": global_p95,
            "alarm": global_p99,
            "mean": global_mean,
            "std": global_std,
            "mean_plus_2sigma": global_mean + 2 * global_std,
            "mean_plus_3sigma": global_mean + 3 * global_std,
            "p95": global_p95,
            "p99": global_p99,
        }

        global_smoothed_p95 = float(np.percentile(smoothed_per_sample, 95))
        global_smoothed_p99 = float(np.percentile(smoothed_per_sample, 99))
        global_thresholds["window_warning_smoothed"] = global_smoothed_p95
        global_thresholds["window_alarm_smoothed"] = global_smoothed_p99
        global_thresholds["smoothing_alpha"] = smoothing_alpha

        print(f"\n  Global thresholds (on {len(per_sample_mse)} training samples):")
        print(f"    Mean error:    {global_mean:.8f}")
        print(f"    Std:           {global_std:.8f}")
        print(f"    P95 (warning): {global_p95:.8f}")
        print(f"    P99 (alarm):   {global_p99:.8f}")
        print(f"    P95 smoothed (window warning): {global_smoothed_p95:.8f}")
        print(f"    P99 smoothed (window alarm):   {global_smoothed_p99:.8f}")
        print(f"    μ+3σ:          {global_mean + 3 * global_std:.8f}")

        day_df = None
        if ts_train is not None:
            ts_arr = np.asarray(ts_train)
            day_df = pd.DataFrame({"mse": per_sample_mse, "pump_id": pids_arr, "ts": ts_arr})
            day_df["date"] = pd.to_datetime(day_df["ts"]).dt.date
            day_means = day_df.groupby(["pump_id", "date"])["mse"].mean().values

            day_p95 = float(np.percentile(day_means, 95))
            day_p99 = float(np.percentile(day_means, 99))
            global_thresholds["day_warning"] = day_p95
            global_thresholds["day_alarm"] = day_p99

            print(f"    P95 day-mean (day warning):    {day_p95:.8f}")
            print(f"    P99 day-mean (day alarm):      {day_p99:.8f}")
            print(f"    (from {len(day_means)} operational pump-days)")

        per_pump = {}
        unique_pumps = sorted(np.unique(pids_arr))
        for pid in unique_pumps:
            mask = pids_arr == pid
            pump_errors = per_sample_mse[mask]

            if len(pump_errors) < 10:
                print(f"    ⚠ Pump {pid}: only {len(pump_errors)} samples — using global thresholds")
                continue

            p_mean = float(np.mean(pump_errors))
            p_std = float(np.std(pump_errors))
            p_p95 = float(np.percentile(pump_errors, 95))
            p_p99 = float(np.percentile(pump_errors, 99))

            pump_smoothed = smoothed_per_sample[mask]
            p_smoothed_p95 = float(np.percentile(pump_smoothed, 95))
            p_smoothed_p99 = float(np.percentile(pump_smoothed, 99))

            per_pump[str(int(pid))] = {
                "warning": p_p95,
                "alarm": p_p99,
                "mean": p_mean,
                "std": p_std,
                "mean_plus_2sigma": p_mean + 2 * p_std,
                "mean_plus_3sigma": p_mean + 3 * p_std,
                "p95": p_p95,
                "p99": p_p99,
                "window_warning_smoothed": p_smoothed_p95,
                "window_alarm_smoothed": p_smoothed_p99,
                "smoothing_alpha": smoothing_alpha,
                "n_samples": int(np.sum(mask)),
            }

            if ts_train is not None and day_df is not None:
                pump_day_df = day_df[day_df["pump_id"] == pid]
                pump_day_means = pump_day_df.groupby("date")["mse"].mean().values
                if len(pump_day_means) >= 3:
                    per_pump[str(int(pid))]["day_warning"] = float(np.percentile(pump_day_means, 95))
                    per_pump[str(int(pid))]["day_alarm"] = float(np.percentile(pump_day_means, 99))

            print(f"\n  Pump {int(pid)} ({np.sum(mask)} samples):")
            print(f"    P95 (warning): {p_p95:.8f}")
            print(f"    P99 (alarm):   {p_p99:.8f}")
            print(f"    P95 smoothed (window warning): {p_smoothed_p95:.8f}")
            print(f"    P99 smoothed (window alarm):   {p_smoothed_p99:.8f}")

        calibration = {}
        if youden_threshold is not None:
            calibration["youden_j_threshold"] = youden_threshold
            calibration["note"] = (
                "Youden's J threshold is computed on labeled validation data. "
                "Compare against P99/μ+3σ: if they are within 2-3× of each other, "
                "the statistical thresholds are well-calibrated."
            )
            print(f"\n  Youden's J (validation, day-level): {youden_threshold:.8f}")

            ratio = global_p99 / (youden_threshold + 1e-12)
            if 0.3 < ratio < 3.0:
                calibration["calibration_status"] = "GOOD"
                print(f"  ✓ Calibration: GOOD (P99/Youden ratio = {ratio:.2f})")
            else:
                calibration["calibration_status"] = "CHECK"
                print(f"  ⚠ Calibration: CHECK (P99/Youden ratio = {ratio:.2f})")

        thresholds = {
            "description": (
                "Production thresholds for pump anomaly detection. "
                "WARNING = likely degraded performance, review recommended. "
                "ALARM = anomalous day, operator alert required."
            ),
            "method": "percentile-based (P95 warning, P99 alarm) on training data errors",
            "global": global_thresholds,
            "per_pump": per_pump,
            "calibration": calibration,
        }

        out_path = os.path.join(self.version_dir, "production_thresholds.json")
        os.makedirs(self.version_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(thresholds, f, indent=4)

        print(f"\n  ✓ Production thresholds saved to {out_path}")
        print("=" * 50)

        return thresholds

    def write_summary_json(self, prefix="val", clf_metrics=None):
        """Create summary JSON from exported metrics CSV files."""
        version_path = Path(self.version_dir)
        normal_path = version_path / f"{prefix}_normal_performance_metrics.csv"
        abnormal_path = version_path / f"{prefix}_abnormal_performance_metrics.csv"
        metrics_path = version_path / "metrics.csv"

        if not normal_path.exists() or not abnormal_path.exists():
            raise FileNotFoundError(
                f"Missing performance metrics CSVs in {self.version_dir}. "
                f"Expected {prefix}_normal_performance_metrics.csv and "
                f"{prefix}_abnormal_performance_metrics.csv"
            )

        def overall_mse(perf_path: Path) -> float:
            df = pd.read_csv(perf_path)
            if "MSE_Error" not in df.columns:
                raise ValueError(f"{perf_path.name} is missing 'MSE_Error' column")
            return float(pd.to_numeric(df["MSE_Error"], errors="coerce").dropna().mean())

        mse_normal_denorm = overall_mse(normal_path)
        mse_abnormal_denorm = overall_mse(abnormal_path)
        ratio_denorm = mse_abnormal_denorm / (mse_normal_denorm + 1e-12)

        summary = {
            "model": "temporal_cvae",
            "denormalized": {
                "mse_normal": round(mse_normal_denorm, 6),
                "mse_abnormal": round(mse_abnormal_denorm, 6),
                "ratio": round(ratio_denorm, 4),
            },
        }

        if prefix == "val" and metrics_path.exists():
            mdf = pd.read_csv(metrics_path)
            for col in ("val_mse_normal", "val_mse_abnormal", "val_ratio"):
                if col in mdf.columns:
                    mdf[col] = pd.to_numeric(mdf[col], errors="coerce")

            if "val_ratio" in mdf.columns and mdf["val_ratio"].notna().any():
                best_idx = int(mdf["val_ratio"].idxmax())
                best_ratio = float(mdf.loc[best_idx, "val_ratio"])
                best_mse_n = (
                    float(mdf.loc[best_idx, "val_mse_normal"])
                    if "val_mse_normal" in mdf.columns and pd.notna(mdf.loc[best_idx, "val_mse_normal"])
                    else None
                )
                best_mse_a = (
                    float(mdf.loc[best_idx, "val_mse_abnormal"])
                    if "val_mse_abnormal" in mdf.columns and pd.notna(mdf.loc[best_idx, "val_mse_abnormal"])
                    else None
                )
                normalized = {"ratio": round(best_ratio, 4)}
                if best_mse_n is not None and best_mse_a is not None:
                    normalized.update(
                        {
                            "mse_normal": round(best_mse_n, 6),
                            "mse_abnormal": round(best_mse_a, 6),
                        }
                    )
                summary["normalized"] = normalized

        if clf_metrics:
            summary["classification"] = clf_metrics

        out_path = version_path / f"{prefix}_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"  ✓ {prefix.capitalize()} summary saved to {out_path}")
        return str(out_path)

    def _reconstruct(self, x):
        """Run model and return reconstruction tensor only."""
        output = self.model(x)
        if isinstance(output, tuple):
            return output[0]
        return output

    def predict(self, data):
        return self.model.forward(data)

    def _load_for_testing(self, version_dir):
        """Load trained checkpoint and reconstruct deterministic held-out test sets."""
        version_path = Path(version_dir)
        ckpt_path = version_path / "model_weights.ckpt"
        tuned_weights_path = version_path / "best_weights.pt"
        norm_path = version_path / "norm_params.json"

        if not ckpt_path.exists() and not tuned_weights_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found. Expected one of: {ckpt_path} or {tuned_weights_path}"
            )
        if not norm_path.exists():
            raise FileNotFoundError(f"Norm params not found at {norm_path}")

        if ckpt_path.exists():
            print(f"✓ Loading TemporalCVAE from {ckpt_path}")
            self.model = TemporalCVAE.load_from_checkpoint(str(ckpt_path))

            loaded_hparams = dict(self.model.hparams)
            if loaded_hparams:
                self.hparams2 = loaded_hparams
                self.hparams = Namespace(**self.hparams2)
        else:
            print(f"✓ Loading tuned TemporalCVAE weights from {tuned_weights_path}")
            tuned_ckpt = torch.load(tuned_weights_path, map_location="cpu")

            required_keys = {
                "temporal_embedding_state_dict",
                "temporal_attention_state_dict",
                "encoder_state_dict",
                "decoder_state_dict",
                "params",
            }
            missing = [k for k in required_keys if k not in tuned_ckpt]
            if missing:
                raise KeyError(
                    f"best_weights.pt missing required keys: {missing} (path={tuned_weights_path})"
                )

            self.hparams2 = dict(tuned_ckpt["params"])
            self.hparams = Namespace(**self.hparams2)
            self.model = TemporalCVAE(**self.hparams2)
            self.model.temporal_embedding.load_state_dict(tuned_ckpt["temporal_embedding_state_dict"])
            self.model.temporal_attention.load_state_dict(tuned_ckpt["temporal_attention_state_dict"])
            self.model.encoder.load_state_dict(tuned_ckpt["encoder_state_dict"])
            self.model.decoder.load_state_dict(tuned_ckpt["decoder_state_dict"])

        self.version_dir = str(version_dir)

        with open(norm_path, "r") as f:
            norm_params = json.load(f)

        all_train_files = self.preprocessor._get_file_list(self.hparams.train_path, is_training=True)
        _, temp_files = train_test_split(
            all_train_files,
            test_size=(1 - self.preprocessor.train_split),
            random_state=self.seed,
        )
        remaining = max(1e-12, 1 - self.preprocessor.train_split)
        val_frac = self.preprocessor.val_normal_split / remaining
        _, test_files = train_test_split(
            temp_files,
            test_size=(1 - val_frac),
            random_state=self.seed,
        )

        all_abnormal_files = self.preprocessor._get_file_list(self.hparams.test_path, is_training=False)
        _, test_abn_files = train_test_split(
            all_abnormal_files,
            test_size=(1 - self.preprocessor.abnormal_val_split),
            random_state=self.seed,
        )

        print(f"\n  Test normal files:   {len(test_files)}")
        print(f"  Test abnormal files: {len(test_abn_files)}")

        df_test_n = self.preprocessor._load_files(self.hparams.train_path, test_files)
        test_n_normalized = self.preprocessor.normalize_data(
            df_test_n, norm_params, self.hparams.norm_method
        )
        test_n_normalized = self.preprocessor.rebuild_pump_id(test_n_normalized)
        (
            x_test_normal_n,
            y_test_normal_n,
            ts_test_normal,
            pids_test_normal,
        ) = self.preprocessor.build_preprocessing_window(
            test_n_normalized, self.hparams.past_history
        )

        df_test_abn = self.preprocessor._load_files(self.hparams.test_path, test_abn_files)
        test_abn_normalized = self.preprocessor.normalize_data(
            df_test_abn, norm_params, self.hparams.norm_method
        )
        test_abn_normalized = self.preprocessor.rebuild_pump_id(test_abn_normalized)
        (
            x_test_abnormal_n,
            y_test_abnormal_n,
            ts_test_abnormal,
            pids_test_abnormal,
        ) = self.preprocessor.build_preprocessing_window(
            test_abn_normalized, self.hparams.past_history
        )

        self.x_test_normal = torch.tensor(x_test_normal_n, dtype=torch.float32).view(
            x_test_normal_n.shape[0], -1
        )
        self.y_test_normal = torch.tensor(y_test_normal_n, dtype=torch.float32)
        self.ts_test_normal = np.array(ts_test_normal)
        self.pids_test_normal = np.array(pids_test_normal)

        self.x_test_abnormal = torch.tensor(x_test_abnormal_n, dtype=torch.float32).view(
            x_test_abnormal_n.shape[0], -1
        )
        self.y_test_abnormal = torch.tensor(y_test_abnormal_n, dtype=torch.float32)
        self.ts_test_abnormal = np.array(ts_test_abnormal)
        self.pids_test_abnormal = np.array(pids_test_abnormal)

        if self.y_test_normal.ndim == 3 and self.y_test_normal.shape[1] == 1:
            self.y_test_normal = self.y_test_normal.squeeze(1)
        if self.y_test_abnormal.ndim == 3 and self.y_test_abnormal.shape[1] == 1:
            self.y_test_abnormal = self.y_test_abnormal.squeeze(1)

        print(f"  Test normal samples:   {self.x_test_normal.shape[0]}")
        print(f"  Test abnormal samples: {self.x_test_abnormal.shape[0]}")

    def test(self, version_dir=None):
        """Evaluate the best model on held-out test sets and export results."""
        if version_dir is not None:
            self._load_for_testing(version_dir)
        elif not hasattr(self, "x_test_normal") or not hasattr(self, "version_dir"):
            raise Exception(
                "No test data available. Either call train() first or "
                "provide version_dir for standalone testing."
            )

        print("\n" + "=" * 50)
        print("TESTING PHASE: EVALUATING ON HELD-OUT TEST SETS")
        print("=" * 50)

        print("\nTest set sizes:")
        print(f"  Test (normal):   {self.x_test_normal.shape[0]} samples")
        print(f"  Test (abnormal): {self.x_test_abnormal.shape[0]} samples")

        print("\n1. Test (Normal data - should have LOW reconstruction error):")
        mse_test_normal = self.export_predictions_to_csv(
            self.version_dir,
            self.x_test_normal,
            self.y_test_normal,
            self.ts_test_normal,
            self.pids_test_normal,
            prefix="test_normal_",
        )

        print("\n2. Test (Abnormal data - should have HIGH reconstruction error):")
        mse_test_abnormal = self.export_predictions_to_csv(
            self.version_dir,
            self.x_test_abnormal,
            self.y_test_abnormal,
            self.ts_test_abnormal,
            self.pids_test_abnormal,
            prefix="test_abnormal_",
        )

        test_clf_metrics = self.compute_classification_metrics(
            self.x_test_normal,
            self.y_test_normal,
            self.ts_test_normal,
            self.pids_test_normal,
            self.x_test_abnormal,
            self.y_test_abnormal,
            self.ts_test_abnormal,
            self.pids_test_abnormal,
        )

        print("\n" + "=" * 50)
        print("FINAL TEST RESULTS (UNBIASED)")
        print("=" * 50)
        print("\n  Test samples evaluated:")
        print(f"    Normal:   {self.x_test_normal.shape[0]}")
        print(f"    Abnormal: {self.x_test_abnormal.shape[0]}")
        print(f"\n  MSE on Normal data:   {mse_test_normal:.6f}")
        print(f"  MSE on Abnormal data: {mse_test_abnormal:.6f}")
        print(f"  Ratio (Abnormal/Normal): {mse_test_abnormal / (mse_test_normal + 1e-12):.2f}x")
        print("\n  --- Classification Metrics (day-level) ---")
        print(f"    Normal days:  {test_clf_metrics['n_normal_days']}")
        print(f"    Abnormal days: {test_clf_metrics['n_abnormal_days']}")
        print(f"    AUC-ROC:    {test_clf_metrics['auc_roc']:.4f}")
        print(f"    Precision:  {test_clf_metrics['precision']:.4f}")
        print(f"    Recall:     {test_clf_metrics['recall']:.4f}")
        print(f"    F1-Score:   {test_clf_metrics['f1_score']:.4f}")
        print("\n  ✓ Higher ratio = better anomaly detection capability")

        self.write_summary_json(prefix="test", clf_metrics=test_clf_metrics)

        print("=" * 50)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TemporalCVAE failure detector trainer/tester")
    parser.add_argument("--parameters", required=True, help="Path to parameters JSON")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--version_dir", default=None, help="Version dir for standalone test mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_normal_split", type=float, default=0.1)
    parser.add_argument("--abnormal_val_split", type=float, default=0.5)
    return parser


def main():
    args = _build_arg_parser().parse_args()
    fd = FailureDetector(
        args.parameters,
        seed=args.seed,
        train_split=args.train_split,
        val_normal_split=args.val_normal_split,
        abnormal_val_split=args.abnormal_val_split,
    )

    if args.mode == "train":
        fd.train()
    else:
        fd.test(version_dir=args.version_dir)


if __name__ == "__main__":
    main()