# Production Inference — Anomaly Detection for Pump Operational Days
"""
Production inference script for pump anomaly detection.

Usage:
    pumps-inference path/to/day1.csv [path/to/day2.csv ...] [--version VERSION_DIR]

Given 1–4 CSV files (one per pump running that day), the script:
    1. Loads the trained model + normalization params + production thresholds
    2. Preprocesses each CSV identically to training (Savitzky–Golay, normalize, window)
    3. Reconstructs each window with the VAE
    4. Computes per-window MSE + EMA-smoothed per-window classification
    5. Aggregates to day-level error AND provides per-window timeline
    6. Compares against production thresholds → classifies as NORMAL / WARNING / ALARM
    7. Outputs a JSON report including wall-clock timing
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import PARAMETERS_DIR
from .preprocessing import Preprocessor
from .device import select_accelerator


logger = logging.getLogger(__name__)


# ============================================================================
# PRODUCTION DETECTOR
# ============================================================================

class ProductionDetector:
    """Load a trained model and classify new pump-day CSVs."""

    def __init__(self, version_dir: str, parameters_file: str | None = None):
        """
        Args:
            version_dir: Path to the trained model version directory.
                         Must contain: model_weights.ckpt, norm_params.json,
                         production_thresholds.json
            parameters_file: Optional path to parameters JSON.
                             If None, uses hparams.yaml from version_dir when
                             available, else falls back to package defaults.
        """
        self.version_dir = Path(version_dir)
        self.parameters_file = parameters_file

        # Load parameters
        self.hparams = self._load_hparams()

        # Load normalization parameters
        norm_path = self.version_dir / 'norm_params.json'
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalization params not found at {norm_path}")
        with open(norm_path, 'r') as f:
            self.norm_params = json.load(f)

        # Load production thresholds
        thresh_path = self.version_dir / 'production_thresholds.json'
        if not thresh_path.exists():
            raise FileNotFoundError(
                f"Production thresholds not found at {thresh_path}.\n"
                f"Re-run training with the latest code to generate them,\n"
                f"or call FailureDetector.compute_production_thresholds() on an "
                f"existing version directory."
            )
        with open(thresh_path, 'r') as f:
            self.thresholds = json.load(f)

        # Load model
        self._load_model()

        # Preprocessor (only used for normalize / filter / window utilities)
        from argparse import Namespace
        self.preprocessor = Preprocessor(Namespace(**self.hparams))

    def _load_hparams(self) -> dict:
        """Load hyperparameters for inference.

        Priority:
            1) Explicit --parameters JSON file
            2) hparams.yaml in version directory
            3) package parameters/parameters.json
        """
        if self.parameters_file:
            with open(self.parameters_file, 'r') as f:
                hparams = json.load(f)
        else:
            version_hparams = self.version_dir / 'hparams.yaml'
            if version_hparams.exists():
                try:
                    import yaml
                except ImportError as e:
                    raise ImportError(
                        "hparams.yaml found but PyYAML is not installed. "
                        "Install pyyaml or pass --parameters path/to/parameters.json"
                    ) from e
                with open(version_hparams, 'r') as f:
                    hparams = yaml.safe_load(f) or {}
            else:
                default_params = PARAMETERS_DIR / 'parameters.json'
                with open(default_params, 'r') as f:
                    hparams = json.load(f)

        # Backwards-compatible column upgrades
        from .failure_detector import FailureDetector
        return FailureDetector._upgrade_parameters(hparams)

    def _load_model(self):
        """Load the trained model from checkpoint."""
        from .failure_detector import _get_model_registry

        model_name = self.hparams.get('model', 'vae').lower()
        registry = _get_model_registry()
        if model_name not in registry:
            raise ValueError(f"Unknown model: '{model_name}'. Available: {list(registry.keys())}")

        model_class = registry[model_name]
        ckpt_path = self.version_dir / 'model_weights.ckpt'
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        self.model = model_class.load_from_checkpoint(str(ckpt_path))
        self.model.eval()

        # Select device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model = self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")

    @staticmethod
    def _ema_smooth(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Exponential moving average. alpha=0.3 → ~3-window half-life (~15 min)."""
        smoothed = np.empty_like(values)
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    def _preprocess_csv(self, csv_path: str) -> dict:
        """Preprocess a single production CSV file.

        Returns dict with keys:
            pump_id, date, x_tensor, y_tensor, n_samples
        """
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Extract pump_id and date before processing
        if 'pump_id' not in df.columns:
            raise ValueError(f"CSV {csv_path} missing 'pump_id' column")

        pump_id = int(df['pump_id'].iloc[0])
        date = df.index[0].date()

        # One-hot encode pump_id
        df = self.preprocessor.create_dummies(df)

        # Clean and smooth
        df_filled = df.ffill(limit=1).dropna()
        if len(df_filled) < 5:
            raise ValueError(
                f"Insufficient data in {csv_path} after cleaning "
                f"({len(df_filled)} rows, need ≥5)"
            )

        df_filtered = self.preprocessor.filter_savitzky_golay(df_filled)

        # Normalize using saved training params
        df_normalized = self.preprocessor.normalize_data(
            df_filtered, self.norm_params, self.hparams['norm_method']
        )

        # Rebuild pump_id for windowing
        df_normalized = self.preprocessor.rebuild_pump_id(df_normalized)

        # Build windows
        input_vars = self.hparams['input_variables']
        output_vars = self.hparams['output_variables']
        past_history = self.hparams['past_history']

        x_windows, y_windows, timestamps = [], [], []
        df_work = df_normalized.copy()
        df_work = df_work.drop(columns=['pump_id'])

        if len(df_work) <= past_history:
            raise ValueError(
                f"Not enough rows ({len(df_work)}) for past_history={past_history} "
                f"in {csv_path}"
            )

        for start in range(past_history, len(df_work)):
            input_window = df_work[input_vars].iloc[start - past_history + 1:start + 1].values
            output_window = df_work[output_vars].iloc[start - past_history + 1:start + 1].values
            x_windows.append(input_window)
            y_windows.append(output_window)
            timestamps.append(df_work.index[start].isoformat())

        x_np = np.array(x_windows)
        y_np = np.array(y_windows)

        x_tensor = torch.tensor(x_np, dtype=torch.float32).view(x_np.shape[0], -1)
        y_tensor = torch.tensor(y_np, dtype=torch.float32).view(y_np.shape[0], -1)

        return {
            'pump_id': pump_id,
            'date': str(date),
            'x_tensor': x_tensor,
            'y_tensor': y_tensor,
            'n_samples': x_tensor.shape[0],
            'filename': os.path.basename(csv_path),
            'timestamps': timestamps,
        }

    def classify(self, csv_paths: list[str]) -> dict:
        """Classify one or more pump-day CSV files.

        Args:
            csv_paths: List of paths to CSV files (1–4, one per pump).

        Returns:
            Dict with per-pump results and timing.
        """
        t_start = time.perf_counter()

        # ── Preprocessing ───────────────────────────────────────────────
        t_preprocess_start = time.perf_counter()
        preprocessed = []
        for path in csv_paths:
            try:
                data = self._preprocess_csv(path)
                preprocessed.append(data)
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"Skipping {path}: {e}")
        t_preprocess = time.perf_counter() - t_preprocess_start

        if not preprocessed:
            return {
                'status': 'ERROR',
                'message': 'No valid CSV files could be processed',
                'timing': {'total_seconds': time.perf_counter() - t_start},
            }

        # ── Inference ───────────────────────────────────────────────────
        t_inference_start = time.perf_counter()
        pump_results = []

        for data in preprocessed:
            x = data['x_tensor'].to(self.device)
            y = data['y_tensor'].to(self.device)

            with torch.no_grad():
                output = self.model(x)
                recon = output[0] if isinstance(output, tuple) else output

                # Per-sample MSE
                per_sample_mse = ((recon - y) ** 2).mean(dim=1).cpu().numpy()

            pump_id = data['pump_id']
            pump_key = str(pump_id)

            # Try per-pump threshold first, fall back to global
            if pump_key in self.thresholds.get('per_pump', {}):
                pump_thresh = self.thresholds['per_pump'][pump_key]
            else:
                pump_thresh = self.thresholds['global']

            alarm_threshold = pump_thresh['alarm']
            warning_threshold = pump_thresh['warning']

            smoothing_alpha = pump_thresh.get('smoothing_alpha', 0.3)
            smoothed_mse = self._ema_smooth(per_sample_mse, alpha=smoothing_alpha)
            window_warning = pump_thresh.get('window_warning_smoothed', warning_threshold)
            window_alarm = pump_thresh.get('window_alarm_smoothed', alarm_threshold)

            window_results = []
            for i in range(len(per_sample_mse)):
                sm = float(smoothed_mse[i])
                if sm >= window_alarm:
                    w_status = 'ALARM'
                elif sm >= window_warning:
                    w_status = 'WARNING'
                else:
                    w_status = 'NORMAL'
                window_results.append({
                    'window_index': i,
                    'timestamp': data['timestamps'][i],
                    'raw_mse': round(float(per_sample_mse[i]), 8),
                    'smoothed_mse': round(sm, 8),
                    'status': w_status,
                })

            n_windows = len(window_results)
            fraction_warning = sum(
                1 for w in window_results if w['status'] in ('WARNING', 'ALARM')
            ) / n_windows
            fraction_alarm = sum(1 for w in window_results if w['status'] == 'ALARM') / n_windows

            # Day-level aggregation: mean of all sample errors
            day_error = float(np.mean(per_sample_mse))

            # Classification

            if day_error >= alarm_threshold:
                status = 'ALARM'
            elif day_error >= warning_threshold:
                status = 'WARNING'
            else:
                status = 'NORMAL'

            pump_results.append({
                'pump_id': pump_id,
                'date': data['date'],
                'filename': data['filename'],
                'n_samples': data['n_samples'],
                'day_error_mse': round(day_error, 8),
                'warning_threshold': round(warning_threshold, 8),
                'alarm_threshold': round(alarm_threshold, 8),
                'status': status,
                'window_results': window_results,
                'smoothing_alpha': smoothing_alpha,
                'window_warning_threshold': round(float(window_warning), 8),
                'window_alarm_threshold': round(float(window_alarm), 8),
                'fraction_windows_warning': round(fraction_warning, 4),
                'fraction_windows_alarm': round(fraction_alarm, 4),
            })

        t_inference = time.perf_counter() - t_inference_start

        t_total = time.perf_counter() - t_start

        result = {
            'pump_results': pump_results,
            'model_version': str(self.version_dir),
            'model_architecture': self.hparams.get('model', 'vae'),
            'timing': {
                'preprocessing_seconds': round(t_preprocess, 4),
                'inference_seconds': round(t_inference, 4),
                'total_seconds': round(t_total, 4),
            },
        }

        return result


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pump anomaly detection — production inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pumps-inference pump_1_2025-10-06.csv pump_3_2025-10-06.csv
  pumps-inference *.csv --version lightning_logs/metrics/version_3
        """
    )
    parser.add_argument(
        'csv_files', nargs='+',
        help='One or more CSV files (one per pump-day)',
    )
    parser.add_argument(
        '--version', '-v',
        default=None,
        help='Path to the version directory (default: latest in lightning_logs/metrics/)',
    )
    parser.add_argument(
        '--parameters', '-p',
        default=None,
        help='Path to parameters.json (default: version hparams.yaml, then package default)',
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Path to save JSON report (default: print to stdout)',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print full JSON report including threshold details',
    )
    parser.add_argument(
        '--windows', action='store_true',
        help='Display per-window timeline for each pump',
    )

    args = parser.parse_args()

    # Auto-detect latest version if not specified
    version_dir = args.version
    if version_dir is None:
        metrics_dir = Path('lightning_logs/metrics')
        if metrics_dir.exists():
            versions = sorted(
                [d for d in metrics_dir.iterdir() if d.is_dir()],
                key=lambda d: d.name,
            )
            if versions:
                version_dir = str(versions[-1])

    if version_dir is None:
        print("ERROR: No version directory found. Use --version to specify one.")
        sys.exit(1)

    print(f"✓ Using model version: {version_dir}")
    print(f"✓ Processing {len(args.csv_files)} file(s)")

    # Run inference
    detector = ProductionDetector(version_dir, parameters_file=args.parameters)
    result = detector.classify(args.csv_files)

    # Save / print full JSON
    if args.output:
        report = json.dumps(result, indent=4, default=str)
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\n✓ Report saved to {args.output}")
    elif args.verbose:
        report = json.dumps(result, indent=4, default=str)
        print("\n" + "=" * 60)
        print("PRODUCTION INFERENCE REPORT")
        print("=" * 60)
        print(report)

    # Human-readable summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for pr in result.get('pump_results', []):
        icon = {'NORMAL': '🟢', 'WARNING': '🟡', 'ALARM': '🔴'}.get(pr['status'], '❓')
        print(f"  {icon} Pump {pr['pump_id']} ({pr['date']}): {pr['status']}"
              f"  — error={pr['day_error_mse']:.6f}"
              f"  (warn={pr['warning_threshold']:.6f}, alarm={pr['alarm_threshold']:.6f})")

    if args.windows:
        for pr in result.get('pump_results', []):
            windows = pr.get('window_results', [])
            if not windows:
                continue
            print(f"\n  PUMP {pr['pump_id']} — Per-Window Timeline ({pr['date']})")
            print(f"  {'Time':<22} {'Raw MSE':>12} {'Smoothed':>12} {'Status':<8}")
            print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*8}")
            for w in windows:
                ts_short = w['timestamp']
                icon = {'NORMAL': '🟢', 'WARNING': '🟡', 'ALARM': '🔴'}.get(w['status'], '  ')
                print(
                    f"  {ts_short:<22} {w['raw_mse']:>12.8f} "
                    f"{w['smoothed_mse']:>12.8f} {icon} {w['status']}"
                )

    timing = result.get('timing', {})
    print(f"\n  ⏱  Preprocessing: {timing.get('preprocessing_seconds', 0):.3f}s")
    print(f"  ⏱  Inference:     {timing.get('inference_seconds', 0):.3f}s")
    print(f"  ⏱  Total:         {timing.get('total_seconds', 0):.3f}s")

    print("=" * 60)


if __name__ == '__main__':
    main()
