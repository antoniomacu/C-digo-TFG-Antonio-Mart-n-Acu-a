#!/usr/bin/env python3
"""Recompute production thresholds from an existing trained checkpoint.

This script does NOT retrain. It:
1) Loads the same training split via Preprocessor.build_dataset(train=True)
2) Loads model weights from an existing checkpoint
3) Recomputes production thresholds (including smoothed + day-level)
4) Saves to <version_dir>/production_thresholds.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from model.failure_detector import FailureDetector


def _resolve_version_dir(script_dir: Path, version_dir_arg: str) -> Path:
    candidate = Path(version_dir_arg).expanduser()
    if not candidate.is_absolute():
        candidate = (script_dir / candidate).resolve()
    return candidate


def _resolve_parameters_file(script_dir: Path, version_dir: Path, parameters_arg: str) -> Path:
    candidate = Path(parameters_arg).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate

    relative_candidate = (script_dir / candidate).resolve()
    if relative_candidate.exists():
        return relative_candidate

    # Prefer version-specific hparams so window size and architecture always
    # match the trained checkpoint used for recalibration.
    version_hparams = version_dir / "hparams.yaml"
    if parameters_arg == "parameters.json" and version_hparams.exists():
        print(f"[info] Using version hparams: {version_hparams}")
        return version_hparams

    # Backward-compatible fallback for this repository layout
    fallback = (script_dir / "model" / "parameters" / "parameters.json").resolve()
    if parameters_arg == "parameters.json" and fallback.exists():
        print(f"[info] Using fallback parameters file: {fallback}")
        return fallback

    raise FileNotFoundError(
        "Parameters file not found. "
        f"Tried '{candidate}' and '{relative_candidate}'."
    )


def _select_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_previous_youden_threshold(version_dir: Path) -> float | None:
    thresholds_path = version_dir / "production_thresholds.json"
    if not thresholds_path.exists():
        return None

    try:
        with open(thresholds_path, "r") as f:
            data = json.load(f)
        return data.get("calibration", {}).get("youden_j_threshold")
    except (json.JSONDecodeError, OSError):
        return None


def _materialize_json_parameters(parameters_file: Path) -> Path:
    """Return a JSON file path compatible with FailureDetector constructor."""
    if parameters_file.suffix.lower() == ".json":
        return parameters_file

    if parameters_file.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            f"Unsupported parameters format: {parameters_file}. Use JSON or YAML."
        )

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required when using hparams.yaml. Install it or pass --parameters to a JSON file."
        ) from exc

    with open(parameters_file, "r") as f:
        data = yaml.safe_load(f) or {}

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="recalibrate_params_",
        delete=False,
    )
    with tmp:
        json.dump(data, tmp, indent=4)
    return Path(tmp.name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute production thresholds from existing trained weights."
    )
    parser.add_argument(
        "--version-dir",
        default="final_metrics/",
        help="Directory containing model_weights.ckpt and output production_thresholds.json",
    )
    parser.add_argument(
        "--parameters",
        default="parameters.json",
        help="Path to parameters JSON (default: parameters.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to reproduce the same deterministic file-level data split",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    version_dir = _resolve_version_dir(script_dir, args.version_dir)
    parameters_file = _resolve_parameters_file(script_dir, version_dir, args.parameters)
    json_parameters_file = _materialize_json_parameters(parameters_file)

    ckpt_path = version_dir / "model_weights.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print("=" * 60)
    print("RECALIBRATING PRODUCTION THRESHOLDS (NO TRAINING)")
    print("=" * 60)
    print(f"Version dir:      {version_dir}")
    print(f"Parameters file:  {parameters_file}")
    print(f"Checkpoint:       {ckpt_path}")

    # Reuse exactly the same detector + preprocessing pipeline as training
    fd = FailureDetector(
        str(json_parameters_file),
        seed=args.seed,
        train_split=0.8,
        val_normal_split=0.1,
        abnormal_val_split=0.5,
    )

    (
        x_train_n,
        y_train_n,
        ts_train_n,
        pids_train_n,
        *_
    ) = fd.preprocessor.build_dataset(train=True)

    x_train = torch.tensor(x_train_n, dtype=torch.float32).view(x_train_n.shape[0], -1)
    y_train = torch.tensor(y_train_n, dtype=torch.float32).view(y_train_n.shape[0], -1)

    model_class = fd._get_model_class()
    fd.model = model_class.load_from_checkpoint(str(ckpt_path))

    device = _select_torch_device()
    fd.model = fd.model.to(device)
    fd.model.eval()
    fd.version_dir = str(version_dir)
    print(f"Model loaded:     {model_class.__name__} on {device}")

    previous_youden = _extract_previous_youden_threshold(version_dir)
    if previous_youden is not None:
        print(f"Reusing Youden J threshold from existing JSON: {previous_youden}")

    thresholds = fd.compute_production_thresholds(
        x_train,
        y_train,
        pids_train_n,
        youden_threshold=previous_youden,
        ts_train=ts_train_n,
    )

    output_path = version_dir / "production_thresholds.json"
    with open(output_path, "r") as f:
        saved = json.load(f)

    global_keys = saved.get("global", {})
    required_global_keys = [
        "window_warning_smoothed",
        "window_alarm_smoothed",
        "smoothing_alpha",
    ]
    missing = [k for k in required_global_keys if k not in global_keys]
    if missing:
        raise RuntimeError(f"Missing expected keys in global thresholds: {missing}")

    has_day_keys = "day_warning" in global_keys and "day_alarm" in global_keys

    print("\n" + "=" * 60)
    print("RECALIBRATION COMPLETE")
    print("=" * 60)
    print(f"Saved: {output_path}")
    print("Verified global keys:")
    print("  - window_warning_smoothed")
    print("  - window_alarm_smoothed")
    print("  - smoothing_alpha")
    print(f"  - day_warning/day_alarm present: {has_day_keys}")
    print("\nNew global thresholds:")
    print(f"  warning:                  {global_keys.get('warning')}")
    print(f"  alarm:                    {global_keys.get('alarm')}")
    print(f"  window_warning_smoothed:  {global_keys.get('window_warning_smoothed')}")
    print(f"  window_alarm_smoothed:    {global_keys.get('window_alarm_smoothed')}")
    if has_day_keys:
        print(f"  day_warning:              {global_keys.get('day_warning')}")
        print(f"  day_alarm:                {global_keys.get('day_alarm')}")
    print(f"  smoothing_alpha:          {global_keys.get('smoothing_alpha')}")

    # Keep variable referenced to avoid accidental linter warning in some environments.
    _ = thresholds

    return 0


if __name__ == "__main__":
    raise SystemExit(main())