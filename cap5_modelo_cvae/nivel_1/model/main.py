"""
Entry point for cond_reg_v2 - Temporal Conditional VAE Regressor.

Usage:
    cond-reg-v2 --mode train
    cond-reg-v2 --mode train --test-after-train
    cond-reg-v2 --mode test --version-dir path/to/version
    cond-reg-v2 --mode tune --n-trials 50
    cond-reg-v2 --mode calibrate --train-path ../../data/train/
"""

import argparse
import os

import pytorch_lightning as pl
import torch


def main():
    parser = argparse.ArgumentParser(description="cond_reg_v2 - Temporal Conditional VAE Regressor")
    parser.add_argument("--mode", choices=["train", "test", "tune", "calibrate"], default="train")
    parser.add_argument("--parameters", default=None, help="Path to parameters.json")
    parser.add_argument("--version-dir", default=None, help="Version directory for test mode")
    parser.add_argument("--n-trials", type=int, default=300, help="Number of Optuna trials for tune mode")
    parser.add_argument("--train-path", default=None, help="Training data path for calibrate mode")
    parser.add_argument(
        "--test-after-train",
        action="store_true",
        help="When --mode train, also run held-out test evaluation after training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Resolve default parameters path
    if args.parameters is None:
        args.parameters = os.path.join(os.path.dirname(os.path.dirname(__file__)), "parameters.json")

    # Set seeds for reproducibility
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps"):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if args.mode == "train":
        from .failure_detector import FailureDetector

        fd = FailureDetector(args.parameters, seed=args.seed)
        fd.train()
        if args.test_after_train:
            fd.test()

    elif args.mode == "test":
        from .failure_detector import FailureDetector

        fd = FailureDetector(args.parameters, seed=args.seed)
        fd.test(version_dir=args.version_dir)

    elif args.mode == "tune":
        from .failure_detector import FailureDetector
        from .fine_tuning import HyperparameterTuner

        tuner = HyperparameterTuner(args.parameters, n_trials=args.n_trials, seed=args.seed)
        best_params = tuner.run()
        print(f"\nBest parameters: {best_params}")

        # Automatically evaluate tuned weights on held-out test split.
        fd = FailureDetector(args.parameters, seed=args.seed)
        fd.test(version_dir=str(tuner.output_dir))

    elif args.mode == "calibrate":
        import json

        from .threshold_calibration import ThresholdCalibrator

        train_path = args.train_path
        if train_path is None:
            with open(args.parameters, "r", encoding="utf-8") as f:
                params = json.load(f)
            train_path = params.get("train_path", "../../data/train/")

        calibrator = ThresholdCalibrator(train_path=train_path)
        thresholds = calibrator.calibrate()

        output_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "model",
            "weights",
            "production_thresholds.json",
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=2)
        print(f"Thresholds saved to {output_path}")


if __name__ == "__main__":
    main()
