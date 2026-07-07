# Main
import os
import random

import numpy as np
import torch
import pytorch_lightning as pl

from . import PARAMETERS_DIR
from .failure_detector import FailureDetector

def set_global_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Set global RNG seeds for reproducible training/inference."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True   # safe with deterministic=True; lets cuDNN pick fastest kernels

    # TF32 on Ampere+ GPUs: ~2× faster matmuls with negligible precision loss
    torch.set_float32_matmul_precision("medium")

    pl.seed_everything(seed, workers=True)
    print(f"✓ Random seed set to {seed} for reproducibility")
    return seed



def main():
    seed = set_global_seed(42)
    parameters_file = str(PARAMETERS_DIR / 'best_params.json')  # Resolved from package location

    # ── Toggle switches ─────────────────────────────────────────────────
    tune      = True     # Set True to run full pipeline: fine-tune → train → test
    train     = False    # Set True to train the model
    test      = False    # Set True to evaluate on held-out test sets after training
    benchmark = False    # Set True to run the full model comparison benchmark
    # ────────────────────────────────────────────────────────────────────

    # When tuning, always run the full pipeline so the best params are
    # immediately validated with a production train + test cycle.
    if tune:
        train = True
        test  = True

    # ── Standalone test configuration ───────────────────────────────────
    # To test a previously trained model without retraining:
    #   1. Set train=False and test=True
    #   2. Specify the version directory containing model_weights.ckpt
    #      and norm_params.json
    #   3. Ensure parameters.json has the correct "model" field
    #      (must match the architecture used during training)
    test_version_dir = None   # e.g., "lightning_logs/metrics/version_0"
    # ────────────────────────────────────────────────────────────────────

    # BENCHMARK: Run full model comparison (separate from main pipeline)
    if benchmark:
        from .comparison.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(parameters_file, seed=seed)
        runner.run()
        return

    # FINE-TUNING PHASE (optional): grid-search over hyperparameters
    # Writes the best configuration found and updates parameters.json
    if tune:
        from .fine_tuning import HyperparameterTuner

        tuner = HyperparameterTuner(
            parameters_file,
            seed=seed,
            # Default: 300 trials, 100% data, same epochs/patience as production.
            # Override defaults here if you want a smaller/faster search:
            # n_trials=100,
            # data_fraction=0.5,
            # grid_epochs=50,
            # grid_patience=15,
        )
        best_params = tuner.run()

        # Persist best params so training uses them
        import json
        with open(parameters_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"\n✓ parameters.json updated with best hyperparameters")

    # Initialize the model with:
    # - pump-balanced sampling (WeightedRandomSampler — all data used)
    # - 80% training / 10% validation / 10% test split for normal data
    # - 50% validation / 50% test split for abnormal data
    fd = FailureDetector(
        parameters_file, 
        seed=seed,
        train_split=0.8,           # 80% of normal data for training
        val_normal_split=0.1,      # 10% of normal data for validation
        abnormal_val_split=0.5     # 50% of abnormal data for validation, 50% for test
    )

    # TRAINING PHASE: Train model and evaluate on validation sets only
    # Test sets are held out and NOT touched during this phase
    # The model architecture is selected via "model" in parameters.json
    # Available: vae, standard_ae, sparse_ae, denoising_ae, lstm_ae,
    #            cnn_ae, transformer_ae, usad
    if train:
        fd.train()
    
    # TESTING PHASE: Evaluate on held-out test sets
    # Can be run immediately after training or standalone with a version_dir
    if test:
        fd.test(version_dir=test_version_dir)


if __name__ == '__main__':  # ensure the code only runs if you execute this file
    main()