"""Device auto-detection for platform-agnostic training (CUDA / MPS / CPU)."""
import torch


def select_accelerator() -> str:
    """Return the best available Lightning accelerator for this machine."""
    if torch.cuda.is_available():
        return "gpu"  # CUDA
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


def should_pin_memory() -> bool:
    """pin_memory only benefits CUDA host->device transfers."""
    return torch.cuda.is_available()


def select_precision() -> str:
    """Return the best mixed-precision setting for this machine.

    - CUDA Ampere+ (sm_80+): '16-mixed' (FP16 Tensor Cores)
    - Older CUDA: '32' (full precision)
    - MPS / CPU: '32'
    """
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:          # Ampere, Ada Lovelace, Blackwell …
            return "16-mixed"
    return "32"
