#!/usr/bin/env python3
"""Quick import smoke test for cond_reg_v2 package."""
import sys
sys.path.insert(0, '<PATH_TO_PROJECT>')

try:
    from cond_reg_v2.model.models import TemporalCVAE, Encoder, Decoder
    print("OK models.py")
except Exception as e:
    print(f"FAIL models.py: {e}")

try:
    from cond_reg_v2.model.inference import PumpPredictor
    print("OK inference.py")
except Exception as e:
    print(f"FAIL inference.py: {e}")

try:
    from cond_reg_v2.model.failure_detector import FailureDetector
    print("OK failure_detector.py")
except Exception as e:
    print(f"FAIL failure_detector.py: {e}")

try:
    from cond_reg_v2.model.fine_tuning import HyperparameterTuner
    print("OK fine_tuning.py")
except Exception as e:
    print(f"FAIL fine_tuning.py: {e}")

try:
    from cond_reg_v2.model.threshold_calibration import ThresholdCalibrator
    print("OK threshold_calibration.py")
except Exception as e:
    print(f"FAIL threshold_calibration.py: {e}")

print("\nDone.")
