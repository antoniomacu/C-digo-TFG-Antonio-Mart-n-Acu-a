"""Custom detector plugins.

Place .py files in this directory to add custom anomaly detectors.
Each module must expose a DETECTORS dict mapping detector names to functions.

Detector function signature:
    def check(df, period, baseline, cfg) -> tuple[bool, str]

Where:
    df: pd.DataFrame — the full day's data
    period: OperationPeriod — the detected operation period
    baseline: PumpBaseline — per-pump baseline statistics
    cfg: SystemConfig — system configuration

Returns:
    (True, "reason_name") if anomaly detected
    (False, "") if no anomaly

Example module (my_detector.py):

    DETECTORS = {
        "my_custom_check": check_my_condition,
    }

    def check_my_condition(df, period, baseline, cfg):
        # your logic here
        return False, ""

Then add "my_custom_check" to the detectors list in your YAML config.
"""
