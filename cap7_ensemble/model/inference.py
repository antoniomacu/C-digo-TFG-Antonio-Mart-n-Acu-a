"""
Ensemble Pump Anomaly Detection — Production Inference (Day + Per-Window).

Usage:
    ensemble-inference pump_1_2025-10-06.csv pump_3_2025-10-06.csv [options]

Given CSV files (one per pump-day), runs both Level 1 (instantaneous) and
Level 2 (temporal) anomaly detection, fuses results, and outputs a JSON report.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .ensemble import EnsembleDetector


logger = logging.getLogger(__name__)


class EnsembleInference:
    """Thin wrapper for production use — load once, classify many times."""
    
    def __init__(
        self,
        level1_weights_dir: str | Path | None = None,
        level2_version_dir: str | Path | None = None,
        device: str = "auto",
    ):
        self._detector = EnsembleDetector(
            level1_weights_dir=level1_weights_dir,
            level2_version_dir=level2_version_dir,
            device=device,
        )
    
    def run(self, csv_paths: list[str], include_window_details: bool = True) -> dict:
        """Run inference and return report as dict."""
        report = self._detector.classify(csv_paths)
        return report.to_dict(include_window_details=include_window_details)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Ensemble pump anomaly detection — production inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ensemble-inference pump_1.csv pump_3.csv
  ensemble-inference *.csv --level1-weights cond_reg/model/weights/
  ensemble-inference *.csv -o report.json --verbose
        """
    )
    parser.add_argument('csv_files', nargs='+', help='Pump-day CSV files')
    parser.add_argument('--level1-weights', default=None, help='Level 1 weights directory')
    parser.add_argument('--level2-version', default=None, help='Level 2 model version directory')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Compute device')
    parser.add_argument('--output', '-o', default=None, help='Save JSON report to file')
    parser.add_argument('--verbose', action='store_true', help='Print full JSON report')
    parser.add_argument('--windows', action='store_true', help='Display per-window ensemble timeline')
    parser.add_argument('--no-window-details', action='store_true', help='Exclude per-window arrays from JSON output')
    
    args = parser.parse_args()
    
    # Validate files exist
    for f in args.csv_files:
        if not Path(f).exists():
            print(f"ERROR: File not found: {f}")
            sys.exit(1)
    
    # Run inference
    inference = EnsembleInference(
        level1_weights_dir=args.level1_weights,
        level2_version_dir=args.level2_version,
        device=args.device,
    )
    result = inference.run(
        args.csv_files,
        include_window_details=not getattr(args, 'no_window_details', False),
    )
    
    # Save or print full JSON
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=4, default=str)
        print(f"Report saved to {args.output}")
    elif args.verbose:
        print(json.dumps(result, indent=4, default=str))
    
    # Human-readable summary (always printed)
    print("\n" + "=" * 70)
    print("ENSEMBLE ANOMALY DETECTION REPORT")
    print("=" * 70)
    
    status_icons = {'NORMAL': '🟢', 'WARNING': '🟡', 'ALARM': '🔴'}
    
    for pr in result.get('pump_results', []):
        icon = status_icons.get(pr['overall_status'], '❓')
        print(f"\n  {icon} Pump {pr['pump_id']} ({pr['date']}): {pr['overall_status']}")
        print(f"     Severity: {pr['overall_severity']:.3f}")
        
        l1 = pr.get('level1', {})
        l1_icon = status_icons.get(l1.get('status', ''), '  ')
        print(f"     {l1_icon} Level 1 (instantaneous): {l1.get('status', 'N/A')}"
              f"  — max Mahalanobis={l1.get('day_max_mahalanobis', 0):.3f}")
        
        l2 = pr.get('level2')
        if l2:
            l2_icon = status_icons.get(l2.get('status', ''), '  ')
            print(f"     {l2_icon} Level 2 (temporal):      {l2.get('status', 'N/A')}"
                  f"  — day MSE={l2.get('day_error_mse', 0):.6f}")
            if l2.get('fraction_windows_alarm', 0) > 0:
                print(f"     Window stats: {l2.get('fraction_windows_warning', 0):.1%} windows ≥ WARNING, "
                      f"{l2.get('fraction_windows_alarm', 0):.1%} windows ≥ ALARM")
        else:
            print(f"     ⬜ Level 2 (temporal):      unavailable")
        
        print(f"     Reasoning: {pr.get('ensemble_reasoning', '')}")

    if args.windows:
        for pr in result.get('pump_results', []):
            windows = pr.get('window_ensemble_results', [])
            if not windows:
                continue
            print(f"\n  PUMP {pr['pump_id']} — Per-Window Ensemble Timeline ({pr['date']})")
            print(f"  {'Time':<22} {'L2 MSE(sm)':>12} {'L2':>8} {'L1 Mahal':>10} {'L1':>8} {'Ensemble':>10} {'Severity':>8}")
            print(f"  {'─'*22} {'─'*12} {'─'*8} {'─'*10} {'─'*8} {'─'*10} {'─'*8}")
            icons = {'NORMAL': '🟢', 'WARNING': '🟡', 'ALARM': '🔴'}
            for w in windows:
                ens_icon = icons.get(w.get('ensemble_status', ''), '  ')
                print(
                    f"  {w.get('timestamp', ''):<22}"
                    f" {w.get('level2_smoothed_mse', 0):>12.8f}"
                    f" {w.get('level2_status', 'N/A'):>8}"
                    f" {w.get('level1_max_mahalanobis', 0):>10.3f}"
                    f" {w.get('level1_status', 'N/A'):>8}"
                    f" {ens_icon} {w.get('ensemble_status', 'N/A'):>8}"
                    f" {w.get('ensemble_severity', 0):>8.3f}"
                )
    
    timing = result.get('timing', {})
    print(f"\n  ⏱  Level 1:  {timing.get('level1_seconds', 0):.3f}s")
    print(f"  ⏱  Level 2:  {timing.get('level2_seconds', 0):.3f}s")
    print(f"  ⏱  Total:    {timing.get('total_seconds', 0):.3f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
