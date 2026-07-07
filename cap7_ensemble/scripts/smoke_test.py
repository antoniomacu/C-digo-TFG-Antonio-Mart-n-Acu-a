"""Temporary smoke test for plotting_demo - delete after use."""
import os
from pathlib import Path

os.chdir(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import sys
sys.argv = ['plotting_demo', '--pump', '1', '--source', 'abnormal', '--speed', '0']

from demos.plotting_demo import _parse_args, _choose_csv, _load_day, SHORT_NAMES, SENSOR_GROUPS
from model.streaming import create_streaming_detector

args = _parse_args()
csv_path, pump_id, source_name = _choose_csv(args)
print('CSV:', csv_path.name)
print('Pump:', pump_id, 'Source:', source_name)

df = _load_day(csv_path)
print('Rows:', len(df))

all_sensors = [s for _, group in SENSOR_GROUPS for s in group]
print('Sensors:', len(all_sensors))

detector = create_streaming_detector()
detector.reset_pump(pump_id)
row = df.iloc[0]
ts = df.index[0]
result = detector.process_timestep(pump_id, ts, row)
print('L1 actual keys:', len(result.l1_actual))
print('L1 predicted keys:', len(result.l1_predicted))
print('L1 z_scores keys:', len(result.l1_z_scores))
print('Ensemble:', result.ensemble_status, result.ensemble_health)
print('Mahalanobis:', result.l1_mahalanobis)
print('L2 status:', result.l2_status)
print('SMOKE TEST PASSED')
