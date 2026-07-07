"""
Feedwater pump data extraction script (v3).

Extends v2 with:
  - Per-sensor quality columns (<sensor>_quality) with numeric error codes:
      0 = OK
      1 = Missing data (gap in expected time range)
      2 = Null / NaN
      3 = Out of range (below min_value or above max_value)
      4 = Fall down (high variability / sensor failure)
      5 = Frozen data (constant value for extended period)
    Highest code wins when multiple issues overlap on the same point.

  - Frozen-day detection: when ALL value columns in a file are frozen,
    the filename is prefixed with [FROZEN].

  - Fixed column order across all output files (value, quality pairs).

  - Data source modes:
      --source files  → reads from pre-downloaded v2 xlsx files  (FAST, no DB)
      --source local  → downloads from DB, sensor config from the Excel
      --source db     → downloads from DB, sensor config from kks_description
      --source auto   → Windows→files, Linux→files  (default)

Reads KKS definitions from 'SENSOR_DEFINITIONS.xlsx'
and either downloads raw data from the plant_db_raw database or reads
pre-existing v2 Excel files, producing four Excel files per day:

  - fw_pump_1_<date>.xlsx  (equipment_id = 979)
  - fw_pump_2_<date>.xlsx  (equipment_id = 980)
  - fw_pump_3_<date>.xlsx  (equipment_id = 981)
  - common_<date>.xlsx     (equipment_id = 978 + ambient temperature)

Output structure:
    data/fw_pumps_v3/<YYYY>/<MM>/<DD>/*.xlsx

Log file:
    data/fw_pumps_v3/fw_pump_extraction_v3.log

Usage:
    python extract_fw_pump_data_v3.py --start 2026-01-01 --end 2026-01-31
    python extract_fw_pump_data_v3.py -s 2026-02-01 -e 2026-02-15 -p 1,3
    python extract_fw_pump_data_v3.py -s 2026-03-01 -e 2026-03-01 --source db
    python extract_fw_pump_data_v3.py -s 2023-01-01 -e 2026-03-24 --source files
"""

import os
import sys
import re
import time
import platform
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import click
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Lazy import: only needed for 'local' and 'db' source modes (requires DB)
raw_models = None

def _ensure_raw_models():
    global raw_models
    if raw_models is None:
        from models import raw as _raw
        raw_models = _raw


# =============================================================================
# CONSTANTS
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_FILE = os.path.join(SCRIPT_DIR, 'SENSOR_DEFINITIONS.xlsx')
OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, 'data', 'fw_pumps_v3')
V2_INPUT_DIR = os.path.join(SCRIPT_DIR, 'data', 'fw_pumps_v2')

PUMP_EQUIPMENT_MAP = {979: 1, 980: 2, 981: 3}
COMMON_EQUIPMENT_ID = 978

EXCLUDED_KKS = frozenset({
    'r1lac21ct005', 'r1lac21ct004',
    'r1lac22ct005', 'r1lac22ct004',
    'r1lac23ct004', 'r1lac23ct005',
})

AMBIENT_TEMPERATURE_KKS = 'r1cf_01ct007'

PUMP_PREFIX_RE = re.compile(r'^FW\s+Pump\s+\d+\s+', re.IGNORECASE)

# Quality codes (ordinal — highest wins)
Q_OK = 0
Q_MISSING = 1
Q_NULL = 2
Q_OUT_OF_RANGE = 3
Q_FALL_DOWN = 4
Q_FROZEN = 5

# Quality-check thresholds
FROZEN_TOLERANCE = 1e-6
FROZEN_PERIOD_SECONDS = 3600
VARIABILITY_THRESHOLD = 0.05
ROLLING_WINDOW = '30s'
ROLLING_MIN_PERIODS = 3

# Excel read engine: calamine (Rust-based) is ~2x faster than openpyxl.
# Falls back to openpyxl when python-calamine is not installed or pandas
# does not support it (requires pandas >= 2.2).
def _detect_read_engine() -> str:
    try:
        import io
        import python_calamine  # noqa: F401
        pd.read_excel(io.BytesIO(b''), engine='calamine')
    except ImportError:
        return 'openpyxl'
    except Exception:
        # BytesIO(b'') is not a valid xlsx, but if we get past ImportError
        # and into a parse/value error, it means the engine is accepted.
        return 'calamine'
    return 'calamine'

XLSX_READ_ENGINE = _detect_read_engine()

# Fallback sensor defaults (when no config available from Excel or DB)
SENSOR_DEFAULTS = {
    "Speed": {
        "min_value": 0, "max_value": 3000,
        "check_constant": True, "check_falldown": True,
    },
    "Flow": {
        "min_value": 0, "max_value": 3500,
        "check_constant": True, "check_falldown": True,
    },
    "Current Consumption": {
        "min_value": 0, "max_value": 500,
        "check_constant": False, "check_falldown": True,
    },
    "Outlet Pressure": {
        "min_value": 0, "max_value": 30,
        "check_constant": True, "check_falldown": True,
    },
    "Inlet Temperature": {
        "min_value": 50, "max_value": 450,
        "check_constant": False, "check_falldown": True,
    },
    "Motor current": {
        "min_value": 0, "max_value": 500,
        "check_constant": False, "check_falldown": True,
    },
    "Ambient temperature": {
        "min_value": -10, "max_value": 60,
        "check_constant": False, "check_falldown": False,
    },
}


# =============================================================================
# FAST FILE WRITERS
# =============================================================================

def _fast_write_excel(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to xlsx using xlsxwriter directly with
    constant_memory mode.  ~2x faster than pandas ``to_excel`` for large
    DataFrames (avoids per-cell Python overhead in the pandas bridge)."""
    import xlsxwriter

    idx_str = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
    cols = list(df.columns)
    data = df.values
    nan_mask = np.isnan(data)
    nrows, ncols = data.shape

    wb = xlsxwriter.Workbook(path, {'constant_memory': True})
    ws = wb.add_worksheet('Sheet1')
    ws.write_row(0, 0, [df.index.name or 'timestamp'] + cols)
    for i in range(nrows):
        row = [idx_str[i]]
        d = data[i]
        m = nan_mask[i]
        for j in range(ncols):
            row.append(None if m[j] else d[j])
        ws.write_row(i + 1, 0, row)
    wb.close()


def _fast_write_csv(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV.  ~10x faster than xlsx for large files."""
    df.to_csv(path)


def _fast_write(df: pd.DataFrame, path: str) -> None:
    """Atomic write: writes to a temp file in the same directory, then
    renames to the final path.  This avoids leaving corrupt partial files
    if the process is interrupted mid-write."""
    import tempfile

    target_dir = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(
        suffix=os.path.splitext(path)[1],
        dir=target_dir,
    )
    os.close(fd)
    try:
        if path.endswith('.csv'):
            _fast_write_csv(df, tmp_path)
        else:
            _fast_write_excel(df, tmp_path)
        os.replace(tmp_path, path)  # atomic on same filesystem
    except BaseException:
        # Clean up temp file on any failure (including KeyboardInterrupt)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# =============================================================================
# LOGGER
# =============================================================================

def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'fw_pump_extraction_v3.log')

    logger = logging.getLogger('fw_pump_extraction_v3')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =============================================================================
# QUALITY CHECK
# =============================================================================

def _infer_timestep(idx: pd.DatetimeIndex) -> int:
    """Infer the most common timestep (in seconds) from a DatetimeIndex."""
    if len(idx) < 2:
        return 60
    # Use numpy for speed — avoid pd.Series overhead
    diffs_ns = np.diff(idx.values).astype('timedelta64[s]').astype(np.int64)
    vals, counts = np.unique(diffs_ns, return_counts=True)
    return max(1, int(vals[counts.argmax()]))


def compute_quality(
    df: pd.DataFrame,
    col_name: str,
    timestep: int | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    check_constant: bool = True,
    check_falldown: bool = True,
) -> pd.DataFrame:
    """
    Add a ``<col_name>_quality`` column to *df* (which must have a
    DatetimeIndex and a value column *col_name*).

    Quality codes (highest wins per point):
        0 = OK
        1 = Missing data (timestamp gap)
        2 = Null / NaN value
        3 = Out of range
        4 = Fall down (high variability)
        5 = Frozen data (constant for ≥ FROZEN_PERIOD_SECONDS)

    Frozen detection only runs when *check_constant* is True.
    If *timestep* is ``None`` it is auto-detected from the index.
    """
    qcol = f'{col_name}_quality'

    if timestep is None or timestep <= 0:
        timestep = _infer_timestep(df.index)

    # --- 1. Fill missing timestamps (rare, so check first) ---------------
    missing_mask: np.ndarray | None = None
    if not df.empty:
        idx_set = set(df.index)
        expected = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq=f'{timestep}s',
        )
        if len(expected) != len(idx_set):
            missing_ts = sorted(set(expected) - idx_set)
            if missing_ts:
                missing_set = set(missing_ts)
                missing_df = pd.DataFrame(
                    {col_name: np.nan},
                    index=pd.DatetimeIndex(missing_ts, name=df.index.name),
                )
                df = pd.concat([df, missing_df])
                df.sort_index(inplace=True)
                # Track which rows were gap-inserted so we can label them
                # Q_MISSING after all other checks.
                missing_mask = np.array(
                    [t in missing_set for t in df.index], dtype=bool,
                )

    # Work on numpy arrays for speed
    vals = df[col_name].values.astype(np.float64, copy=False)
    n = len(vals)
    q = np.zeros(n, dtype=np.int8)

    # --- 2. Null / NaN ---------------------------------------------------
    is_nan = np.isnan(vals)
    q[is_nan] = Q_NULL

    # --- 2b. Override gap-inserted rows with Q_MISSING -------------------
    # These rows have NaN because the timestamp was absent, not because the
    # sensor reported null.  Force Q_MISSING so it is distinguishable.
    if missing_mask is not None:
        q[missing_mask] = Q_MISSING

    # Fast path: if everything is NaN, nothing else to check
    if is_nan.all():
        df[qcol] = q
        return df

    # --- 3. Out of range -------------------------------------------------
    if min_value is not None or max_value is not None:
        oor = np.zeros(n, dtype=bool)
        if min_value is not None:
            oor |= vals < min_value
        if max_value is not None:
            oor |= vals > max_value
        # NaN comparisons return False, which is correct
        upgrade = oor & (q < Q_OUT_OF_RANGE)
        q[upgrade] = Q_OUT_OF_RANGE

    # --- 4. Fall down (high variability) ---------------------------------
    if check_falldown and min_value is not None and max_value is not None:
        value_range = abs(max_value - min_value)
        if value_range > 0:
            rolling_std = df[col_name].rolling(
                window=ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS,
            ).std()
            ratio = rolling_std.values / value_range
            fd = np.abs(ratio) > VARIABILITY_THRESHOLD
            upgrade = fd & (q < Q_FALL_DOWN)
            q[upgrade] = Q_FALL_DOWN

    # --- 5. Frozen data (only when check_constant=True) ------------------
    if check_constant and timestep > 0:
        diff = np.abs(np.diff(vals, prepend=np.nan))
        frozen = diff < FROZEN_TOLERANCE
        # Find run lengths of consecutive frozen points
        points_needed = max(1, int(FROZEN_PERIOD_SECONDS / timestep))
        # Use numpy run-length encoding for speed
        changes = np.diff(frozen.astype(np.int8), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) >= points_needed:
                mask = slice(s, e)
                q[mask] = np.maximum(q[mask], Q_FROZEN)

    df[qcol] = q
    return df


# =============================================================================
# FROZEN-DAY DETECTION
# =============================================================================

def is_all_frozen(df: pd.DataFrame) -> bool:
    """
    Return True when EVERY value column (i.e. all columns whose name does
    NOT end with ``_quality``) is frozen for the entire file.

    A single column is frozen when every consecutive pair of non-NaN values
    has ``abs(diff) < FROZEN_TOLERANCE`` (i.e. the signal never changes).
    If a column is entirely NaN it is also considered frozen.

    This is intentionally SUPER restrictive: one single non-frozen column
    means the whole file is NOT frozen.
    """
    if df.empty:
        return False

    value_cols = [c for c in df.columns if not c.endswith('_quality')]
    if not value_cols:
        return False

    for col in value_cols:
        arr = df[col].values.astype(np.float64, copy=False)
        # Drop NaN using numpy — faster than pandas .dropna()
        valid = arr[~np.isnan(arr)]
        if len(valid) <= 1:
            continue
        if np.any(np.abs(np.diff(valid)) >= FROZEN_TOLERANCE):
            return False

    return True


# =============================================================================
# DEFINITIONS LOADER — LOCAL (Excel)
# =============================================================================

def load_definitions_local(
    excel_path: str,
) -> tuple[dict[int, list[dict]], list[dict]]:
    """
    Read the Excel and return enriched sensor dicts that include quality-check
    config read from the same sheet:

      pump_sensors  – {pump_number: [{kks, column, timestep, min_value,
                        max_value, check_constant, check_falldown}, ...]}
      common_sensors – same structure

    Columns read from Excel ('FW' sheet):
        customer_name (kks), visiom_name, equipment_id,
        timestep, min_value, max_value, check_constant, check_falldown
    """
    df = pd.read_excel(excel_path, sheet_name='FW')

    raw_pump: dict[int, list[tuple]] = {1: [], 2: [], 3: []}
    raw_common: list[tuple] = []

    for _, row in df.iterrows():
        kks = str(row['customer_name (kks)']).strip().lower()
        visiom = str(row['visiom_name']).strip()
        eid = int(row['equipment_id'])

        if kks in EXCLUDED_KKS:
            continue

        # Quality-check config from Excel (with safe fallbacks)
        cfg = {
            'timestep': int(row['timestep']) if pd.notna(row.get('timestep')) else 60,
            'min_value': float(row['min_value']) if pd.notna(row.get('min_value')) else None,
            'max_value': float(row['max_value']) if pd.notna(row.get('max_value')) else None,
            'check_constant': bool(row['check_constant']) if pd.notna(row.get('check_constant')) else False,
            'check_falldown': bool(row['check_falldown']) if pd.notna(row.get('check_falldown')) else False,
        }

        if eid == COMMON_EQUIPMENT_ID:
            raw_common.append((kks, visiom, cfg))
        elif eid in PUMP_EQUIPMENT_MAP:
            pump_num = PUMP_EQUIPMENT_MAP[eid]
            generic = PUMP_PREFIX_RE.sub('', visiom)
            raw_pump[pump_num].append((kks, generic, cfg))

    def _deduplicate(entries):
        seen: dict[str, int] = {}
        result: list[dict] = []
        for kks, col, cfg in entries:
            seen[col] = seen.get(col, 0) + 1
            if seen[col] > 1:
                col = f'{col} ({seen[col]})'
            result.append({'kks': kks, 'column': col, **cfg})
        return result

    pump_sensors = {pn: _deduplicate(raw_pump[pn]) for pn in raw_pump}
    common_sensors = _deduplicate(raw_common)

    return pump_sensors, common_sensors


# =============================================================================
# DEFINITIONS LOADER — REMOTE (Database)
# =============================================================================

def load_definitions_db(
    excel_path: str,
) -> tuple[dict[int, list[dict]], list[dict]]:
    """
    Same interface as ``load_definitions_local`` but overrides quality-check
    config with values from the ``kks_description`` table in the raw database.

    The Excel is still used to determine the KKS→pump mapping, column names,
    and equipment grouping — only the quality-check parameters are replaced.
    """
    pump_sensors, common_sensors = load_definitions_local(excel_path)

    _ensure_raw_models()

    def _enrich(sensors: list[dict]) -> list[dict]:
        for s in sensors:
            try:
                desc = raw_models.KKSDescription.get_kks_description(s['kks'])
                if desc.min_value is not None:
                    s['min_value'] = float(desc.min_value)
                if desc.max_value is not None:
                    s['max_value'] = float(desc.max_value)
                if desc.check_constant is not None:
                    s['check_constant'] = bool(desc.check_constant)
                if desc.check_falldown is not None:
                    s['check_falldown'] = bool(desc.check_falldown)
                if desc.timestep is not None:
                    s['timestep'] = int(desc.timestep)
            except raw_models.NoKksData:
                pass  # keep Excel / default values
        return sensors

    for pn in pump_sensors:
        pump_sensors[pn] = _enrich(pump_sensors[pn])
    common_sensors = _enrich(common_sensors)

    return pump_sensors, common_sensors


# =============================================================================
# APPLY FALLBACK DEFAULTS
# =============================================================================

def apply_sensor_defaults(sensors: list[dict]) -> list[dict]:
    """
    For sensors that still have no min/max values after loading from Excel/DB,
    try to match by column name against SENSOR_DEFAULTS and fill in the gaps.
    """
    for s in sensors:
        for key, defaults in SENSOR_DEFAULTS.items():
            if key.lower() in s['column'].lower():
                if s.get('min_value') is None and 'min_value' in defaults:
                    s['min_value'] = defaults['min_value']
                if s.get('max_value') is None and 'max_value' in defaults:
                    s['max_value'] = defaults['max_value']
                if 'check_constant' in defaults and not s.get('check_constant'):
                    s['check_constant'] = defaults['check_constant']
                if 'check_falldown' in defaults and not s.get('check_falldown'):
                    s['check_falldown'] = defaults['check_falldown']
                break
    return sensors


# =============================================================================
# CANONICAL COLUMN ORDER
# =============================================================================

def canonical_column_order(pump_sensors: dict[int, list[dict]]) -> list[str]:
    """
    Column order derived from pump 1's sensor list.
    Produces interleaved [value, value_quality] pairs.
    """
    order: list[str] = []
    for s in pump_sensors[1]:
        order.append(s['column'])
        order.append(f"{s['column']}_quality")
    return order


def common_column_order(common_sensors: list[dict]) -> list[str]:
    """
    Column order for the common file (interleaved value + quality pairs).
    Ambient temperature is appended at the end.
    """
    order: list[str] = []
    for s in common_sensors:
        order.append(s['column'])
        order.append(f"{s['column']}_quality")
    order.append('Ambient temperature')
    order.append('Ambient temperature_quality')
    return order


# =============================================================================
# DOWNLOAD + QUALITY
# =============================================================================

def download_kks_with_quality(
    kks: str,
    date: datetime,
    col_name: str,
    sensor_cfg: dict,
    logger: logging.Logger,
) -> pd.DataFrame | None:
    """
    Download one KKS for one day and add a quality column.
    Returns a two-column DataFrame (value + quality) or None.
    """
    _ensure_raw_models()
    try:
        df = raw_models.KKSDescription.get_kks_df(
            kks.lower(), date,
            column_name=col_name,
            index_name='timestamp',
        )
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = compute_quality(
            df,
            col_name=col_name,
            timestep=sensor_cfg.get('timestep', 60),
            min_value=sensor_cfg.get('min_value'),
            max_value=sensor_cfg.get('max_value'),
            check_constant=sensor_cfg.get('check_constant', False),
            check_falldown=sensor_cfg.get('check_falldown', False),
        )
        return df

    except raw_models.NoKksData:
        return None
    except Exception as e:
        logger.error(f"Error downloading KKS '{kks}': {e}")
        return None


# =============================================================================
# SINGLE-DAY EXTRACTION
# =============================================================================

def extract_day(
    date: datetime,
    pump_sensors: dict[int, list[dict]],
    common_sensors: list[dict],
    pump_col_order: list[str],
    common_col_ord: list[str],
    output_dir: str,
    logger: logging.Logger,
    selected_pumps: set[int] | None = None,
):
    """
    Downloads and saves the four Excel files for a single day.
    Returns a dict mapping category -> list of missing KKS (only if any).
    """
    day_str = date.strftime('%Y-%m-%d')
    day_dir = os.path.join(
        output_dir,
        date.strftime('%Y'), date.strftime('%m'), date.strftime('%d'),
    )

    logger.info(f"{'=' * 60}")
    logger.info(f"Processing {day_str}")
    logger.info(f"{'=' * 60}")

    missing_report: dict[str, list[str]] = {}

    # ---- Pumps 1, 2, 3 -----------------------------------------------
    for pump_num in sorted(pump_sensors):
        if selected_pumps and pump_num not in selected_pumps:
            continue

        sensors = pump_sensors[pump_num]
        label = f"FW Pump {pump_num}"
        dfs: list[pd.DataFrame] = []
        missing: list[str] = []

        for sensor in sensors:
            result = download_kks_with_quality(
                sensor['kks'], date, sensor['column'], sensor, logger,
            )
            if result is not None:
                dfs.append(result)
            else:
                missing.append(sensor['kks'])

        if dfs:
            merged = pd.concat(dfs, axis=1)
            # Enforce canonical column order (skip cols not present)
            ordered = [c for c in pump_col_order if c in merged.columns]
            merged = merged[ordered]
            merged.sort_index(inplace=True)

            # Frozen-day detection (only on value columns)
            frozen = is_all_frozen(merged)

            os.makedirs(day_dir, exist_ok=True)
            fname = f'fw_pump_{pump_num}_{day_str}.xlsx'
            if frozen:
                fname = f'[FROZEN]fw_pump_{pump_num}_{day_str}.xlsx'
                logger.warning(f"{label:<12s} | {day_str} | ALL COLUMNS FROZEN")

            path = os.path.join(day_dir, fname)
            _fast_write(merged, path)
            logger.info(
                f"{label:<12s} | {day_str} | OK  "
                f"({merged.shape[0]} rows, {merged.shape[1]} cols)"
                + (" [FROZEN]" if frozen else "")
            )
        else:
            logger.warning(f"{label:<12s} | {day_str} | No data — file skipped")

        if missing:
            missing_report[label] = missing
            logger.warning(
                f"{label:<12s} | {day_str} | MISSING KKS "
                f"({len(missing)}/{len(sensors)}): " + ', '.join(missing)
            )

    # ---- Common -------------------------------------------------------
    dfs_common: list[pd.DataFrame] = []
    missing_common: list[str] = []

    for sensor in common_sensors:
        result = download_kks_with_quality(
            sensor['kks'], date, sensor['column'], sensor, logger,
        )
        if result is not None:
            dfs_common.append(result)
        else:
            missing_common.append(sensor['kks'])

    # Ambient temperature
    amb_cfg = SENSOR_DEFAULTS.get('Ambient temperature', {})
    amb = download_kks_with_quality(
        AMBIENT_TEMPERATURE_KKS, date, 'Ambient temperature',
        {'timestep': 60, **amb_cfg}, logger,
    )
    if amb is not None:
        dfs_common.append(amb)
    else:
        missing_common.append(AMBIENT_TEMPERATURE_KKS)

    if dfs_common:
        merged = pd.concat(dfs_common, axis=1)
        ordered = [c for c in common_col_ord if c in merged.columns]
        merged = merged[ordered]
        merged.sort_index(inplace=True)

        frozen = is_all_frozen(merged)

        os.makedirs(day_dir, exist_ok=True)
        fname = f'common_{day_str}.xlsx'
        if frozen:
            fname = f'[FROZEN]common_{day_str}.xlsx'
            logger.warning(f"{'Common':<12s} | {day_str} | ALL COLUMNS FROZEN")

        path = os.path.join(day_dir, fname)
        _fast_write(merged, path)
        logger.info(
            f"{'Common':<12s} | {day_str} | OK  "
            f"({merged.shape[0]} rows, {merged.shape[1]} cols)"
            + (" [FROZEN]" if frozen else "")
        )
    else:
        logger.warning(f"{'Common':<12s} | {day_str} | No data — file skipped")

    if missing_common:
        missing_report['Common'] = missing_common
        logger.warning(
            f"{'Common':<12s} | {day_str} | MISSING KKS "
            f"({len(missing_common)}): " + ', '.join(missing_common)
        )

    return missing_report


# =============================================================================
# REPROCESS FROM EXISTING V2 FILES
# =============================================================================

def _build_sensor_config_map(
    pump_sensors: dict[int, list[dict]],
    common_sensors: list[dict],
) -> dict[str, dict]:
    """
    Build a column_name → sensor_config lookup from the loaded definitions.
    Works for both pump and common sensors.
    """
    cfg_map: dict[str, dict] = {}
    for pn in pump_sensors:
        for s in pump_sensors[pn]:
            cfg_map[s['column']] = s
    for s in common_sensors:
        cfg_map[s['column']] = s
    # Ambient temperature (always present in common files)
    if 'Ambient temperature' not in cfg_map:
        amb = SENSOR_DEFAULTS.get('Ambient temperature', {})
        cfg_map['Ambient temperature'] = {
            'column': 'Ambient temperature',
            'timestep': 60,
            'min_value': amb.get('min_value'),
            'max_value': amb.get('max_value'),
            'check_constant': amb.get('check_constant', False),
            'check_falldown': amb.get('check_falldown', False),
        }
    return cfg_map


def _apply_quality_to_df(
    df: pd.DataFrame,
    cfg_map: dict[str, dict],
    col_order: list[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Take a v2 DataFrame (timestamp index, value columns only) and add a
    quality column per sensor, then enforce the canonical column order.
    """
    if df.empty:
        return pd.DataFrame()

    # Auto-detect actual timestep once for all columns
    timestep = _infer_timestep(df.index) if len(df) >= 2 else 60

    # Build a single-column DataFrame for compute_quality.
    # .copy() avoids SettingWithCopyWarning when compute_quality assigns
    # a new column on the slice.
    original_cols = list(df.columns)
    for col in original_cols:
        sensor_cfg = cfg_map.get(col, {})
        single = df[[col]].copy()
        single = compute_quality(
            single,
            col_name=col,
            timestep=timestep,
            min_value=sensor_cfg.get('min_value'),
            max_value=sensor_cfg.get('max_value'),
            check_constant=sensor_cfg.get('check_constant', False),
            check_falldown=sensor_cfg.get('check_falldown', False),
        )
        qcol = f'{col}_quality'
        if len(single) != len(df):
            df = df.reindex(single.index)
        df[qcol] = single[qcol].values

    # Enforce canonical column order
    ordered = [c for c in col_order if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    return df[ordered + extra]


def _write_with_frozen_check(
    merged: pd.DataFrame,
    day_dir: str,
    base_fname: str,
    label: str,
    day_str: str,
    logger: logging.Logger,
) -> bool:
    """
    Write the merged DataFrame to Excel, prefixing [FROZEN] if all value
    columns are frozen.  Returns True if frozen.
    """
    frozen = is_all_frozen(merged)

    os.makedirs(day_dir, exist_ok=True)
    fname = f'[FROZEN]{base_fname}' if frozen else base_fname
    # Remove stale counterpart (frozen ↔ unfrozen) from previous runs
    stale = base_fname if frozen else f'[FROZEN]{base_fname}'
    stale_path = os.path.join(day_dir, stale)
    if os.path.isfile(stale_path):
        os.unlink(stale_path)
    path = os.path.join(day_dir, fname)
    _fast_write(merged, path)

    suffix = " [FROZEN]" if frozen else ""
    if frozen:
        logger.warning(f"{label:<12s} | {day_str} | ALL COLUMNS FROZEN")
    logger.info(
        f"{label:<12s} | {day_str} | OK  "
        f"({merged.shape[0]} rows, {merged.shape[1]} cols){suffix}"
    )
    return frozen


def reprocess_day(
    date: datetime,
    pump_sensors: dict[int, list[dict]],
    common_sensors: list[dict],
    pump_col_order: list[str],
    common_col_ord: list[str],
    input_dir: str,
    output_dir: str,
    logger: logging.Logger,
    cfg_map: dict[str, dict],
    selected_pumps: set[int] | None = None,
) -> dict[str, str]:
    """
    Read pre-existing v2 Excel files for *date*, apply quality checks,
    enforce column order, detect frozen days, and write the v3 output.

    Much faster than downloading from the database since the data is
    already on disk.

    Returns a dict mapping category -> status note (missing file, frozen, ok).
    """
    day_str = date.strftime('%Y-%m-%d')
    in_day_dir = os.path.join(
        input_dir,
        date.strftime('%Y'), date.strftime('%m'), date.strftime('%d'),
    )
    out_day_dir = os.path.join(
        output_dir,
        date.strftime('%Y'), date.strftime('%m'), date.strftime('%d'),
    )

    logger.info(f"{'=' * 60}")
    logger.info(f"Reprocessing {day_str}")
    logger.info(f"{'=' * 60}")

    report: dict[str, str] = {}

    # ---- Pumps 1, 2, 3 -----------------------------------------------
    for pump_num in sorted(pump_sensors):
        if selected_pumps and pump_num not in selected_pumps:
            continue

        label = f"FW Pump {pump_num}"
        v2_fname = f'fw_pump_{pump_num}_{day_str}.xlsx'
        v2_path = os.path.join(in_day_dir, v2_fname)

        if not os.path.isfile(v2_path):
            logger.warning(f"{label:<12s} | {day_str} | v2 file not found — skipped")
            report[label] = 'missing'
            continue

        df = pd.read_excel(v2_path, index_col=0, engine=XLSX_READ_ENGINE)
        df.index.name = 'timestamp'

        merged = _apply_quality_to_df(df, cfg_map, pump_col_order, logger)
        if merged.empty:
            logger.warning(f"{label:<12s} | {day_str} | Empty after processing — skipped")
            report[label] = 'empty'
            continue

        frozen = _write_with_frozen_check(
            merged, out_day_dir,
            f'fw_pump_{pump_num}_{day_str}.xlsx',
            label, day_str, logger,
        )
        report[label] = 'frozen' if frozen else 'ok'

    # ---- Common -------------------------------------------------------
    label = 'Common'
    v2_fname = f'common_{day_str}.xlsx'
    v2_path = os.path.join(in_day_dir, v2_fname)

    if not os.path.isfile(v2_path):
        logger.warning(f"{label:<12s} | {day_str} | v2 file not found — skipped")
        report[label] = 'missing'
    else:
        df = pd.read_excel(v2_path, index_col=0, engine=XLSX_READ_ENGINE)
        df.index.name = 'timestamp'

        merged = _apply_quality_to_df(df, cfg_map, common_col_ord, logger)
        if merged.empty:
            logger.warning(f"{label:<12s} | {day_str} | Empty after processing — skipped")
            report[label] = 'empty'
        else:
            frozen = _write_with_frozen_check(
                merged, out_day_dir,
                f'common_{day_str}.xlsx',
                label, day_str, logger,
            )
            report[label] = 'frozen' if frozen else 'ok'

    return report


# =============================================================================
# PARALLEL WORKER (top-level for pickling)
# =============================================================================

def _process_file_task(task: dict) -> dict:
    """
    Process a single v2 xlsx file → v3 xlsx.  Designed to be called from a
    ProcessPoolExecutor.  All arguments are plain dicts/strings (picklable).

    *task* keys:
        v2_path, out_dir, fname, label, day_str,
        col_order, cfg_map
    Returns {label, day_str, status, message}.
    """
    import warnings
    warnings.filterwarnings('ignore')
    pd.options.mode.chained_assignment = None

    v2_path = task['v2_path']
    out_dir = task['out_dir']
    fname = task['fname']
    label = task['label']
    day_str = task['day_str']
    col_order = task['col_order']
    cfg_map = task['cfg_map']

    if not os.path.isfile(v2_path):
        return {'label': label, 'day_str': day_str, 'status': 'missing',
                'message': f'{label:<12s} | {day_str} | v2 file not found — skipped'}

    try:
        df = pd.read_excel(v2_path, index_col=0, engine=XLSX_READ_ENGINE)
        df.index.name = 'timestamp'

        # Minimal logger for _apply_quality_to_df (it only uses .debug)
        _log = logging.getLogger(f'worker.{label}.{day_str}')
        _log.setLevel(logging.WARNING)

        merged = _apply_quality_to_df(df, cfg_map, col_order, _log)
        if merged.empty:
            return {'label': label, 'day_str': day_str, 'status': 'empty',
                    'message': f'{label:<12s} | {day_str} | Empty — skipped'}

        frozen = is_all_frozen(merged)
        # Directory is pre-created by the main process; no makedirs here.
        out_fname = f'[FROZEN]{fname}' if frozen else fname
        # Remove stale counterpart (frozen ↔ unfrozen) from previous runs
        stale = f'{fname}' if frozen else f'[FROZEN]{fname}'
        stale_path = os.path.join(out_dir, stale)
        if os.path.isfile(stale_path):
            os.unlink(stale_path)
        _fast_write(merged, os.path.join(out_dir, out_fname))

        status = 'frozen' if frozen else 'ok'
        suffix = ' [FROZEN]' if frozen else ''
        msg = (f'{label:<12s} | {day_str} | OK  '
               f'({merged.shape[0]} rows, {merged.shape[1]} cols){suffix}')
        return {'label': label, 'day_str': day_str, 'status': status,
                'message': msg}
    except Exception as exc:
        return {'label': label, 'day_str': day_str, 'status': 'error',
                'message': f'{label:<12s} | {day_str} | ERROR: {exc}'}


def _build_file_tasks(
    dates: list[datetime],
    pump_sensors: dict[int, list[dict]],
    common_sensors: list[dict],
    pump_col_order: list[str],
    common_col_ord: list[str],
    input_dir: str,
    output_dir: str,
    cfg_map: dict[str, dict],
    selected_pumps: set[int] | None = None,
    out_ext: str = '.xlsx',
) -> list[dict]:
    """Build a flat list of file-level tasks for parallel execution."""
    tasks: list[dict] = []
    for date in dates:
        day_str = date.strftime('%Y-%m-%d')
        in_day = os.path.join(
            input_dir, date.strftime('%Y'), date.strftime('%m'), date.strftime('%d'),
        )
        out_day = os.path.join(
            output_dir, date.strftime('%Y'), date.strftime('%m'), date.strftime('%d'),
        )
        for pump_num in sorted(pump_sensors):
            if selected_pumps and pump_num not in selected_pumps:
                continue
            tasks.append({
                'v2_path': os.path.join(in_day, f'fw_pump_{pump_num}_{day_str}.xlsx'),
                'out_dir': out_day,
                'fname': f'fw_pump_{pump_num}_{day_str}{out_ext}',
                'label': f'FW Pump {pump_num}',
                'day_str': day_str,
                'col_order': pump_col_order,
                'cfg_map': cfg_map,
            })
        tasks.append({
            'v2_path': os.path.join(in_day, f'common_{day_str}.xlsx'),
            'out_dir': out_day,
            'fname': f'common_{day_str}{out_ext}',
            'label': 'Common',
            'day_str': day_str,
            'col_order': common_col_ord,
            'cfg_map': cfg_map,
        })
    return tasks


# =============================================================================
# CLI
# =============================================================================

@click.command()
@click.option(
    '--start', '-s', required=True,
    type=click.DateTime(formats=['%Y-%m-%d']),
    help='Start date (inclusive), YYYY-MM-DD',
)
@click.option(
    '--end', '-e', required=True,
    type=click.DateTime(formats=['%Y-%m-%d']),
    help='End date (inclusive), YYYY-MM-DD',
)
@click.option(
    '--output', '-o', default=None, type=click.Path(file_okay=False),
    help=f'Output base directory (default: {OUTPUT_BASE_DIR})',
)
@click.option(
    '--pumps', '-p', default=None,
    help='Comma-separated pump numbers to process (e.g. "1,3"). Default: all.',
)
@click.option(
    '--source', '-src',
    type=click.Choice(['files', 'local', 'db', 'auto'], case_sensitive=False),
    default='auto',
    help='Data source: files (read v2 xlsx, no DB), local (DB + Excel config), '
         'db (DB + kks_description config), auto (→files). Default: auto.',
)
@click.option(
    '--input-dir', '-i', default=None, type=click.Path(file_okay=False),
    help=f'Input directory with v2 xlsx files (only for --source files). '
         f'Default: {V2_INPUT_DIR}',
)
@click.option(
    '--workers', '-w', default=None, type=int,
    help='Number of parallel workers for --source files.  '
         'Default: number of CPU cores.  Use 1 to disable parallelism.',
)
@click.option(
    '--format', '-f', 'out_format',
    type=click.Choice(['xlsx', 'csv'], case_sensitive=False),
    default='xlsx',
    help='Output format.  csv is ~10x faster to write than xlsx.  Default: xlsx.',
)
def main(
    start: datetime,
    end: datetime,
    output: str | None,
    pumps: str | None,
    source: str,
    input_dir: str | None,
    workers: int | None,
    out_format: str,
):
    """
    Download FW pump raw data day by day into Excel files with quality columns.

    Produces four .xlsx per day (pump 1, pump 2, pump 3, common).
    Each sensor column gets a companion <sensor>_quality column.

    \b
    Quality codes:
        0 = OK
        1 = Missing data
        2 = Null value
        3 = Out of range
        4 = Fall down
        5 = Frozen data

    \b
    Examples:
        python extract_fw_pump_data_v3.py -s 2023-01-01 -e 2026-03-24 --source files
        python extract_fw_pump_data_v3.py -s 2026-01-01 -e 2026-01-31
        python extract_fw_pump_data_v3.py -s 2026-02-01 -e 2026-02-15 -p 1,3
        python extract_fw_pump_data_v3.py -s 2026-03-01 -e 2026-03-01 --source db
    """
    out_dir = output or OUTPUT_BASE_DIR
    logger = setup_logger(out_dir)

    # ---- Resolve config source -----------------------------------------
    if source == 'auto':
        source = 'files'
    logger.info(f'Config source: {source}')

    use_files = (source == 'files')

    # ---- Load definitions ----------------------------------------------
    if not os.path.isfile(EXCEL_FILE):
        logger.error(f"Excel not found: {EXCEL_FILE}")
        sys.exit(1)

    if source == 'db':
        _ensure_raw_models()
        pump_sensors, common_sensors = load_definitions_db(EXCEL_FILE)
    else:
        pump_sensors, common_sensors = load_definitions_local(EXCEL_FILE)

    # Apply fallback defaults for sensors still missing config
    for pn in pump_sensors:
        pump_sensors[pn] = apply_sensor_defaults(pump_sensors[pn])
    common_sensors = apply_sensor_defaults(common_sensors)

    pump_col_order = canonical_column_order(pump_sensors)
    common_col_ord = common_column_order(common_sensors)

    selected_pumps: set[int] | None = None
    if pumps:
        selected_pumps = {int(p.strip()) for p in pumps.split(',')}

    in_dir = input_dir or V2_INPUT_DIR

    # ---- Header --------------------------------------------------------
    logger.info('=' * 70)
    logger.info('FW PUMP DATA EXTRACTION v3 — START')
    logger.info(f'  Period : {start.strftime("%Y-%m-%d")} → {end.strftime("%Y-%m-%d")}')
    logger.info(f'  Source : {source}')
    if use_files:
        logger.info(f'  Input  : {os.path.abspath(in_dir)}')
    logger.info(f'  Pumps  : {sorted(selected_pumps) if selected_pumps else [1, 2, 3]}')
    logger.info(f'  Output : {os.path.abspath(out_dir)}')
    for pn in sorted(pump_sensors):
        logger.info(f'  Pump {pn} : {len(pump_sensors[pn])} sensors')
    logger.info(f'  Common : {len(common_sensors)} sensors + ambient temperature')
    logger.info(f'  Pump col order ({len(pump_col_order)} cols): {pump_col_order}')
    logger.info(f'  Common col order ({len(common_col_ord)} cols): {common_col_ord}')
    logger.info('=' * 70)

    # ---- Day-by-day processing -----------------------------------------
    cfg_map = _build_sensor_config_map(pump_sensors, common_sensors) if use_files else None

    # Build date list
    dates: list[datetime] = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    total_days = len(dates)

    if use_files:
        # ---- PARALLEL file-level processing ----------------------------
        out_ext = '.csv' if out_format == 'csv' else '.xlsx'
        n_workers = workers if workers is not None else min(
            multiprocessing.cpu_count(), total_days * 4,
        )
        n_workers = max(1, n_workers)

        tasks = _build_file_tasks(
            dates, pump_sensors, common_sensors,
            pump_col_order, common_col_ord,
            in_dir, out_dir, cfg_map,
            selected_pumps,
            out_ext=out_ext,
        )

        # Pre-create ALL output directories in the main process so that
        # workers never race on os.makedirs for the same day directory.
        out_dirs = sorted({t['out_dir'] for t in tasks})
        for d in out_dirs:
            os.makedirs(d, exist_ok=True)

        logger.info(f'  Workers: {n_workers}  |  Files: {len(tasks)}  |  Format: {out_format}')

        missing_files = 0
        errors = 0
        frozen_files = 0
        processed = 0
        t_start = time.perf_counter()

        if n_workers == 1:
            # Sequential — keeps log order deterministic
            for task in tasks:
                result = _process_file_task(task)
                processed += 1
                if result['status'] == 'missing':
                    missing_files += 1
                    logger.warning(result['message'])
                elif result['status'] == 'error':
                    errors += 1
                    logger.error(result['message'])
                else:
                    if result['status'] == 'frozen':
                        frozen_files += 1
                        logger.warning(result['message'])
                    else:
                        logger.info(result['message'])
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_process_file_task, t): t for t in tasks}
                for future in as_completed(futures):
                    processed += 1
                    try:
                        result = future.result()
                    except Exception as exc:
                        errors += 1
                        task_info = futures[future]
                        logger.error(
                            f"{task_info['label']:<12s} | {task_info['day_str']} "
                            f"| WORKER CRASH: {exc}"
                        )
                        continue

                    if result['status'] == 'missing':
                        missing_files += 1
                    elif result['status'] == 'error':
                        errors += 1
                        logger.error(result['message'])
                    elif result['status'] == 'frozen':
                        frozen_files += 1

                    # Progress every 200 files
                    if processed % 200 == 0 or processed == len(tasks):
                        elapsed = time.perf_counter() - t_start
                        rate = processed / elapsed if elapsed > 0 else 0
                        eta = (len(tasks) - processed) / rate if rate > 0 else 0
                        logger.info(
                            f'  Progress: {processed}/{len(tasks)} files  '
                            f'({rate:.1f} files/s, ETA {eta:.0f}s)'
                        )

        elapsed = time.perf_counter() - t_start
        days_with_gaps = missing_files  # approximate (file-level)

    else:
        # ---- Sequential DB processing ----------------------------------
        days_with_gaps = 0
        t_start = time.perf_counter()
        for date in dates:
            _ensure_raw_models()
            report = extract_day(
                date, pump_sensors, common_sensors,
                pump_col_order, common_col_ord,
                out_dir, logger, selected_pumps,
            )
            if report:
                days_with_gaps += 1

        elapsed = time.perf_counter() - t_start
        missing_files = 0
        errors = 0
        frozen_files = 0
        processed = total_days * 4

    # ---- Summary -------------------------------------------------------
    logger.info('=' * 70)
    logger.info('FW PUMP DATA EXTRACTION v3 — FINISHED')
    logger.info(f'  Days processed       : {total_days}')
    logger.info(f'  Files processed      : {processed}')
    logger.info(f'  Missing files        : {missing_files}')
    logger.info(f'  Frozen files         : {frozen_files}')
    logger.info(f'  Errors               : {errors}')
    logger.info(f'  Elapsed              : {elapsed:.1f}s ({elapsed/60:.1f}min)')
    logger.info(f'  Rate                 : {processed/elapsed:.1f} files/s' if elapsed > 0 else '')
    logger.info(
        f'  Log file             : '
        f'{os.path.join(out_dir, "fw_pump_extraction_v3.log")}'
    )
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
