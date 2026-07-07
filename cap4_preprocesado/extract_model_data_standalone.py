"""
HTF pump data extraction script for model training (standalone).

This script extracts ALL available pump sensor data from two PostgreSQL
databases (plant_db for pump measurements, plant_db_raw for ambient temperature)
and saves it to CSV files compatible with the fault detection model.

Requires a .env file with database credentials.

Usage:
    python extract_model_data_standalone.py
    python extract_model_data_standalone.py -o data/train/
    python extract_model_data_standalone.py --list-pumps
"""

import os

import click
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from rich import print
from rich.console import Console

# Load .env from the same directory as this script
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

_DB_USER = os.environ['PLANT_DB_USER']
_DB_PASSWORD = os.environ['PLANT_DB_PASSWORD']
_DB_HOST = os.environ['PLANT_DB_HOST']
_DB_PORT = os.environ.get('PLANT_DB_PORT', '5432')

PLANT_DB_URL = f"postgresql://{_DB_USER}:{_DB_PASSWORD}@{_DB_HOST}:{_DB_PORT}/plant_db"
SHAMS_RAW_DB_URL = f"postgresql://{_DB_USER}:{_DB_PASSWORD}@{_DB_HOST}:{_DB_PORT}/plant_db_raw"

# KKS table for ambient temperature
AMBIENT_TEMPERATURE_TABLE = "t_r1cf_01ct007:av"

# Resampling interval
RESAMPLE_INTERVAL = "5min"


# =============================================================================
# PUMP / SENSOR CONFIGURATION
# =============================================================================

# Mapping of pump_id (database ID) to pump number in sensor names
# Pump "12" (ID: 1) -> sensors with "Main HTF Pump 1 ..."
# Pump "14" (ID: 2) -> sensors with "Main HTF Pump 2 ..."
# Pump "16" (ID: 3) -> sensors with "Main HTF Pump 3 ..."
# Pump "18" (ID: 4) -> sensors with "Main HTF Pump 4 ..."
PUMP_ID_TO_SENSOR_NUMBER = {
    1: 1,   # Pump "12" (Main pump)
    2: 2,   # Pump "14" (Main pump)
    3: 3,   # Pump "16" (Main pump)
    4: 4,   # Pump "18" (Main pump)
}

# Required sensors (with X as placeholder for pump number)
# The generic name is what will appear in the output CSV
SENSOR_TEMPLATES = [
    ("Main HTF Pump X Speed", "Main HTF Pump Speed"),
    ("Main HTF Pump X Inlet Temperature", "Main HTF Pump Inlet Temperature"),
    ("Main HTF Pump X Current Consumption", "Main HTF Pump Current Consumption"),
    ("Main HTF Pump X Flow", "Main HTF Pump Flow"),
    ("Main HTF Pump X Outlet Pressure", "Main HTF Pump Outlet Pressure"),
    ("Main HTF Pump X NDE Outboard bearing", "Main HTF Pump NDE Outboard bearing"),
    ("Main HTF Pump X NDE Inboard bearing", "Main HTF Pump NDE Inboard bearing"),
    ("Main HTF Pump X DE bearing", "Main HTF Pump DE bearing"),
    ("Main HTF Pump X Motor bearing Temp 1", "Main HTF Pump Motor bearing Temp 1"),
    ("Main HTF Pump X Motor bearing Temp 2", "Main HTF Pump Motor bearing Temp 2"),
    ("Main HTF Pump X Motor U winding Temp 1", "Main HTF Pump Motor U winding Temp 1"),
    ("Main HTF Pump X Motor U winding Temp 2", "Main HTF Pump Motor U winding Temp 2"),
    ("Main HTF Pump X Motor U winding Temp 3", "Main HTF Pump Motor U winding Temp 3"),
    ("Main HTF Pump X DE Side Bearing vibration", "Main HTF Pump DE Side Bearing vibration"),
    ("Main HTF Pump X NDE Side Bearing vibration", "Main HTF Pump NDE Side Bearing vibration"),
]

# Required output columns
REQUIRED_OUTPUT_COLUMNS = [
    "timestamp",
    "pump_id",
    "Ambient temperature",
    "Main HTF Pump Speed",
    "Main HTF Pump Inlet Temperature",
    "Main HTF Pump Current Consumption",
    "Main HTF Pump Flow",
    "Main HTF Pump Outlet Pressure",
    "Main HTF Pump NDE Outboard bearing",
    "Main HTF Pump NDE Inboard bearing",
    "Main HTF Pump DE bearing",
    "Main HTF Pump Motor bearing Temp 1",
    "Main HTF Pump Motor bearing Temp 2",
    "Main HTF Pump Motor U winding Temp 1",
    "Main HTF Pump Motor U winding Temp 2",
    "Main HTF Pump Motor U winding Temp 3",
    "Main HTF Pump DE Side Bearing vibration",
    "Main HTF Pump NDE Side Bearing vibration",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_pump_id_by_name(session: Session, pump_name: str) -> int | None:
    """Gets the pump ID by its name (e.g., "Pump 12" or just "12")."""
    # Extract the numeric part if the user passes "Pump 12" style names
    import re
    match = re.search(r'\d+', pump_name)
    number = match.group() if match else pump_name

    query = text(
        "SELECT id, name FROM pumps_pump WHERE name = :name AND type_id = 1"
    )
    result = session.execute(query, {'name': number}).fetchone()
    if result:
        return result[0]
    return None


def get_sensor_number_for_pump(pump_id: int) -> int | None:
    """Gets the sensor number (1-4) corresponding to the pump_id."""
    return PUMP_ID_TO_SENSOR_NUMBER.get(pump_id)


def build_sensor_mapping(pump_id: int) -> dict[str, str]:
    """Builds the mapping from specific sensor names to generic names."""
    sensor_number = get_sensor_number_for_pump(pump_id)
    if sensor_number is None:
        raise ValueError(f"No sensor mapping for pump_id {pump_id}")

    mapping = {}
    for template, generic_name in SENSOR_TEMPLATES:
        specific_name = template.replace("X", str(sensor_number))
        mapping[specific_name] = generic_name
    return mapping


def get_ambient_temperature(start_datetime: datetime, end_datetime: datetime) -> pd.DataFrame:
    """
    Gets the ambient temperature from the plant_db_raw database.

    The raw table stores one row per day with an array of values sampled
    every ``timestep`` seconds. This function expands the array into a
    time-indexed DataFrame resampled to 5-minute resolution.

    Processes data in monthly chunks to avoid excessive memory usage.
    """
    engine_raw = create_engine(SHAMS_RAW_DB_URL)
    console = Console()

    try:
        all_chunks = []
        # Process in monthly chunks
        chunk_start = datetime(start_datetime.year, start_datetime.month, 1)
        final_end = end_datetime

        while chunk_start < final_end:
            # End of this month
            if chunk_start.month == 12:
                chunk_end = datetime(chunk_start.year + 1, 1, 1)
            else:
                chunk_end = datetime(chunk_start.year, chunk_start.month + 1, 1)
            chunk_end = min(chunk_end, final_end)

            with Session(engine_raw) as session:
                query = text(f"""
                    SELECT date, timestep, value
                    FROM "{AMBIENT_TEMPERATURE_TABLE}"
                    WHERE date >= :start_date AND date <= :end_date
                    ORDER BY date
                """)
                result = session.execute(query, {
                    'start_date': chunk_start,
                    'end_date': chunk_end,
                }).fetchall()

            for row_date, timestep, values in result:
                if not values:
                    continue
                base = pd.Timestamp(row_date)
                timestamps = [base + pd.Timedelta(seconds=i * timestep) for i in range(len(values))]
                day_df = pd.DataFrame({
                    'timestamp': timestamps,
                    'Ambient temperature': values,
                })
                # Resample each day immediately to save memory
                day_df = day_df.set_index('timestamp').resample(RESAMPLE_INTERVAL).mean().dropna()
                all_chunks.append(day_df)

            chunk_start = chunk_end

        if not all_chunks:
            return pd.DataFrame(columns=['Ambient temperature'])

        temp_df = pd.concat(all_chunks)
        temp_df = temp_df[~temp_df.index.duplicated(keep='first')]

        # Filter to exact time range
        start_filter = pd.Timestamp(start_datetime)
        end_filter = pd.Timestamp(end_datetime)
        temp_df = temp_df[(temp_df.index >= start_filter) & (temp_df.index < end_filter)]

        return temp_df

    except Exception as e:
        print(f"[yellow]⚠ No ambient temperature data available: {e}[/yellow]")
        return pd.DataFrame(columns=['Ambient temperature'])
    finally:
        engine_raw.dispose()


def extract_pump_data(
    session: Session,
    pump_id: int,
    pump_name: str,
    start_datetime: datetime,
    end_datetime: datetime,
    output_dir: str = ".",
) -> bool:
    """Extracts data from a specific pump and saves it to CSV."""
    console = Console()

    # Build sensor mapping for this pump
    try:
        sensor_mapping = build_sensor_mapping(pump_id)
    except ValueError as e:
        print(f"[red]✗ Error: {e}[/red]")
        return False

    # Get sensor IDs for this pump
    sensor_query = text(
        "SELECT id, description FROM pumps_sensor WHERE pump_id = :pump_id"
    )
    sensor_result = session.execute(sensor_query, {'pump_id': pump_id}).fetchall()
    sensor_ids = [r[0] for r in sensor_result]
    sensor_descriptions = {r[0]: r[1] for r in sensor_result}

    if not sensor_ids:
        print(f"[yellow]⚠ No sensors found for pump {pump_name} (ID: {pump_id})[/yellow]")
        return False

    # Find the Speed sensor ID to filter on pump running
    speed_sensor_id = None
    sensor_number = get_sensor_number_for_pump(pump_id)
    speed_sensor_name = f"Main HTF Pump {sensor_number} Speed"
    for sid, desc in sensor_descriptions.items():
        if desc == speed_sensor_name:
            speed_sensor_id = sid
            break

    # Build dynamic CASE statements for pivoting
    case_statements = []
    for sid, desc in sensor_descriptions.items():
        generic_name = sensor_mapping.get(desc, desc)
        safe_name = generic_name.replace("'", "''")
        case_statements.append(
            f'AVG(CASE WHEN prm.sensor_id = {sid} THEN prm.value END) AS "{safe_name}"'
        )
    cases_sql = ",\n        ".join(case_statements)

    # Truncate to 5-minute buckets for aggregation
    trunc_expr = "date_trunc('hour', prm.timestamp) + INTERVAL '5 min' * FLOOR(EXTRACT(MINUTE FROM prm.timestamp) / 5)"

    # Build query with Speed > 0 filter
    if speed_sensor_id:
        query = f"""
        WITH speed_filtered AS (
            SELECT DISTINCT {trunc_expr} as ts_bucket
            FROM pumps_raw_measurement prm
            WHERE sensor_id = :speed_sensor_id
              AND timestamp >= :start_date
              AND timestamp < :end_date
              AND value > 0
        )
        SELECT
            {trunc_expr} as timestamp,
            {cases_sql}
        FROM pumps_raw_measurement prm
        INNER JOIN speed_filtered sf ON ({trunc_expr}) = sf.ts_bucket
        WHERE prm.sensor_id = ANY(:sensor_ids)
          AND prm.timestamp >= :start_date
          AND prm.timestamp < :end_date
        GROUP BY ({trunc_expr})
        ORDER BY timestamp ASC
        """
    else:
        query = f"""
        SELECT
            {trunc_expr} as timestamp,
            {cases_sql}
        FROM pumps_raw_measurement prm
        WHERE prm.sensor_id = ANY(:sensor_ids)
          AND prm.timestamp >= :start_date
          AND prm.timestamp < :end_date
        GROUP BY ({trunc_expr})
        ORDER BY timestamp ASC
        """

    try:
        console.print(f"[dim]Querying data for pump {pump_name}...[/dim]")

        params = {
            'start_date': start_datetime,
            'end_date': end_datetime,
            'sensor_ids': sensor_ids,
        }
        if speed_sensor_id:
            params['speed_sensor_id'] = speed_sensor_id

        result = session.execute(text(query), params)
        df_pivot = pd.DataFrame(result.fetchall(), columns=list(result.keys()))

        if df_pivot.empty:
            print(f"[yellow]⚠ No data for pump {pump_name} ({pump_id}) in the specified range[/yellow]")
            return False

        console.print(f"[dim]Retrieved {len(df_pivot)} records, processing...[/dim]")

        # Add pump_id
        df_pivot['pump_id'] = pump_id

        # Get ambient temperature from plant_db_raw
        console.print("[dim]Getting ambient temperature...[/dim]")
        temp_df = get_ambient_temperature(start_datetime, end_datetime)

        if not temp_df.empty:
            df_pivot['timestamp'] = pd.to_datetime(df_pivot['timestamp']).dt.tz_localize(None)
            temp_df.index = pd.to_datetime(temp_df.index).tz_localize(None)

            df_pivot = df_pivot.sort_values('timestamp')
            temp_df_reset = temp_df.reset_index()
            temp_df_reset.columns = ['timestamp', 'Ambient temperature']
            temp_df_reset = temp_df_reset.sort_values('timestamp')

            df_pivot = pd.merge_asof(
                df_pivot,
                temp_df_reset,
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('5min'),
            )
        else:
            df_pivot['Ambient temperature'] = np.nan

        # Check for missing columns
        available_columns = set(df_pivot.columns)
        required_columns = set(REQUIRED_OUTPUT_COLUMNS)
        missing = required_columns - available_columns
        if missing:
            print(f"[yellow]⚠ Missing columns for pump {pump_name}: {', '.join(missing)}[/yellow]")

        # Select only required columns that exist
        output_columns = [col for col in REQUIRED_OUTPUT_COLUMNS if col in df_pivot.columns]
        df_output = df_pivot[output_columns]

        # Generate filename
        filename = f"pump_{pump_id}_all_data.csv"
        filepath = os.path.join(output_dir, filename)

        df_output.to_csv(filepath, index=False)
        print(f"[green]✓ Data saved: {filepath} ({len(df_output)} records)[/green]")
        return True

    except Exception as e:
        print(f"[red]✗ Error extracting data for pump {pump_name}: {e}[/red]")
        return False


# =============================================================================
# CLI
# =============================================================================

# All 4 main HTF pumps
ALL_MAIN_PUMPS = ["Pump 12", "Pump 14", "Pump 16", "Pump 18"]


def get_data_time_range(session: Session) -> tuple[datetime, datetime]:
    """Gets the full available time range from pumps_raw_measurement."""
    query = text(
        "SELECT MIN(timestamp), MAX(timestamp) FROM pumps_raw_measurement"
    )
    result = session.execute(query).fetchone()
    start, end = result
    # Strip timezone info if present
    if hasattr(start, 'tzinfo') and start.tzinfo is not None:
        start = start.replace(tzinfo=None)
    if hasattr(end, 'tzinfo') and end.tzinfo is not None:
        end = end.replace(tzinfo=None)
    return start, end


@click.command()
@click.option(
    '--output', '-o',
    default='.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    help='Output directory for CSV files (default: current directory)',
)
@click.option(
    '--list-pumps', '-l',
    is_flag=True,
    help='List all available pumps and exit',
)
@click.option(
    '--start', '-s',
    type=click.DateTime(formats=['%Y-%m-%d']),
    default=None,
    help='Optional start date in format YYYY-MM-DD (default: auto-detected)',
)
@click.option(
    '--end', '-e',
    type=click.DateTime(formats=['%Y-%m-%d']),
    default=None,
    help='Optional end date in format YYYY-MM-DD (default: auto-detected)',
)
def main(output: str, list_pumps: bool, start: datetime | None, end: datetime | None):
    """
    Extracts ALL available HTF pump data (all 4 main pumps) from the
    database and saves it to CSV files, resampled every 5 minutes.

    Requires a .env file with PLANT_DB_USER, PLANT_DB_PASSWORD, PLANT_DB_HOST,
    and optionally PLANT_DB_PORT.

    Examples:\n
        python extract_model_data_standalone.py\n
        python extract_model_data_standalone.py -o data/train/\n
        python extract_model_data_standalone.py --list-pumps
    """
    console = Console()
    engine = create_engine(PLANT_DB_URL)

    with Session(engine) as session:

        if list_pumps:
            query = text("SELECT id, name, description FROM pumps_pump ORDER BY id")
            results = session.execute(query).fetchall()
            console.print("\n[bold]Available pumps:[/bold]\n")
            for pump_id, name, description in results:
                desc = description or "No description"
                console.print(f"  • [cyan]{name}[/cyan] (ID: {pump_id}) - {desc}")
            console.print()
            return

        if output != '.' and not os.path.exists(output):
            os.makedirs(output)
            print(f"[blue]ℹ Directory created: {output}[/blue]")

        # Auto-detect full time range
        console.print("[dim]Detecting available data range...[/dim]")
        auto_start, auto_end = get_data_time_range(session)
        start_source = "auto-detected"
        end_source = "auto-detected"

        cli_start = start
        cli_end = end

        if cli_start is not None:
            start = cli_start.replace(tzinfo=None)
            start_source = "user-specified"
        else:
            start = auto_start

        if cli_end is not None:
            end = cli_end.replace(tzinfo=None)
            end_source = "user-specified"
        else:
            end = auto_end
        pumps = ALL_MAIN_PUMPS

        console.print(f"\n[bold]HTF Pump Data Extraction (5-min resample)[/bold]")
        console.print(f"  Period: {start} ({start_source}) → {end} ({end_source})")
        console.print(f"  Pumps: {', '.join(pumps)}")
        console.print(f"  Output: {os.path.abspath(output)}\n")

        success_count = 0
        for pump_name in pumps:
            pump_id = get_pump_id_by_name(session, pump_name)

            if pump_id is None:
                print(f"[red]✗ Pump '{pump_name}' not found in database[/red]")
                continue

            if pump_id not in PUMP_ID_TO_SENSOR_NUMBER:
                print(f"[red]✗ Pump '{pump_name}' (ID: {pump_id}) has no sensor mapping configured[/red]")
                print(f"  [dim]Valid IDs: {list(PUMP_ID_TO_SENSOR_NUMBER.keys())}[/dim]")
                continue

            if extract_pump_data(session, pump_id, pump_name, start, end, output):
                success_count += 1

        console.print(f"\n[bold]Summary:[/bold] {success_count}/{len(pumps)} extractions completed\n")

    engine.dispose()


if __name__ == '__main__':
    main()
