from __future__ import annotations

from pathlib import Path

import click
import pandas as pd


CLEAN_COLUMN_NAMES = [
    "timestamp",
    "ambient_temp",
    "speed",
    "inlet_temp",
    "motor_current",
    "flow",
    "outlet_pressure",
    "nde_outboard_temp",
    "nde_inboard_temp",
    "de_bearing_temp",
    "motor_bearing_temp_1",
    "motor_bearing_temp_2",
    "motor_u_winding_temp_1",
    "motor_u_winding_temp_2",
    "motor_u_winding_temp_3",
    "de_vibration",
    "nde_vibration",
]


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=Path("demo_data/raw"),
    show_default=True,
    help="Directory containing pump_X_all_data.csv files.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=Path("demo_data/daily"),
    show_default=True,
    help="Directory where per-day files will be written.",
)
def main(input_dir: Path, output_dir: Path) -> None:
    """Split monolithic pump CSV files into per-day CSV files."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists():
        raise click.ClickException(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(input_dir.glob("pump_*_all_data.csv"))
    if not source_files:
        click.echo(f"No files matching 'pump_*_all_data.csv' found in {input_dir}")
        return

    total_created = 0
    total_days = 0

    for source_file in source_files:
        parts = source_file.stem.split("_")
        if len(parts) < 4 or parts[0] != "pump" or parts[2] != "all" or parts[3] != "data":
            click.echo(f"Skipping unrecognized file name: {source_file.name}")
            continue

        pump_id = parts[1]

        try:
            df = pd.read_csv(source_file, parse_dates=["timestamp"])
        except ValueError as exc:
            click.echo(f"Skipping {source_file.name}: could not parse 'timestamp' column ({exc})")
            continue

        if df.empty:
            click.echo(f"Skipping {source_file.name}: file has 0 rows")
            continue

        if df.shape[1] < 18:
            click.echo(
                f"Skipping {source_file.name}: expected at least 18 columns, found {df.shape[1]}"
            )
            continue

        # Match downstream schema: take first 18 columns, drop pump_id (index 1), rename by position.
        df = df.iloc[:, :18].copy()
        df = df.drop(columns=df.columns[1])
        df.columns = CLEAN_COLUMN_NAMES

        day_count = 0
        for day, day_df in df.groupby(df["timestamp"].dt.date):
            if day_df.empty:
                continue

            output_name = f"pump_{pump_id}_{day:%Y-%m-%d}.csv"
            output_path = output_dir / output_name
            day_df.to_csv(output_path, index=False)
            day_count += 1
            total_created += 1

        total_days += day_count
        click.echo(f"{source_file.name}: created {day_count} files")

    click.echo("\nSummary")
    click.echo(f"Input directory: {input_dir}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Source files processed: {len(source_files)}")
    click.echo(f"Daily files created: {total_created}")
    click.echo(f"Distinct days exported: {total_days}")


if __name__ == "__main__":
    main()
