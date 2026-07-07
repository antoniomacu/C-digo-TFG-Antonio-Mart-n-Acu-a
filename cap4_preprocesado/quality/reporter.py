from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from quality.quality_config import QualityConfig
from quality.labeler import (
    CODE_NAMES,
    Q_FALL_DOWN,
    Q_FROZEN,
    Q_MISSING,
    Q_NULL,
    Q_OK,
    Q_OUT_OF_RANGE,
)


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_code(code: object) -> int | None:
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _build_aggregates(all_stats: list[dict]) -> dict[str, object]:
    code_order = [Q_MISSING, Q_NULL, Q_OUT_OF_RANGE, Q_FALL_DOWN, Q_FROZEN]
    code_totals: dict[int, int] = {code: 0 for code in code_order}
    sensor_code_totals: dict[str, dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    total_files = len(all_stats)
    files_written = 0
    skipped_frozen = 0
    skipped_no_op = 0
    skipped_other = 0
    sentinel_total = 0

    pump_files: dict[str, int] = defaultdict(int)
    pump_input_rows: dict[str, int] = defaultdict(int)
    pump_output_rows: dict[str, int] = defaultdict(int)

    for stats in all_stats:
        if bool(stats.get("written", False)):
            files_written += 1

        if bool(stats.get("skipped", False)):
            reason = str(stats.get("skip_reason", "")).strip().lower()
            if reason == "all-frozen":
                skipped_frozen += 1
            elif reason == "no operational rows":
                skipped_no_op += 1
            else:
                skipped_other += 1

        sentinel_total += _to_int(stats.get("sentinel_rows", 0))

        pump_id = str(stats.get("pump_id", "")).strip()
        if pump_id:
            pump_files[pump_id] += 1
            pump_input_rows[pump_id] += _to_int(stats.get("input_rows", 0))
            pump_output_rows[pump_id] += _to_int(stats.get("output_rows", 0))

        quality_counts = stats.get("quality_counts", {})
        if not isinstance(quality_counts, dict):
            continue

        for sensor_name, counts in quality_counts.items():
            if not isinstance(counts, dict):
                continue

            for raw_code, raw_count in counts.items():
                code = _normalize_code(raw_code)
                if code is None or code == Q_OK:
                    continue

                count = _to_int(raw_count)
                if code in code_totals:
                    code_totals[code] += count
                sensor_code_totals[str(sensor_name)][code] += count

    sensor_totals: dict[str, int] = {}
    for sensor_name in sorted(sensor_code_totals):
        sensor_totals[sensor_name] = sum(sensor_code_totals[sensor_name].values())

    return {
        "code_order": code_order,
        "code_totals": code_totals,
        "sensor_code_totals": sensor_code_totals,
        "sensor_totals": sensor_totals,
        "total_files": total_files,
        "files_written": files_written,
        "skipped_frozen": skipped_frozen,
        "skipped_no_op": skipped_no_op,
        "skipped_other": skipped_other,
        "sentinel_total": sentinel_total,
        "pump_files": pump_files,
        "pump_input_rows": pump_input_rows,
        "pump_output_rows": pump_output_rows,
    }


def generate_text_summary(
    all_stats: list[dict],
    system_name: str,
    output_path: Path,
    dry_run: bool = False,
) -> None:
    """Generate and print a text summary for quality pipeline execution."""

    aggregates = _build_aggregates(all_stats)
    skipped_total = (
        aggregates["skipped_frozen"]
        + aggregates["skipped_no_op"]
        + aggregates["skipped_other"]
    )

    lines: list[str] = []
    lines.append(f"{system_name} Quality Pipeline")
    lines.append("=" * len(lines[-1]))
    lines.append("")
    lines.append("Summary")
    lines.append("=======")
    lines.append(f"Files processed:  {aggregates['total_files']}")
    lines.append(f"Files written:    {aggregates['files_written']}")

    skip_text = (
        f"{skipped_total} ({aggregates['skipped_frozen']} all-frozen, "
        f"{aggregates['skipped_no_op']} no operational rows"
    )
    if aggregates["skipped_other"]:
        skip_text += f", {aggregates['skipped_other']} other"
    skip_text += ")"
    lines.append(f"Files skipped:    {skip_text}")
    lines.append("")

    lines.append("Quality issues detected (cells flagged):")
    for code in aggregates["code_order"]:
        code_name = CODE_NAMES.get(code, f"Code {code}")
        lines.append(f"  Code {code} ({code_name}): {aggregates['code_totals'][code]:,}")
    lines.append(f"  Sentinel rows: {aggregates['sentinel_total']:,}")
    lines.append("")

    lines.append("Per-sensor breakdown:")
    if not aggregates["sensor_totals"]:
        lines.append("  No quality columns were computed.")
    else:
        for sensor_name in sorted(aggregates["sensor_totals"]):
            lines.append(
                f"  {sensor_name}: {aggregates['sensor_totals'][sensor_name]:,} flagged"
            )
    lines.append("")

    lines.append("Per-pump totals:")
    if not aggregates["pump_files"]:
        lines.append("  No pump totals available.")
    else:
        for pump_id in sorted(aggregates["pump_files"], key=str):
            lines.append(
                f"  Pump {pump_id}: files={aggregates['pump_files'][pump_id]}, "
                f"input_rows={aggregates['pump_input_rows'][pump_id]:,}, "
                f"output_rows={aggregates['pump_output_rows'][pump_id]:,}"
            )

    summary_text = "\n".join(lines)
    print()
    print(summary_text)

    if dry_run:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary_text + "\n", encoding="utf-8")
    print(f"\nSummary saved to: {output_path}")


def generate_html_dashboard(
    all_stats: list[dict],
    quality_config: QualityConfig,
    system_name: str,
    output_path: Path,
) -> None:
    """Generate an HTML dashboard with Plotly if available."""

    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed. Skipping HTML dashboard generation.")
        return

    aggregates = _build_aggregates(all_stats)

    sensors = sorted(aggregates["sensor_code_totals"].keys())
    code_order = aggregates["code_order"]
    code_labels = [CODE_NAMES.get(code, f"Code {code}") for code in code_order]

    bar_fig = go.Figure()
    if sensors:
        for code, code_label in zip(code_order, code_labels):
            bar_fig.add_trace(
                go.Bar(
                    name=f"Code {code} ({code_label})",
                    x=sensors,
                    y=[
                        aggregates["sensor_code_totals"][sensor].get(code, 0)
                        for sensor in sensors
                    ],
                )
            )
        bar_fig.update_layout(
            barmode="stack",
            title=f"{system_name}: Quality codes per sensor",
            xaxis_title="Sensor",
            yaxis_title="Flagged cells",
        )
    else:
        bar_fig.add_annotation(
            text="No quality counts available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 14},
        )
        bar_fig.update_layout(title=f"{system_name}: Quality codes per sensor")

    processed = aggregates["total_files"] - (
        aggregates["skipped_frozen"]
        + aggregates["skipped_no_op"]
        + aggregates["skipped_other"]
    )
    skipped = (
        aggregates["skipped_frozen"]
        + aggregates["skipped_no_op"]
        + aggregates["skipped_other"]
    )

    pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=["Processed", "Skipped"],
                values=[processed, skipped],
                hole=0.3,
            )
        ]
    )
    pie_fig.update_layout(title=f"{system_name}: Processed vs skipped files")

    pump_ids = sorted(aggregates["pump_files"], key=str)
    if not pump_ids:
        table_fig = go.Figure()
        table_fig.add_annotation(
            text="No pump statistics available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 14},
        )
        table_fig.update_layout(title=f"{system_name}: Per-pump statistics")
    else:
        table_fig = go.Figure(
            data=[
                go.Table(
                    header={
                        "values": ["Pump", "Files", "Input rows", "Output rows"],
                        "align": "left",
                    },
                    cells={
                        "values": [
                            [f"Pump {pump_id}" for pump_id in pump_ids],
                            [aggregates["pump_files"][pump_id] for pump_id in pump_ids],
                            [
                                aggregates["pump_input_rows"][pump_id]
                                for pump_id in pump_ids
                            ],
                            [
                                aggregates["pump_output_rows"][pump_id]
                                for pump_id in pump_ids
                            ],
                        ],
                        "align": "left",
                    },
                )
            ]
        )
        table_fig.update_layout(title=f"{system_name}: Per-pump statistics")

    # Touch quality_config to keep signature meaningful and avoid dead argument drift.
    configured_sensor_count = len(quality_config.sensors)

    bar_div = bar_fig.to_html(full_html=False, include_plotlyjs="cdn")
    pie_div = pie_fig.to_html(full_html=False, include_plotlyjs=False)
    table_div = table_fig.to_html(full_html=False, include_plotlyjs=False)

    html_output = (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        f"  <title>{system_name} Quality Dashboard</title>\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "  <style>\n"
        "    body { font-family: Arial, sans-serif; margin: 24px; }\n"
        "    h1 { margin-bottom: 8px; }\n"
        "    .meta { color: #555; margin-bottom: 20px; }\n"
        "    .section { margin-bottom: 28px; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"  <h1>{system_name} Quality Dashboard</h1>\n"
        f"  <div class=\"meta\">Configured sensors: {configured_sensor_count}</div>\n"
        f"  <div class=\"section\">{bar_div}</div>\n"
        f"  <div class=\"section\">{pie_div}</div>\n"
        f"  <div class=\"section\">{table_div}</div>\n"
        "</body>\n"
        "</html>\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_output, encoding="utf-8")
    print(f"HTML dashboard saved to: {output_path}")
