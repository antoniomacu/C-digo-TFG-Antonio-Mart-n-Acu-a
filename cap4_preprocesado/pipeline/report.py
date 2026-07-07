from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from .system_config import SystemConfig


DETECTOR_DESCRIPTIONS = {
    "pressure_off": {
        "summary": "Discharge pressure while pump is OFF deviates significantly from the baseline.",
        "physical_meaning": "When a pump is OFF, the discharge line should reflect system header pressure. A drop suggests valve leakage, header isolation, or a neighboring pump tripping. A rise could indicate check-valve backflow.",
    },
    "off_state_current": {
        "summary": "Motor current during OFF periods (speed ≈ 0) is abnormally high.",
        "physical_meaning": "In a healthy stopped pump, motor current should be near zero. Elevated OFF-state current may indicate energized auxiliary circuits (heaters, lube oil pumps), instrument faults, or current transformer drift.",
    },
    "vibration": {
        "summary": "One or more vibration proxy signals (mean or peak) exceed the baseline threshold.",
        "physical_meaning": "Excessive vibration typically signals mechanical issues: bearing degradation, shaft misalignment, impeller imbalance, or loose mounting. Both sustained mean elevation and transient spikes are checked.",
    },
    "pressure_off_variability": {
        "summary": "OFF-state discharge pressure shows excessive variability (coefficient of variation).",
        "physical_meaning": "Stable OFF-state pressure is expected when the header system is healthy. High variability suggests intermittent valve leaks, pressure transient events, or instrumentation noise that could mask real anomalies.",
    },
    "speed_stability": {
        "summary": "Pump speed is unstable or deviates significantly from the baseline during steady-state operation.",
        "physical_meaning": "Stable pump speed is fundamental for centrifugal pump operation. Instability may indicate VFD issues, load swings, or control system problems. Large mean deviation suggests the pump ran at an unusual setpoint.",
    },
    "current_speed_ratio": {
        "summary": "The ratio of motor current to pump speed deviates from the established baseline.",
        "physical_meaning": "The current-to-speed ratio reflects pump loading. A rising ratio suggests increased hydraulic load (clogged strainer, closing valve) or degrading motor efficiency. A dropping ratio may indicate cavitation or low-load operation.",
    },
    "current_anomaly": {
        "summary": "Motor current mean or variability during operation exceeds the baseline threshold.",
        "physical_meaning": "Abnormal current draw during operation signals changes in pump loading. High mean current suggests the pump is working harder (increased system resistance). High variability may indicate intermittent mechanical or hydraulic disturbances.",
    },
    "pressure_on": {
        "summary": "Discharge pressure during ON (running) periods is abnormally high.",
        "physical_meaning": "High discharge pressure while running may indicate a closing or partially blocked discharge valve, system over-pressurization, or sensor drift. It can also signal deadheading — a dangerous condition where the pump runs against a closed valve.",
    },
    "temp_elevated": {
        "summary": "One or more bearing or motor temperature signals exceed the baseline threshold during operation.",
        "physical_meaning": "Elevated temperatures in bearings or motor windings suggest inadequate lubrication, excessive loading, cooling system failure, or incipient bearing failure. Temperature trends are critical early indicators of mechanical degradation.",
    },
    "pressure_off_extended": {
      "summary": "OFF-state pressure shows a sustained day-wide deviation before pump startup, confirmed by rebound after operation.",
      "physical_meaning": "The standard pressure_off detector uses a narrow +-30-minute window around the period. This extended detector examines all OFF-state data from the beginning of the day (or the end of the previous period) through to the current period start. It fires when: (1) the extended OFF-state mean deviates by more than 2.5sigma from the baseline, (2) the absolute deviation exceeds 4.5 bar, and (3) the pressure rebounds toward normal after the pump operates. This triple-gate logic catches slow-developing pressure anomalies - such as gradual header depressurization overnight - that partially recover before the narrow window would see them.",
    },
    "pressure_response": {
      "summary": "Pressure does not respond correctly to pump startup - the expected OFF->ON pressure increase is missing or reversed.",
      "physical_meaning": "When a healthy pump starts, discharge pressure should increase. This detector compares the observed OFF->ON pressure change against a robust baseline, split by sampling resolution (high-res <= 30 s, low-res otherwise). It flags periods where the response is statistically deviated or physically negative (pressure drops when the pump runs). A negative response suggests suction-side issues, blocked flow path, or a pump that is mechanically running but not producing the expected hydraulic output.",
    },
    "ml_ensemble": {
      "summary": "Unanimous vote from three independent ML models (LOF, Isolation Forest, Mahalanobis) flags the period as a multivariate outlier.",
      "physical_meaning": "This detector complements the rule-based detectors by examining the joint distribution of 13 steady-state features (speed, current, current/speed ratio, pressure, 5 temperatures, and 2 proximitor aggregates). Three models - Local Outlier Factor, Isolation Forest, and Mahalanobis distance - are trained on the curated normal training set. The detector fires only when all three models unanimously flag the period as anomalous, ensuring high confidence. It catches multivariate distribution shifts that no single rule-based detector covers: subtle joint deviations across multiple sensor channels that individually remain within normal bounds.",
    },
    "outlet_pressure": {
      "summary": "Outlet (discharge) pressure during operation deviates significantly from the baseline.",
      "physical_meaning": "Abnormally high discharge pressure may indicate a closing or blocked downstream valve, system over-pressurization, or deadheading. Low discharge pressure suggests pump degradation, worn impeller, cavitation, or open bypass.",
    },
    "differential_pressure": {
      "summary": "Pump head instability (outlet minus inlet pressure variability) exceeds baseline.",
      "physical_meaning": "Unstable pump differential pressure indicates erratic hydraulic performance. This can result from cavitation, air entrainment, control valve hunting, or impeller damage causing flow instabilities.",
    },
    "filter_condition": {
      "summary": "Filter differential pressure is elevated above the baseline threshold.",
      "physical_meaning": "Rising differential pressure across the inlet strainer or filter indicates progressive clogging by debris. This is a critical predictive maintenance signal — a blocked filter reduces NPSH and can trigger cavitation, leading to pump damage.",
    },
    "flow_anomaly": {
      "summary": "Total feedwater flow rate or flow-to-speed ratio deviates from the baseline.",
      "physical_meaning": "Flow is the purpose of the pump. Low flow at normal speed signals cavitation, worn impeller, closed valve, or air entrainment. An abnormal flow/speed ratio indicates the pump is operating off its performance curve, suggesting degradation.",
    },
    "winding_temp_imbalance": {
      "summary": "Excessive temperature spread between the three motor winding phases (U, V, W).",
      "physical_meaning": "A healthy 3-phase motor has balanced winding temperatures. A large spread (>10°C) indicates power supply phase imbalance, degrading winding insulation, inter-turn shorts, or asymmetric cooling — all early indicators of motor winding failure.",
    },
}


def _base_reason(reason: str) -> str:
    """Extract base detector name: 'vibration:nde_thrust_prox_mean' -> 'vibration'"""
    return reason.split(":", 1)[0] if ":" in reason else reason


def _normalize_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _build_stats(df: pd.DataFrame) -> dict:
    """Compute total, normal, abnormal counts and date range."""
    total = len(df)
    normal = int((df["classification"] == "normal").sum()) if total else 0
    abnormal = int((df["classification"] == "abnormal").sum()) if total else 0

    if total:
        dates = sorted(_normalize_str(v) for v in df["date"].tolist() if _normalize_str(v))
        start_date = dates[0] if dates else "-"
        end_date = dates[-1] if dates else "-"
    else:
        start_date = "-"
        end_date = "-"

    per_pump: dict[str, dict[str, int]] = {}
    for pump_id, group in df.groupby("pump", sort=False):
        pump_key = _normalize_str(pump_id)
        per_pump[pump_key] = {
            "total": int(len(group)),
            "normal": int((group["classification"] == "normal").sum()),
            "abnormal": int((group["classification"] == "abnormal").sum()),
        }

    return {
        "total": total,
        "normal": normal,
        "abnormal": abnormal,
        "start_date": start_date,
        "end_date": end_date,
        "per_pump": per_pump,
    }


def _build_reason_frequency(df: pd.DataFrame) -> list[tuple[str, int]]:
    """Split semicolon-separated reasons, count base detectors, return sorted descending."""
    counts: dict[str, int] = {}

    abnormal_rows = df[df["classification"] == "abnormal"]
    for reasons_raw in abnormal_rows["reasons"].tolist():
        reasons_text = _normalize_str(reasons_raw)
        if not reasons_text:
            continue
        for part in reasons_text.split(";"):
            token = _normalize_str(part)
            if not token:
                continue
            base = _base_reason(token)
            counts[base] = counts.get(base, 0) + 1

    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _spread_indices(length: int, wanted: int) -> list[int]:
    if length <= 0 or wanted <= 0:
        return []
    if wanted >= length:
        return list(range(length))
    if wanted == 1:
        return [length // 2]

    indices: list[int] = []
    seen: set[int] = set()
    for i in range(wanted):
        idx = round(i * (length - 1) / (wanted - 1))
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    return indices


def _select_error_examples(df: pd.DataFrame, max_per_reason: int = 3) -> dict[str, list[dict]]:
    """Select representative abnormal examples per base reason.

    Strategy: prefer pump diversity, spread across date range.
    """
    abnormal_rows = df[df["classification"] == "abnormal"].copy()
    if abnormal_rows.empty:
        return {}

    abnormal_rows["date"] = abnormal_rows["date"].map(_normalize_str)
    abnormal_rows = abnormal_rows.sort_values("date")

    grouped: dict[str, list[dict]] = {}
    for _, row in abnormal_rows.iterrows():
        reasons_text = _normalize_str(row.get("reasons", ""))
        if not reasons_text:
            continue

        row_payload = {
            "date": _normalize_str(row.get("date", "")),
            "pump": _normalize_str(row.get("pump", "")),
            "period_start": _normalize_str(row.get("period_start", "")),
            "period_end": _normalize_str(row.get("period_end", "")),
            "duration_minutes": _normalize_str(row.get("duration_minutes", "")),
            "reasons": reasons_text,
        }

        unique_bases: set[str] = set()
        for part in reasons_text.split(";"):
            token = _normalize_str(part)
            if not token:
                continue
            unique_bases.add(_base_reason(token))

        for base in unique_bases:
            grouped.setdefault(base, []).append(row_payload)

    selected: dict[str, list[dict]] = {}
    for base_reason, candidates in grouped.items():
        if not candidates:
            selected[base_reason] = []
            continue

        picked: list[dict] = []
        picked_keys: set[tuple[str, str, str, str]] = set()
        seen_pumps: set[str] = set()

        for c in candidates:
            if len(picked) >= max_per_reason:
                break
            key = (c["date"], c["pump"], c["period_start"], c["period_end"])
            if c["pump"] in seen_pumps or key in picked_keys:
                continue
            picked.append(c)
            picked_keys.add(key)
            seen_pumps.add(c["pump"])

        if len(picked) < max_per_reason:
            for idx in _spread_indices(len(candidates), max_per_reason):
                c = candidates[idx]
                key = (c["date"], c["pump"], c["period_start"], c["period_end"])
                if key in picked_keys:
                    continue
                picked.append(c)
                picked_keys.add(key)
                if len(picked) >= max_per_reason:
                    break

        if len(picked) < max_per_reason:
            for c in candidates:
                key = (c["date"], c["pump"], c["period_start"], c["period_end"])
                if key in picked_keys:
                    continue
                picked.append(c)
                picked_keys.add(key)
                if len(picked) >= max_per_reason:
                    break

        selected[base_reason] = picked

    return selected


def _lookup_description(reason: str) -> dict[str, str]:
    """Return description dict for a reason, with fallback for custom/unknown detectors."""
    base = _base_reason(_normalize_str(reason))
    if base in DETECTOR_DESCRIPTIONS:
        return DETECTOR_DESCRIPTIONS[base]
    return {
        "summary": "Custom detector reason produced by the active pipeline configuration.",
        "physical_meaning": "This detector is not part of the built-in catalog. Review your custom detector implementation and thresholds for system-specific interpretation.",
    }


def _render_error_examples(
    reason_freq: list[tuple[str, int]],
    error_examples: dict[str, list[dict]],
) -> str:
    if not reason_freq:
        return '<div class="empty-msg">No abnormal periods detected.</div>'

    parts: list[str] = []
    for reason, count in reason_freq:
        desc = _lookup_description(reason)
        safe_reason = html.escape(reason)
        safe_summary = html.escape(desc["summary"])
        safe_physical = html.escape(desc["physical_meaning"])

        rows = error_examples.get(reason, [])
        if rows:
            body_rows = "".join(
                (
                    "<tr>"
                    f"<td>{html.escape(r['date'])}</td>"
                    f"<td>Pump {html.escape(r['pump'])}</td>"
                    f"<td>{html.escape(r['period_start'])}</td>"
                    f"<td>{html.escape(r['period_end'])}</td>"
                    f"<td>{html.escape(r['duration_minutes'])}</td>"
                    f"<td>{html.escape(r['reasons'])}</td>"
                    "</tr>"
                )
                for r in rows
            )
            table_html = (
                '<div class="examples-table-wrap">'
                '<table class="examples-table">'
                "<thead><tr><th>Date</th><th>Pump</th><th>Start</th><th>End</th><th>Duration (min)</th><th>Reasons</th></tr></thead>"
                f"<tbody>{body_rows}</tbody>"
                "</table>"
                "</div>"
            )
        else:
            table_html = '<div class="empty-msg">No representative examples available.</div>'

        parts.append(
            "\n".join(
                [
                    '<details class="reason-item">',
                    "<summary>",
                    f'<span class="reason-badge">{safe_reason}</span>',
                    f'<span class="reason-count">{count}</span>',
                    f'<span class="reason-summary">{safe_summary}</span>',
                    "</summary>",
                    '<div class="reason-body">',
                    f'<p class="physical">{safe_physical}</p>',
                    table_html,
                    "</div>",
                    "</details>",
                ]
            )
        )

    return "\n".join(parts)


def _render_dataset_section(dataset_info: dict | None) -> str:
    """Render the Dataset Curation section HTML."""
    if dataset_info is None:
        return ""

    train = dataset_info.get("train", [])
    test = dataset_info.get("test", [])
    total_pump_days = dataset_info.get("total_pump_days", 0)
    excluded = dataset_info.get("excluded_count", 0)
    copy_stats = dataset_info.get("copy_stats", {})
    sel_config = dataset_info.get("config", {})

    # Count files (accounting for multi-file days via semicolons in source_path)
    def _count_files(selections: list[dict]) -> int:
        count = 0
        for s in selections:
            sp = s.get("source_path")
            if sp:
                count += len(str(sp).split(";"))
            else:
                count += 1
        return count

    train_files = _count_files(train)
    test_files = _count_files(test)

    # Per-pump breakdown
    pump_train: dict[int, int] = {}
    pump_test: dict[int, int] = {}
    pump_excluded: dict[int, int] = {}

    for s in train:
        p = s["pump"]
        pump_train[p] = pump_train.get(p, 0) + 1
    for s in test:
        p = s["pump"]
        pump_test[p] = pump_test.get(p, 0) + 1

    all_pumps = sorted(set(list(pump_train.keys()) + list(pump_test.keys())))

    # Calculate per-pump excluded from total_pump_days breakdown
    # We need per-pump total from the annotations, which we don't have directly
    # So we just show train/test counts per pump

    pump_cards = []
    for p in all_pumps:
        tr = pump_train.get(p, 0)
        te = pump_test.get(p, 0)
        pump_cards.append(
            '<div class="pump-card">'
            f'<span class="pump-title">Pump {p}</span>'
            f'<span class="ds-train">Train: {tr} days</span>'
            f'<span class="ds-test">Test: {te} days</span>'
            "</div>"
        )
    pump_html = "".join(pump_cards) if pump_cards else '<div class="empty-msg">No dataset selections.</div>'

    # Selection criteria
    min_dur = sel_config.get("min_normal_duration_minutes", "n/a")
    max_per = sel_config.get("max_periods_per_day", "n/a")
    excl_reasons = sel_config.get("exclude_reasons", [])
    excl_text = ", ".join(excl_reasons) if excl_reasons else "none"

    # Copy stats
    copy_html = ""
    if copy_stats:
        train_cs = copy_stats.get("train", {})
        test_cs = copy_stats.get("test", {})
        copy_html = (
            '<div class="ds-copy-stats">'
            '<div class="gtitle">File Copy Results</div>'
            '<div class="ds-copy-grid">'
            '<div class="ds-copy-card">'
            '<span class="ds-copy-label">Train</span>'
            f'<span>Copied: {train_cs.get("copied", 0)}</span>'
            f'<span>Skipped: {train_cs.get("skipped", 0)}</span>'
            f'<span>Stale removed: {train_cs.get("stale_removed", 0)}</span>'
            "</div>"
            '<div class="ds-copy-card">'
            '<span class="ds-copy-label">Test</span>'
            f'<span>Copied: {test_cs.get("copied", 0)}</span>'
            f'<span>Skipped: {test_cs.get("skipped", 0)}</span>'
            f'<span>Stale removed: {test_cs.get("stale_removed", 0)}</span>'
            "</div>"
            "</div>"
            "</div>"
        )

    return f"""
    <section class="panel ds-section">
      <h2 class="panel-title">Dataset Curation</h2>
      <div class="stats-grid ds-stats-grid">
        <article class="stat-card"><div class="stat-label">Total Pump-Days</div><div class="stat-value">{total_pump_days}</div></article>
        <article class="stat-card"><div class="stat-label">Train Days</div><div class="stat-value ds-train">{len(train)}</div></article>
        <article class="stat-card"><div class="stat-label">Test Days</div><div class="stat-value ds-test">{len(test)}</div></article>
        <article class="stat-card"><div class="stat-label">Excluded</div><div class="stat-value" style="color: var(--text-muted)">{excluded}</div></article>
      </div>
      <div class="stats-grid" style="grid-template-columns: repeat(2, minmax(0, 1fr)); margin-bottom: 10px;">
        <article class="stat-card"><div class="stat-label">Train Files</div><div class="stat-value ds-train">{train_files}</div></article>
        <article class="stat-card"><div class="stat-label">Test Files</div><div class="stat-value ds-test">{test_files}</div></article>
      </div>
      <div class="pump-breakdown">{pump_html}</div>
      <div class="ds-criteria">
        <div class="gtitle" style="margin-top: 10px;">Selection Criteria</div>
        <span>Min normal duration: <strong>{min_dur} min</strong></span>
        <span>Max periods/day: <strong>{max_per}</strong></span>
        <span>Excluded reasons: <strong>{html.escape(excl_text)}</strong></span>
      </div>
      {copy_html}
    </section>
"""


def _render_html(
    df: pd.DataFrame,
    stats: dict,
    reason_freq: list[tuple[str, int]],
    error_examples: dict[str, list[dict]],
    cfg: SystemConfig,
    dataset_info: dict | None = None,
) -> str:
    records: list[dict[str, str]] = []
    for _, row in df.iterrows():
        records.append(
            {
                "date": _normalize_str(row.get("date", "")),
                "pump": _normalize_str(row.get("pump", "")),
                "period_start": _normalize_str(row.get("period_start", "")),
                "period_end": _normalize_str(row.get("period_end", "")),
                "duration_minutes": _normalize_str(row.get("duration_minutes", "")),
                "classification": _normalize_str(row.get("classification", "")),
                "reasons": _normalize_str(row.get("reasons", "")),
            }
        )

    data_json = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    system_title = cfg.system_name.replace("_", " ").title()
    safe_title = html.escape(system_title)

    cfg_pumps = [_normalize_str(p) for p in cfg.pump_ids]
    data_pumps = sorted({_normalize_str(r["pump"]) for r in records if _normalize_str(r["pump"])})
    pump_ids: list[str] = []
    for pid in cfg_pumps + data_pumps:
        if pid and pid not in pump_ids:
            pump_ids.append(pid)

    pumps_json = json.dumps(pump_ids, ensure_ascii=False).replace("</", "<\\/")

    per_pump_parts: list[str] = []
    for pump_id in pump_ids:
        pstats = stats["per_pump"].get(pump_id, {"total": 0, "normal": 0, "abnormal": 0})
        per_pump_parts.append(
            (
                '<div class="pump-card">'
                f'<span class="pump-title">Pump {html.escape(pump_id)}</span>'
                f'<span>Total: {pstats["total"]}</span>'
                f'<span class="ok">Normal: {pstats["normal"]}</span>'
                f'<span class="bad">Abnormal: {pstats["abnormal"]}</span>'
                "</div>"
            )
        )
    per_pump_html = "".join(per_pump_parts) if per_pump_parts else '<div class="empty-msg">No pump data available.</div>'

    errors_html = _render_error_examples(reason_freq, error_examples)
    dataset_html = _render_dataset_section(dataset_info)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title} - Annotation Report</title>
  <style>
    :root {{
      --bg: #1a1a2e;
      --surface: #16213e;
      --card: #0f3460;
      --text: #e0e0e0;
      --text-muted: #a0a0a0;
      --accent: #e94560;
      --normal: #27ae60;
      --abnormal: #e74c3c;
      --border: #2a2a4a;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background:
        radial-gradient(1100px 700px at -10% -20%, #203a72 0%, transparent 55%),
        radial-gradient(900px 620px at 110% -10%, #2b1f55 0%, transparent 55%),
        linear-gradient(155deg, #131628 0%, #1a1a2e 52%, #13172d 100%);
      color: var(--text);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
    }}
    .app {{ max-width: 1380px; margin: 0 auto; padding: 22px 16px 30px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(1.4rem, 2.8vw, 2.2rem); }}
    .sub {{ color: var(--text-muted); margin-bottom: 14px; }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .stat-card {{
      background: linear-gradient(135deg, rgba(15, 52, 96, 0.65), rgba(22, 33, 62, 0.95));
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .stat-label {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text-muted); }}
    .stat-value {{ margin-top: 5px; font-weight: 760; font-size: clamp(1.1rem, 2.3vw, 1.7rem); }}
    .panel {{
      background: rgba(22, 33, 62, 0.75);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 12px;
      backdrop-filter: blur(5px);
    }}
    .panel-title {{ margin: 0 0 8px; font-size: 0.96rem; color: #f0f2f9; }}
    .pump-breakdown {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 8px; }}
    .pump-card {{
      border: 1px solid var(--border);
      border-radius: 10px;
      background: rgba(15, 52, 96, 0.55);
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 0.9rem;
    }}
    .pump-title {{ font-weight: 700; }}
    .ok {{ color: #a7f5c0; }}
    .bad {{ color: #ffb0aa; }}
    .ctrl-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .gtitle {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 8px; }}
    .toggles {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .btn {{
      padding: 7px 12px;
      border-radius: 999px;
      border: 1px solid transparent;
      color: var(--text);
      background: rgba(255, 255, 255, 0.03);
      cursor: pointer;
      transition: 0.2s ease;
      user-select: none;
      font-size: 0.9rem;
    }}
    .btn:hover {{ transform: translateY(-1px); }}
    .btn.off {{ opacity: 0.38; filter: saturate(0.5); }}
    .btn.normal {{ background: rgba(39, 174, 96, 0.18); border-color: rgba(61, 211, 124, 0.65); }}
    .btn.abnormal {{ background: rgba(231, 76, 60, 0.2); border-color: rgba(247, 112, 98, 0.7); }}
    #search {{
      margin-top: 10px;
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #121c35;
      color: var(--text);
    }}
    #search:focus {{ outline: none; border-color: #4f89d8; box-shadow: 0 0 0 3px rgba(82, 141, 224, 0.25); }}
    .reason-list {{ display: grid; gap: 8px; }}
    .reason-item {{ border: 1px solid var(--border); border-radius: 10px; background: rgba(15, 52, 96, 0.45); overflow: hidden; }}
    .reason-item summary {{
      list-style: none;
      cursor: pointer;
      display: grid;
      grid-template-columns: auto auto 1fr;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
    }}
    .reason-item summary::-webkit-details-marker {{ display: none; }}
    .reason-badge {{
      border-radius: 999px;
      border: 1px solid rgba(233, 69, 96, 0.7);
      color: #ffd6dd;
      background: rgba(233, 69, 96, 0.18);
      padding: 3px 9px;
      font-size: 0.8rem;
      font-weight: 700;
      white-space: nowrap;
    }}
    .reason-count {{ color: var(--text-muted); font-size: 0.85rem; }}
    .reason-summary {{ color: #f0f1f7; font-size: 0.92rem; }}
    .reason-body {{ border-top: 1px solid var(--border); padding: 10px 12px 12px; }}
    .physical {{ margin: 0 0 10px; color: #d3dbef; line-height: 1.45; }}
    .examples-table-wrap {{ overflow: auto; }}
    .examples-table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    .examples-table th, .examples-table td {{
      border-bottom: 1px solid rgba(160, 160, 160, 0.22);
      text-align: left;
      padding: 8px 9px;
      font-size: 0.85rem;
      vertical-align: top;
    }}
    .examples-table th {{ color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.75rem; }}
    .table-shell {{
      background: rgba(22, 33, 62, 0.72);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}
    .wrap {{ max-height: 65vh; overflow: auto; }}
    table {{ width: 100%; border-collapse: separate; border-spacing: 0; min-width: 960px; }}
    thead th {{
      position: sticky;
      top: 0;
      z-index: 2;
      text-align: left;
      padding: 11px 10px;
      background: #1b2b4d;
      border-bottom: 1px solid #34496d;
      font-size: 0.8rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    th.sort {{ cursor: pointer; }}
    th.sort:hover {{ background: #243b66; }}
    .si {{ margin-left: 5px; color: #a8bfdc; }}
    tbody td {{ padding: 10px; border-bottom: 1px solid rgba(160, 160, 160, 0.18); font-size: 0.92rem; vertical-align: top; }}
    tbody tr:nth-child(odd) {{ background: rgba(18, 28, 53, 0.55); }}
    tbody tr:nth-child(even) {{ background: rgba(12, 20, 40, 0.5); }}
    tbody tr:hover {{ background: rgba(67, 98, 145, 0.38); }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 9px;
      border: 1px solid transparent;
      font-size: 0.8rem;
      font-weight: 670;
      white-space: nowrap;
    }}
    .badge.normal {{ color: #d5f7e2; border-color: rgba(74, 206, 131, 0.7); background: rgba(39, 174, 96, 0.2); }}
    .badge.abnormal {{ color: #ffd8d3; border-color: rgba(243, 118, 105, 0.7); background: rgba(231, 76, 60, 0.24); }}
    .reasons {{ max-width: 560px; word-break: break-word; line-height: 1.35; }}
    .empty-msg {{ color: var(--text-muted); font-size: 0.92rem; padding: 10px 0; }}
    .empty {{ text-align: center; color: #c0d1ef; padding: 24px; }}
    .foot {{ padding: 10px 13px; color: var(--text-muted); border-top: 1px solid rgba(160, 160, 160, 0.2); background: rgba(12, 20, 40, 0.85); }}
    .ds-train {{ color: #5dade2; }}
    .ds-test {{ color: #f0b27a; }}
    .ds-criteria {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      padding: 10px 12px;
      margin-top: 8px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: rgba(15, 52, 96, 0.35);
      font-size: 0.9rem;
      color: #d3dbef;
    }}
    .ds-copy-stats {{ margin-top: 10px; }}
    .ds-copy-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .ds-copy-card {{
      border: 1px solid var(--border);
      border-radius: 10px;
      background: rgba(15, 52, 96, 0.35);
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 3px;
      font-size: 0.85rem;
      color: #d3dbef;
    }}
    .ds-copy-label {{ font-weight: 700; font-size: 0.9rem; }}
    @media (max-width: 980px) {{
      .stats-grid {{ grid-template-columns: 1fr 1fr; }}
      .ctrl-grid {{ grid-template-columns: 1fr; }}
      .wrap {{ max-height: 56vh; }}
    }}
    @media (max-width: 640px) {{
      .stats-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="app">
    <h1>{safe_title} Operations Report</h1>
    <div class="sub">Interactive annotation report with filters, text search, sortable columns, and detector examples.</div>

    <section class="stats-grid">
      <article class="stat-card"><div class="stat-label">Total Records</div><div id="sTotal" class="stat-value">{stats['total']}</div></article>
      <article class="stat-card"><div class="stat-label">Normal</div><div id="sNormal" class="stat-value">{stats['normal']}</div></article>
      <article class="stat-card"><div class="stat-label">Abnormal</div><div id="sAbnormal" class="stat-value">{stats['abnormal']}</div></article>
      <article class="stat-card"><div class="stat-label">Date Range</div><div class="stat-value" style="font-size:1rem">{html.escape(stats['start_date'])} → {html.escape(stats['end_date'])}</div></article>
    </section>

    <section class="panel">
      <h2 class="panel-title">Per-Pump Breakdown</h2>
      <div class="pump-breakdown">{per_pump_html}</div>
    </section>

    <section class="panel">
      <h2 class="panel-title">Error Examples</h2>
      <div class="reason-list">{errors_html}</div>
    </section>

    {dataset_html}

    <section class="panel">
      <div class="ctrl-grid">
        <div>
          <div class="gtitle">Pump Filter</div>
          <div class="toggles" id="pumpToggles"></div>
        </div>
        <div>
          <div class="gtitle">Classification Filter</div>
          <div class="toggles" id="classToggles">
            <button class="btn normal" data-class="normal" aria-pressed="true">Normal</button>
            <button class="btn abnormal" data-class="abnormal" aria-pressed="true">Abnormal</button>
          </div>
        </div>
      </div>
      <input id="search" type="text" placeholder="Search all columns (e.g. pressure_off, 2026-01, Pump 3)">
    </section>

    <section class="table-shell">
      <div class="wrap">
        <table>
          <thead>
            <tr>
              <th class="sort" data-key="date">Date <span class="si"></span></th>
              <th class="sort" data-key="pump">Pump <span class="si"></span></th>
              <th class="sort" data-key="period_start">Period Start <span class="si"></span></th>
              <th class="sort" data-key="period_end">Period End <span class="si"></span></th>
              <th class="sort" data-key="duration_minutes">Duration (min) <span class="si"></span></th>
              <th class="sort" data-key="classification">Classification <span class="si"></span></th>
              <th class="sort" data-key="reasons">Reasons <span class="si"></span></th>
            </tr>
          </thead>
          <tbody id="tbody"></tbody>
        </table>
      </div>
      <div id="rowCount" class="foot">Showing 0 of 0 records</div>
    </section>
  </main>

  <script>
    const DATA = {data_json};
    const CONFIG_PUMPS = {pumps_json};
    const PUMP_COLORS = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c", "#e74c3c", "#2ecc71", "#f39c12", "#8e44ad"];

    const discoveredPumps = [...new Set(DATA.map((r) => String(r.pump || "").trim()).filter(Boolean))];
    const PUMPS = [...new Set([...CONFIG_PUMPS.map((p) => String(p).trim()).filter(Boolean), ...discoveredPumps])];
    const pumpClassById = new Map(PUMPS.map((id, idx) => [id, `pump-${{idx}}`]));

    const dynamicPumpStyle = document.createElement("style");
    dynamicPumpStyle.textContent = PUMPS.map((id, idx) => {{
      const color = PUMP_COLORS[idx % PUMP_COLORS.length];
      return [
        `.pump-${{idx}} {{ background: ${{hexToRgba(color, 0.20)}}; border-color: ${{hexToRgba(color, 0.75)}}; }}`,
        `.badge.pump-${{idx}} {{ color: #eef5ff; background: ${{hexToRgba(color, 0.24)}}; border-color: ${{hexToRgba(color, 0.80)}}; }}`
      ].join("\\n");
    }}).join("\\n");
    document.head.appendChild(dynamicPumpStyle);

    const state = {{
      pumps: new Set(PUMPS),
      classes: new Set(["normal", "abnormal"]),
      search: "",
      sortKey: "date",
      sortDir: "asc",
    }};

    const total = DATA.length;
    const el = {{
      body: document.getElementById("tbody"),
      sTotal: document.getElementById("sTotal"),
      sNormal: document.getElementById("sNormal"),
      sAbnormal: document.getElementById("sAbnormal"),
      rowCount: document.getElementById("rowCount"),
      search: document.getElementById("search"),
      pumpToggles: document.getElementById("pumpToggles"),
    }};

    function hexToRgba(hex, alpha) {{
      const h = String(hex || "").replace("#", "").trim();
      if (!/^[0-9a-fA-F]{{6}}$/.test(h)) return `rgba(52,152,219,${{alpha}})`;
      const r = parseInt(h.slice(0, 2), 16);
      const g = parseInt(h.slice(2, 4), 16);
      const b = parseInt(h.slice(4, 6), 16);
      return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
    }}

    function escapeHtml(value) {{
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }}

    function num(v) {{
      const n = Number.parseFloat(v);
      return Number.isFinite(n) ? n : Number.NEGATIVE_INFINITY;
    }}

    function compare(a, b, key) {{
      if (key === "pump") return num(a.pump) - num(b.pump);
      if (key === "duration_minutes") return num(a.duration_minutes) - num(b.duration_minutes);
      const av = String(a[key] || "").toLowerCase();
      const bv = String(b[key] || "").toLowerCase();
      return av < bv ? -1 : av > bv ? 1 : 0;
    }}

    function searchText(r) {{
      return [r.date, `pump ${{r.pump}}`, r.period_start, r.period_end, r.duration_minutes, r.classification, r.reasons]
        .join(" ")
        .toLowerCase();
    }}

    function filtered() {{
      if (!state.pumps.size || !state.classes.size) return [];
      const q = state.search.trim().toLowerCase();
      return DATA.filter((r) => state.pumps.has(String(r.pump)) && state.classes.has(String(r.classification)) && (!q || searchText(r).includes(q)));
    }}

    function sorted(rows) {{
      const d = state.sortDir === "asc" ? 1 : -1;
      return [...rows].sort((a, b) => d * compare(a, b, state.sortKey));
    }}

    function sortIndicators() {{
      document.querySelectorAll("th.sort").forEach((th) => {{
        const s = th.querySelector(".si");
        if (th.dataset.key === state.sortKey) {{
          s.textContent = state.sortDir === "asc" ? "▲" : "▼";
          th.setAttribute("aria-sort", state.sortDir === "asc" ? "ascending" : "descending");
        }} else {{
          s.textContent = "";
          th.setAttribute("aria-sort", "none");
        }}
      }});
    }}

    function renderRows(rows) {{
      if (!rows.length) {{
        el.body.innerHTML = '<tr><td class="empty" colspan="7">No records match the current filters.</td></tr>';
        return;
      }}

      el.body.innerHTML = rows
        .map((r) => {{
          const pump = String(r.pump || "");
          const pc = pumpClassById.get(pump) || "";
          const cc = r.classification === "abnormal" ? "abnormal" : "normal";
          const ct = cc === "abnormal" ? "Abnormal" : "Normal";
          const reasons = r.reasons ? escapeHtml(r.reasons) : "-";
          return [
            "<tr>",
            `<td>${{escapeHtml(r.date)}}</td>`,
            `<td><span class="badge ${{pc}}">Pump ${{escapeHtml(pump)}}</span></td>`,
            `<td>${{escapeHtml(r.period_start)}}</td>`,
            `<td>${{escapeHtml(r.period_end)}}</td>`,
            `<td>${{escapeHtml(r.duration_minutes)}}</td>`,
            `<td><span class="badge ${{cc}}">${{ct}}</span></td>`,
            `<td class="reasons">${{reasons}}</td>`,
            "</tr>",
          ].join("");
        }})
        .join("");
    }}

    function renderStats(rows) {{
      let n = 0;
      let a = 0;
      for (const r of rows) {{
        if (r.classification === "normal") n += 1;
        else if (r.classification === "abnormal") a += 1;
      }}
      el.sTotal.textContent = String(rows.length);
      el.sNormal.textContent = String(n);
      el.sAbnormal.textContent = String(a);
      el.rowCount.textContent = `Showing ${{rows.length}} of ${{total}} records`;
    }}

    function rerender() {{
      const f = filtered();
      const s = sorted(f);
      renderRows(s);
      renderStats(f);
      sortIndicators();
    }}

    function renderPumpToggles() {{
      el.pumpToggles.innerHTML = PUMPS.map((id) => {{
        const cls = pumpClassById.get(id) || "";
        return `<button class="btn ${{cls}}" data-pump="${{escapeHtml(id)}}" aria-pressed="true">Pump ${{escapeHtml(id)}}</button>`;
      }}).join("");

      el.pumpToggles.querySelectorAll(".btn").forEach((b) => {{
        b.addEventListener("click", () => {{
          const id = b.dataset.pump || "";
          if (state.pumps.has(id)) state.pumps.delete(id);
          else state.pumps.add(id);
          const on = state.pumps.has(id);
          b.classList.toggle("off", !on);
          b.setAttribute("aria-pressed", on ? "true" : "false");
          rerender();
        }});
      }});
    }}

    document.querySelectorAll("#classToggles .btn").forEach((b) => {{
      b.addEventListener("click", () => {{
        const id = b.dataset.class;
        if (!id) return;
        if (state.classes.has(id)) state.classes.delete(id);
        else state.classes.add(id);
        const on = state.classes.has(id);
        b.classList.toggle("off", !on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
        rerender();
      }});
    }});

    el.search.addEventListener("input", (e) => {{
      state.search = e.target.value || "";
      rerender();
    }});

    document.querySelectorAll("th.sort").forEach((th) => {{
      th.addEventListener("click", () => {{
        const k = th.dataset.key;
        if (!k) return;
        if (state.sortKey === k) state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
        else {{
          state.sortKey = k;
          state.sortDir = "asc";
        }}
        rerender();
      }});
    }});

    renderPumpToggles();
    rerender();
  </script>
</body>
</html>
"""


def generate_report(df: pd.DataFrame, cfg: SystemConfig, report_path: Path, dataset_info: dict | None = None) -> None:
    stats = _build_stats(df)
    reason_freq = _build_reason_frequency(df)
    error_examples = _select_error_examples(df)
    html_report = _render_html(df, stats, reason_freq, error_examples, cfg, dataset_info=dataset_info)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_report, encoding="utf-8")
    print(f"\n📊  Saved HTML report → {report_path}")