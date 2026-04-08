"""
Markdown report generator — turns a ComparisonReport into a shareable document.

Produces a structured .md file with:
  - Executive summary (best throughput, best efficiency, systems compared)
  - Full comparison table (all rows, all key metrics)
  - Per-system summary (best tok/s, best tok/s-per-watt at each concurrency)
  - Regression section (if regressions were detected)
  - Hardware telemetry section (GPU utilization, power, temperature)

Usage
-----
    from scalelab.core.compare import load_results, compare_results
    from scalelab.core.report import generate_markdown_report

    rows   = load_results(["h100.json", "mi300x.json"])
    report = compare_results(rows)
    md     = generate_markdown_report(report)

    with open("comparison.md", "w") as f:
        f.write(md)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from scalelab.core.compare import ComparisonReport


# ---------------------------------------------------------------------------
# Internal formatting helpers
# ---------------------------------------------------------------------------

def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Build a GitHub-flavoured markdown table."""
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep   = "| " + " | ".join("-" * w for w in widths) + " |"
    hdr   = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    lines = [hdr, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row[i]).ljust(widths[i])
                                       for i in range(len(headers))) + " |")
    return "\n".join(lines)


def _short_model(model: str) -> str:
    """Strip HuggingFace org prefix for display: org/model → model."""
    return model.split("/")[-1] if "/" in model else model


def _pct(value: float) -> str:
    """Format a 0-100 float as a percentage string."""
    return f"{value:.1f}%" if value else "—"


def _val(value, digits: int = 2) -> str:
    """Format a number, returning '—' for zero/None."""
    if value is None or value == 0:
        return "—"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

def _section_summary(report: ComparisonReport) -> str:
    lines = ["## Executive Summary", ""]

    lines.append(f"**Systems compared:** {', '.join(report.systems) or 'n/a'}")
    lines.append(f"**Models:**  {', '.join(_short_model(m) for m in report.models) or 'n/a'}")
    lines.append(f"**Backends:** {', '.join(report.backends) or 'n/a'}")
    lines.append(f"**Total benchmark runs:** {len(report.rows)}")
    lines.append("")

    best_toks = report.best_by("tok_s")
    if best_toks:
        lines.append(
            f"**Highest throughput:** {best_toks['tok_s']} tok/s — "
            f"{best_toks.get('system','?')} / "
            f"{_short_model(best_toks.get('model','?'))} / "
            f"{best_toks.get('backend','?')} @ "
            f"concurrency {best_toks.get('concurrency','?')}"
        )

    best_eff = report.best_by("tok_s_per_watt")
    if best_eff and best_eff.get("tok_s_per_watt", 0) > 0:
        lines.append(
            f"**Best efficiency:** {best_eff['tok_s_per_watt']} tok/s per watt — "
            f"{best_eff.get('system','?')} / "
            f"{_short_model(best_eff.get('model','?'))} / "
            f"{best_eff.get('backend','?')} @ "
            f"concurrency {best_eff.get('concurrency','?')}"
        )

    worst_lat = report.worst_by("p95_ms")
    if worst_lat:
        lines.append(
            f"**Best p95 latency:** {worst_lat.get('p95_ms','?')} ms — "
            f"{worst_lat.get('system','?')} / "
            f"{_short_model(worst_lat.get('model','?'))} / "
            f"{worst_lat.get('backend','?')} @ "
            f"concurrency {worst_lat.get('concurrency','?')}"
        )

    if report.regressions:
        lines.append(f"\n> ⚠️  **{len(report.regressions)} regression(s) detected** — see Regressions section below.")

    return "\n".join(lines)


def _section_comparison_table(report: ComparisonReport) -> str:
    lines = ["## Full Comparison Table", ""]

    has_telemetry = any(r.get("telemetry_available") for r in report.rows)
    has_cost      = any(r.get("tok_s_per_dollar", 0) for r in report.rows)

    headers = [
        "System", "Model", "Backend", "Conc",
        "tok/s", "TTFT ms", "p95 ms", "Success",
    ]
    if has_telemetry:
        headers += ["GPU util%", "Power W", "tok/s/W"]
    if has_cost:
        headers += ["tok/s/$"]
    headers += ["SLO"]

    rows_out = []
    for r in report.rows:
        row = [
            r.get("system",      ""),
            _short_model(r.get("model",   "")),
            r.get("backend",     ""),
            r.get("concurrency", ""),
            _val(r.get("tok_s")),
            _val(r.get("ttft_ms")),
            _val(r.get("p95_ms")),
            _pct(r.get("success_rate", 0) * 100),
        ]
        if has_telemetry:
            row += [
                _pct(r.get("gpu_util_mean_pct", 0)),
                _val(r.get("power_mean_w")),
                _val(r.get("tok_s_per_watt"), digits=4),
            ]
        if has_cost:
            row += [_val(r.get("tok_s_per_dollar"), digits=4)]
        row += ["✓" if r.get("meets_slo") else "✗"]
        rows_out.append(row)

    lines.append(_md_table(headers, rows_out))
    return "\n".join(lines)


def _section_per_system(report: ComparisonReport) -> str:
    """One subsection per system showing its best result at each concurrency."""
    lines = ["## Per-System Highlights", ""]

    for system in report.systems:
        lines.append(f"### {system}")
        system_rows = [r for r in report.rows if r.get("system") == system]
        if not system_rows:
            lines.append("_No results._\n")
            continue

        # Peak throughput row
        best = max(system_rows, key=lambda r: r.get("tok_s", 0))
        lines.append(
            f"- **Peak throughput:** {best.get('tok_s','?')} tok/s "
            f"({_short_model(best.get('model','?'))} / {best.get('backend','?')} "
            f"@ concurrency {best.get('concurrency','?')})"
        )

        # Best efficiency row (only if telemetry available)
        eff_rows = [r for r in system_rows if r.get("tok_s_per_watt", 0) > 0]
        if eff_rows:
            best_eff = max(eff_rows, key=lambda r: r.get("tok_s_per_watt", 0))
            lines.append(
                f"- **Best efficiency:** {best_eff.get('tok_s_per_watt','?')} tok/s/W "
                f"({_short_model(best_eff.get('model','?'))} / {best_eff.get('backend','?')} "
                f"@ concurrency {best_eff.get('concurrency','?')})"
            )

        # Best p95
        lat_rows = [r for r in system_rows if r.get("p95_ms", 0) > 0]
        if lat_rows:
            best_lat = min(lat_rows, key=lambda r: r.get("p95_ms", float("inf")))
            lines.append(
                f"- **Best p95:** {best_lat.get('p95_ms','?')} ms "
                f"(@ concurrency {best_lat.get('concurrency','?')})"
            )

        lines.append("")

    return "\n".join(lines)


def _section_telemetry(report: ComparisonReport) -> str:
    """Hardware telemetry summary — only included when telemetry was collected."""
    telem_rows = [r for r in report.rows if r.get("telemetry_available")]
    if not telem_rows:
        return ""

    lines = ["## Hardware Telemetry", ""]
    lines.append("_Metrics collected via nvidia-smi / rocm-smi during each benchmark window._\n")

    headers = [
        "System", "Model", "Backend", "Conc",
        "GPU util mean", "GPU util peak",
        "VRAM used GB", "VRAM total GB",
        "Power mean W", "Power peak W",
        "Temp mean °C", "Temp peak °C",
    ]
    rows_out = []
    for r in telem_rows:
        rows_out.append([
            r.get("system",         ""),
            _short_model(r.get("model", "")),
            r.get("backend",        ""),
            r.get("concurrency",    ""),
            _pct(r.get("gpu_util_mean_pct", 0)),
            _pct(r.get("gpu_util_peak_pct", 0)),
            _val(r.get("vram_used_mean_gb")),
            _val(r.get("vram_total_gb")),
            _val(r.get("power_mean_w")),
            _val(r.get("power_peak_w")),
            _val(r.get("temp_mean_c")),
            _val(r.get("temp_peak_c")),
        ])

    lines.append(_md_table(headers, rows_out))
    return "\n".join(lines)


def _section_regressions(report: ComparisonReport) -> str:
    """Regression section — only included when regressions were detected."""
    if not report.regressions:
        return ""

    lines = [
        "## Regressions Detected",
        "",
        f"> {len(report.regressions)} metric(s) degraded beyond threshold "
        f"compared to the baseline.\n",
    ]

    headers = ["System", "Model", "Backend", "Conc", "Metric",
               "Baseline", "Candidate", "Degradation"]
    rows_out = []
    for reg in report.regressions:
        rows_out.append([
            reg.get("system",     ""),
            _short_model(reg.get("model", "")),
            reg.get("backend",    ""),
            reg.get("concurrency",""),
            reg.get("metric",     ""),
            str(reg.get("baseline_value",  "")),
            str(reg.get("candidate_value", "")),
            f"{reg.get('degradation_pct','?')}%",
        ])

    lines.append(_md_table(headers, rows_out))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_markdown_report(
    report: ComparisonReport,
    title: str = "AcceleratorLab Benchmark Report",
) -> str:
    """
    Generate a complete markdown report from a ComparisonReport.

    Parameters
    ----------
    report
        The ComparisonReport produced by compare_results().
    title
        Report title shown as the top-level heading.

    Returns
    -------
    str
        Complete markdown document, ready to write to a .md file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = [
        f"# {title}",
        f"_Generated: {timestamp}_",
        "",
        _section_summary(report),
        "",
        _section_comparison_table(report),
        "",
        _section_per_system(report),
    ]

    telemetry_section = _section_telemetry(report)
    if telemetry_section:
        sections += ["", telemetry_section]

    regression_section = _section_regressions(report)
    if regression_section:
        sections += ["", regression_section]

    return "\n".join(sections) + "\n"