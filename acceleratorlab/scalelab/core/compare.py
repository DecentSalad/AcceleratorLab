"""
Comparative reporting — load, normalize, and compare multiple benchmark results.

Accepts any mix of single-run result files, sweep result files, and project
files. Normalizes everything into a flat list of rows, then provides functions
for side-by-side comparison, regression detection, and cost-efficiency ranking.

Usage (Python)
--------------
    from scalelab.core.compare import load_results, compare_results

    rows = load_results(["h100_sweep.json", "mi300x_sweep.json"])
    report = compare_results(rows)
    print(report.summary_table())

Usage (CLI)
-----------
    python -m scalelab.cli.run --compare h100.json mi300x.json
    python -m scalelab.cli.run --compare h100.json mi300x.json --output report.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from scalelab.core.results import normalize_results


# ---------------------------------------------------------------------------
# Loading — accepts any result file format
# ---------------------------------------------------------------------------

def _extract_benchmark_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Pull every benchmark_result out of a JSON payload regardless of whether
    it came from a single run, a sweep, or a project file.

    Single run format:
        { "scenario": {...}, "benchmark_result": {...} }

    Sweep format:
        { "results": [ { "scenario": {...}, "benchmark_result": {...} }, ... ] }

    Project file format (saved by projects.py):
        Same as single run — one result dict per file.
    """
    # Sweep file — has a top-level "results" list
    if "results" in payload and isinstance(payload["results"], list):
        return payload["results"]

    # Single run or project file — has a top-level "benchmark_result"
    if "benchmark_result" in payload:
        return [payload]

    # Unknown format — return empty rather than crashing
    return []


def load_results(paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load one or more result JSON files and return a flat normalized list.

    Each element in the returned list is one benchmark run's metrics,
    ready for comparison. The source file path is added as 'source_file'
    so rows can be traced back to their origin.

    Parameters
    ----------
    paths
        List of paths to JSON result files. Accepts single-run results,
        sweep results, and project files in any combination.

    Returns
    -------
    List of normalized row dicts — one per benchmark run across all files.
    """
    raw_items = []

    for path_str in paths:
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"Result file not found: {path_str}")

        payload = json.loads(p.read_text(encoding="utf-8"))
        rows = _extract_benchmark_rows(payload)

        # Tag each row with the source file so the comparison report
        # can group and label results by their origin
        for row in rows:
            row["_source_file"] = p.name

        raw_items.extend(rows)

    if not raw_items:
        return []

    normalized = normalize_results(raw_items)

    # Carry the source file tag through normalization
    for i, item in enumerate(raw_items):
        if i < len(normalized):
            normalized[i]["source_file"] = item.get("_source_file", "unknown")

    return normalized


# ---------------------------------------------------------------------------
# ComparisonReport — structured output from compare_results()
# ---------------------------------------------------------------------------

@dataclass
class ComparisonReport:
    """
    The output of compare_results(). Contains the normalized rows plus
    derived comparison data.
    """
    rows: List[Dict[str, Any]]

    # Unique systems present across all loaded files, e.g. ["nvidia-h100", "amd-mi300x"]
    systems: List[str] = field(default_factory=list)

    # Unique models present
    models: List[str] = field(default_factory=list)

    # Unique backends present
    backends: List[str] = field(default_factory=list)

    # Regression flags — rows where a metric got worse vs. the baseline
    regressions: List[Dict[str, Any]] = field(default_factory=list)

    def best_by(self, metric: str) -> Optional[Dict[str, Any]]:
        """
        Return the row with the highest value for the given metric.
        Useful for finding "which system had the highest tok/s" etc.
        """
        valid = [r for r in self.rows if r.get(metric, 0) > 0]
        if not valid:
            return None
        return max(valid, key=lambda r: r.get(metric, 0))

    def worst_by(self, metric: str) -> Optional[Dict[str, Any]]:
        """
        Return the row with the lowest non-zero value for the given metric.
        Useful for finding "which system had the worst p95 latency" etc.
        """
        valid = [r for r in self.rows if r.get(metric, 0) > 0]
        if not valid:
            return None
        return min(valid, key=lambda r: r.get(metric, float("inf")))

    def filter_by(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Return rows matching all provided field=value filters.

        Example:
            report.filter_by(system="nvidia-h100", concurrency=64)
        """
        result = self.rows
        for key, value in kwargs.items():
            result = [r for r in result if r.get(key) == value]
        return result

    def summary_table(self) -> str:
        """
        Return a plain-text summary table of all rows, formatted for
        terminal output. Columns: system, model, backend, concurrency,
        tok/s, ttft_ms, p95_ms, power_mean_w, tok_s_per_watt.
        """
        if not self.rows:
            return "No results to display."

        headers = [
            "system", "model", "backend", "conc",
            "tok/s", "ttft_ms", "p95_ms", "power_w", "tok_s/W",
        ]
        col_w = [18, 30, 12, 6, 8, 8, 8, 8, 8]

        def _fmt(row):
            model_short = row.get("model", "")
            if "/" in model_short:
                model_short = model_short.split("/")[-1]
            return [
                str(row.get("system",       ""))[:col_w[0]],
                str(model_short)            [:col_w[1]],
                str(row.get("backend",      ""))[:col_w[2]],
                str(row.get("concurrency",  ""))[:col_w[3]],
                str(row.get("tok_s",        0)) [:col_w[4]],
                str(row.get("ttft_ms",      0)) [:col_w[5]],
                str(row.get("p95_ms",       0)) [:col_w[6]],
                str(row.get("power_mean_w", 0)) [:col_w[7]],
                str(row.get("tok_s_per_watt",0))[:col_w[8]],
            ]

        sep   = "  ".join("-" * w for w in col_w)
        hdr   = "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
        lines = [hdr, sep]
        for row in self.rows:
            cells = _fmt(row)
            lines.append("  ".join(c.ljust(col_w[i]) for i, c in enumerate(cells)))

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "systems":     self.systems,
            "models":      self.models,
            "backends":    self.backends,
            "row_count":   len(self.rows),
            "regressions": self.regressions,
            "rows":        self.rows,
        }


# ---------------------------------------------------------------------------
# compare_results — build a ComparisonReport from normalized rows
# ---------------------------------------------------------------------------

def compare_results(rows: List[Dict[str, Any]]) -> ComparisonReport:
    """
    Build a ComparisonReport from a flat list of normalized rows.

    Extracts the unique systems, models, and backends present, sorts rows
    for consistent ordering, and returns a ComparisonReport ready for
    display or further analysis.

    Parameters
    ----------
    rows
        Normalized rows as returned by load_results().

    Returns
    -------
    ComparisonReport
    """
    if not rows:
        return ComparisonReport(rows=[])

    systems  = sorted(set(r.get("system",  "unknown") for r in rows))
    models   = sorted(set(r.get("model",   "unknown") for r in rows))
    backends = sorted(set(r.get("backend", "unknown") for r in rows))

    # Sort rows for deterministic output:
    # system → model → backend → concurrency → prompt_tokens
    def _sort_key(r):
        return (
            r.get("system",      ""),
            r.get("model",       ""),
            r.get("backend",     ""),
            r.get("concurrency", 0),
        )

    sorted_rows = sorted(rows, key=_sort_key)

    return ComparisonReport(
        rows=sorted_rows,
        systems=systems,
        models=models,
        backends=backends,
    )


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

# Default thresholds — a metric must degrade by more than this fraction
# before it is flagged as a regression. E.g. 0.05 = 5% degradation.
DEFAULT_REGRESSION_THRESHOLDS = {
    "tok_s":    0.05,   # throughput drops >5%
    "ttft_ms":  0.10,   # TTFT gets >10% worse
    "p95_ms":   0.10,   # p95 latency gets >10% worse
}


def detect_regressions(
    baseline: List[Dict[str, Any]],
    candidate: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Compare candidate results against a baseline and flag regressions.

    Matches rows by (system, model, backend, concurrency). For each
    matched pair checks whether candidate metrics degraded beyond the
    threshold relative to baseline.

    Parameters
    ----------
    baseline
        Normalized rows from the reference run (e.g. before a driver update).
    candidate
        Normalized rows from the run being evaluated.
    thresholds
        Dict of metric → fractional degradation threshold.
        Defaults to DEFAULT_REGRESSION_THRESHOLDS if not provided.

    Returns
    -------
    List of regression dicts, each containing:
        metric, baseline_value, candidate_value, degradation_pct,
        system, model, backend, concurrency
    """
    thresholds = thresholds or DEFAULT_REGRESSION_THRESHOLDS

    # Index baseline rows by match key for O(1) lookup
    baseline_index: Dict[tuple, Dict] = {}
    for row in baseline:
        key = (
            row.get("system",      ""),
            row.get("model",       ""),
            row.get("backend",     ""),
            row.get("concurrency", 0),
        )
        baseline_index[key] = row

    regressions = []

    for cand_row in candidate:
        key = (
            cand_row.get("system",      ""),
            cand_row.get("model",       ""),
            cand_row.get("backend",     ""),
            cand_row.get("concurrency", 0),
        )
        base_row = baseline_index.get(key)
        if base_row is None:
            continue   # no matching baseline run — skip

        for metric, threshold in thresholds.items():
            base_val  = float(base_row.get(metric,  0) or 0)
            cand_val  = float(cand_row.get(metric, 0) or 0)

            if base_val == 0:
                continue

            # For throughput metrics (higher is better): regression = drop
            # For latency metrics (lower is better): regression = increase
            latency_metrics = {"ttft_ms", "p95_ms", "mean_latency_ms"}

            if metric in latency_metrics:
                # Latency increased — candidate is worse
                if cand_val > base_val:
                    degradation = (cand_val - base_val) / base_val
                    if degradation > threshold:
                        regressions.append({
                            "metric":           metric,
                            "baseline_value":   round(base_val, 4),
                            "candidate_value":  round(cand_val, 4),
                            "degradation_pct":  round(degradation * 100, 2),
                            "system":           cand_row.get("system", ""),
                            "model":            cand_row.get("model",  ""),
                            "backend":          cand_row.get("backend",""),
                            "concurrency":      cand_row.get("concurrency", 0),
                        })
            else:
                # Throughput decreased — candidate is worse
                if cand_val < base_val:
                    degradation = (base_val - cand_val) / base_val
                    if degradation > threshold:
                        regressions.append({
                            "metric":           metric,
                            "baseline_value":   round(base_val, 4),
                            "candidate_value":  round(cand_val, 4),
                            "degradation_pct":  round(degradation * 100, 2),
                            "system":           cand_row.get("system", ""),
                            "model":            cand_row.get("model",  ""),
                            "backend":          cand_row.get("backend",""),
                            "concurrency":      cand_row.get("concurrency", 0),
                        })

    return regressions


# ---------------------------------------------------------------------------
# Cost-efficiency enrichment
# ---------------------------------------------------------------------------

def add_cost_efficiency(
    rows: List[Dict[str, Any]],
    price_per_hour: float,
) -> List[Dict[str, Any]]:
    """
    Add tok/s per dollar to every row.

    Parameters
    ----------
    rows
        Normalized rows to enrich.
    price_per_hour
        Instance or cluster cost in USD per hour (e.g. 32.77 for an
        8× H100 instance on a major cloud provider).

    Returns
    -------
    Same list with 'tok_s_per_dollar' added to each row.
    Price is converted to per-second for comparison with tok/s.
    """
    price_per_second = price_per_hour / 3600.0
    for row in rows:
        tok_s = float(row.get("tok_s", 0) or 0)
        if tok_s > 0 and price_per_second > 0:
            row["tok_s_per_dollar"] = round(tok_s / price_per_second, 4)
        else:
            row["tok_s_per_dollar"] = 0.0
    return rows