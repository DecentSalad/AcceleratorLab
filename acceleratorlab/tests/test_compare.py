"""
Tests for Phase 4 — Comparative Reporting.

Run with:  pytest tests/test_compare.py -v

Covers:
  - load_results() parsing single-run, sweep, and mixed JSON files
  - compare_results() building a correct ComparisonReport
  - ComparisonReport helpers (best_by, worst_by, filter_by, summary_table)
  - detect_regressions() correctly flagging degraded metrics
  - add_cost_efficiency() computing tok/s per dollar
  - generate_markdown_report() producing valid, complete markdown

No GPU or running server required. All tests use in-memory fake data.
"""
import json
import pytest
from pathlib import Path

from scalelab.core.compare import (
    load_results, compare_results, detect_regressions,
    add_cost_efficiency, ComparisonReport,
)
from scalelab.core.report import generate_markdown_report


# ---------------------------------------------------------------------------
# Helpers — build fake result dicts in the same shape the pipeline produces
# ---------------------------------------------------------------------------

def make_result(
    system: str = "nvidia-h100",
    model: str  = "meta-llama/Llama-3.1-8B-Instruct",
    backend: str = "vllm",
    concurrency: int = 16,
    tok_s: float = 500.0,
    ttft_ms: float = 300.0,
    p95_ms: float = 800.0,
    success_rate: float = 1.0,
    meets_slo: bool = True,
    power_mean_w: float = 0.0,
    gpu_util_mean_pct: float = 0.0,
    telemetry_available: bool = False,
) -> dict:
    """
    Build a fake benchmark result dict in the same structure that
    execute_scenario() produces. Used to construct test data without
    running actual benchmarks.
    """
    return {
        "scenario": {
            "name": f"test-{system}-conc{concurrency}",
            "cluster":  {"accelerator_vendor": system.split("-")[0],
                         "accelerator_arch":   system.split("-")[1] if "-" in system else "unknown"},
            "workload": {"model": model, "backend": backend, "concurrency": concurrency},
            "launch":   {"executor": "local"},
        },
        "launch_result":    {"vendor_validation": "passed"},
        "benchmark_result": {
            "system":              system,
            "model":               model,
            "backend":             backend,
            "concurrency":         concurrency,
            "tok_s":               tok_s,
            "ttft_ms":             ttft_ms,
            "mean_latency_ms":     ttft_ms * 1.5,
            "p95_ms":              p95_ms,
            "success_rate":        success_rate,
            "requests_ok":         100,
            "duration_s":          60.0,
            "traffic_pattern":     "steady",
            "meets_slo":           meets_slo,
            "telemetry_available": telemetry_available,
            "telemetry_vendor":    system.split("-")[0],
            "gpu_count":           8 if telemetry_available else 0,
            "telemetry_samples":   60 if telemetry_available else 0,
            "telemetry_error":     "",
            "gpu_util_mean_pct":   gpu_util_mean_pct,
            "gpu_util_peak_pct":   min(gpu_util_mean_pct + 5, 100),
            "vram_used_mean_gb":   40.0 if telemetry_available else 0.0,
            "vram_total_gb":       80.0 if telemetry_available else 0.0,
            "power_mean_w":        power_mean_w,
            "power_peak_w":        power_mean_w * 1.1 if power_mean_w else 0.0,
            "temp_mean_c":         72.0 if telemetry_available else 0.0,
            "temp_peak_c":         78.0 if telemetry_available else 0.0,
            "tok_s_per_watt":      round(tok_s / power_mean_w, 4) if power_mean_w else 0.0,
        },
    }


def make_sweep_result(results: list, name: str = "test-sweep") -> dict:
    """
    Wrap a list of result dicts in the sweep result format that
    run_sweep().to_dict() produces.
    """
    return {
        "name":           name,
        "config":         {"concurrency": [16], "prompt_tokens": [512], "output_tokens": [128]},
        "total_runs":     len(results),
        "completed_runs": len(results),
        "failed_runs":    0,
        "elapsed_s":      10.0,
        "results":        results,
    }


def write_json(path: Path, data: dict) -> None:
    """Write a dict as JSON to a Path."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ===========================================================================
# load_results
# ===========================================================================

class TestLoadResults:
    """Tests for load_results() — parsing result files into normalized rows."""

    def test_single_run_file(self, tmp_path):
        """
        load_results() should parse a single-run result file into one row.
        A single run file has the structure { scenario, benchmark_result }.
        """
        f = tmp_path / "run.json"
        write_json(f, make_result(system="nvidia-h100", tok_s=500.0))
        rows = load_results([str(f)])
        assert len(rows) == 1
        assert rows[0]["system"] == "nvidia-h100"

    def test_sweep_file_produces_multiple_rows(self, tmp_path):
        """
        A sweep file contains many benchmark results. load_results() should
        return one normalized row per benchmark run inside the sweep.
        """
        results = [
            make_result(concurrency=c) for c in [1, 4, 8, 16]
        ]
        f = tmp_path / "sweep.json"
        write_json(f, make_sweep_result(results))
        rows = load_results([str(f)])
        assert len(rows) == 4

    def test_multiple_files_combined(self, tmp_path):
        """
        When multiple files are passed, load_results() should return the
        combined rows from all files in a single flat list.
        """
        f1 = tmp_path / "h100.json"
        f2 = tmp_path / "mi300x.json"
        write_json(f1, make_result(system="nvidia-h100"))
        write_json(f2, make_result(system="amd-mi300x"))
        rows = load_results([str(f1), str(f2)])
        assert len(rows) == 2
        systems = {r["system"] for r in rows}
        assert "nvidia-h100" in systems
        assert "amd-mi300x" in systems

    def test_source_file_tagged(self, tmp_path):
        """
        Each row should be tagged with the name of the file it came from.
        This allows the comparison report to label rows by their origin.
        """
        f = tmp_path / "my_results.json"
        write_json(f, make_result())
        rows = load_results([str(f)])
        assert rows[0].get("source_file") == "my_results.json"

    def test_missing_file_raises(self, tmp_path):
        """
        Attempting to load a file that doesn't exist should raise
        FileNotFoundError with the path in the message.
        """
        with pytest.raises(FileNotFoundError, match="not_here.json"):
            load_results([str(tmp_path / "not_here.json")])

    def test_empty_file_list_returns_empty(self):
        """
        An empty file list should return an empty list, not raise an error.
        """
        rows = load_results([])
        assert rows == []

    def test_metrics_are_normalized(self, tmp_path):
        """
        The rows returned by load_results() should have all the fields
        that normalize_results() guarantees — tok_s, ttft_ms, p95_ms, etc.
        """
        f = tmp_path / "run.json"
        write_json(f, make_result(tok_s=842.5, ttft_ms=310.2))
        rows = load_results([str(f)])
        assert rows[0]["tok_s"]    == 842.5
        assert rows[0]["ttft_ms"]  == 310.2
        assert "p95_ms"        in rows[0]
        assert "success_rate"  in rows[0]
        assert "meets_slo"     in rows[0]

    def test_telemetry_fields_present(self, tmp_path):
        """
        Phase 1 telemetry fields should survive the load+normalize pipeline
        and be accessible in the returned rows.
        """
        f = tmp_path / "run.json"
        write_json(f, make_result(
            telemetry_available=True, power_mean_w=2400.0, gpu_util_mean_pct=88.5
        ))
        rows = load_results([str(f)])
        assert rows[0]["telemetry_available"] is True
        assert rows[0]["power_mean_w"]        == 2400.0
        assert rows[0]["gpu_util_mean_pct"]   == 88.5

    def test_tok_s_per_watt_computed(self, tmp_path):
        """
        tok_s_per_watt should be computed during normalization when
        both tok_s and power_mean_w are available.
        """
        f = tmp_path / "run.json"
        write_json(f, make_result(tok_s=500.0, power_mean_w=2500.0))
        rows = load_results([str(f)])
        assert rows[0]["tok_s_per_watt"] == round(500.0 / 2500.0, 4)


# ===========================================================================
# compare_results / ComparisonReport
# ===========================================================================

class TestCompareResults:
    """Tests for compare_results() and the ComparisonReport it produces."""

    def test_systems_extracted(self):
        """
        systems in the report should list every unique system string
        present across all rows, sorted alphabetically.
        """
        rows = [
            make_result(system="nvidia-h100")["benchmark_result"],
            make_result(system="amd-mi300x")["benchmark_result"],
        ]
        from scalelab.core.results import normalize_results
        normalized = normalize_results([{"benchmark_result": r} for r in rows])
        report = compare_results(normalized)
        assert "nvidia-h100" in report.systems
        assert "amd-mi300x"  in report.systems
        assert report.systems == sorted(report.systems)

    def test_models_extracted(self):
        """
        models in the report should list every unique model ID present.
        """
        rows = [
            make_result(model="meta-llama/Llama-3.1-8B-Instruct"),
            make_result(model="Qwen/Qwen2.5-7B-Instruct"),
        ]
        from scalelab.core.results import normalize_results
        normalized = normalize_results([{"benchmark_result": r["benchmark_result"]} for r in rows])
        report = compare_results(normalized)
        assert len(report.models) == 2

    def test_empty_rows_returns_empty_report(self):
        """
        compare_results([]) should return an empty ComparisonReport
        without raising an error.
        """
        report = compare_results([])
        assert report.rows     == []
        assert report.systems  == []
        assert report.models   == []
        assert report.backends == []

    def test_rows_sorted_by_system_then_concurrency(self, tmp_path):
        """
        Rows should be sorted by (system, model, backend, concurrency)
        for deterministic output. This makes diffs between reports readable.
        """
        f1 = tmp_path / "r1.json"
        f2 = tmp_path / "r2.json"
        f3 = tmp_path / "r3.json"
        write_json(f1, make_result(system="nvidia-h100", concurrency=32))
        write_json(f2, make_result(system="amd-mi300x",  concurrency=8))
        write_json(f3, make_result(system="nvidia-h100", concurrency=8))
        rows   = load_results([str(f1), str(f2), str(f3)])
        report = compare_results(rows)
        concs  = [r["concurrency"] for r in report.rows
                  if r["system"] == "nvidia-h100"]
        assert concs == sorted(concs)

    def test_best_by_tok_s(self, tmp_path):
        """
        best_by("tok_s") should return the row with the highest throughput.
        """
        f1 = tmp_path / "r1.json"
        f2 = tmp_path / "r2.json"
        write_json(f1, make_result(system="nvidia-h100", tok_s=1200.0))
        write_json(f2, make_result(system="amd-mi300x",  tok_s=980.0))
        rows   = load_results([str(f1), str(f2)])
        report = compare_results(rows)
        best   = report.best_by("tok_s")
        assert best["tok_s"]  == 1200.0
        assert best["system"] == "nvidia-h100"

    def test_worst_by_p95_ms(self, tmp_path):
        """
        worst_by("p95_ms") should return the row with the LOWEST p95
        (best latency), since lower is better for latency metrics.
        """
        f1 = tmp_path / "r1.json"
        f2 = tmp_path / "r2.json"
        write_json(f1, make_result(system="nvidia-h100", p95_ms=400.0))
        write_json(f2, make_result(system="amd-mi300x",  p95_ms=650.0))
        rows   = load_results([str(f1), str(f2)])
        report = compare_results(rows)
        best   = report.worst_by("p95_ms")
        assert best["p95_ms"] == 400.0

    def test_filter_by_system(self, tmp_path):
        """
        filter_by(system=...) should return only rows matching that system.
        """
        f1 = tmp_path / "r1.json"
        f2 = tmp_path / "r2.json"
        write_json(f1, make_result(system="nvidia-h100"))
        write_json(f2, make_result(system="amd-mi300x"))
        rows   = load_results([str(f1), str(f2)])
        report = compare_results(rows)
        h100   = report.filter_by(system="nvidia-h100")
        assert len(h100) == 1
        assert all(r["system"] == "nvidia-h100" for r in h100)

    def test_filter_by_concurrency(self, tmp_path):
        """
        filter_by(concurrency=...) should filter to the specified value.
        Useful for slicing a sweep result at a fixed concurrency level.
        """
        results = [make_result(concurrency=c) for c in [1, 4, 8, 16]]
        f = tmp_path / "sweep.json"
        write_json(f, make_sweep_result(results))
        rows   = load_results([str(f)])
        report = compare_results(rows)
        subset = report.filter_by(concurrency=8)
        assert len(subset) == 1
        assert subset[0]["concurrency"] == 8

    def test_summary_table_is_string(self, tmp_path):
        """
        summary_table() should return a non-empty string.
        The terminal-formatted table is used for CLI output.
        """
        f = tmp_path / "run.json"
        write_json(f, make_result())
        rows   = load_results([str(f)])
        report = compare_results(rows)
        table  = report.summary_table()
        assert isinstance(table, str)
        assert len(table) > 0

    def test_summary_table_empty_report(self):
        """
        summary_table() on an empty report should return a message,
        not raise an error.
        """
        report = compare_results([])
        table  = report.summary_table()
        assert isinstance(table, str)

    def test_to_dict_has_required_keys(self, tmp_path):
        """
        to_dict() must include all keys that consumers of the report
        (GUI, downstream scripts) depend on.
        """
        f = tmp_path / "run.json"
        write_json(f, make_result())
        rows   = load_results([str(f)])
        report = compare_results(rows)
        d      = report.to_dict()
        for key in ["systems", "models", "backends", "row_count", "regressions", "rows"]:
            assert key in d, f"Missing key: {key}"


# ===========================================================================
# detect_regressions
# ===========================================================================

class TestDetectRegressions:
    """Tests for detect_regressions() — comparing candidate runs against a baseline."""

    def _rows(self, **kwargs):
        """Build a single normalized row from a make_result dict."""
        r = make_result(**kwargs)
        from scalelab.core.results import normalize_results
        return normalize_results([{"benchmark_result": r["benchmark_result"]}])

    def test_no_regression_when_metrics_unchanged(self):
        """
        When candidate metrics match baseline exactly, no regressions
        should be flagged.
        """
        baseline  = self._rows(tok_s=500.0, ttft_ms=300.0, p95_ms=800.0)
        candidate = self._rows(tok_s=500.0, ttft_ms=300.0, p95_ms=800.0)
        regressions = detect_regressions(baseline, candidate)
        assert regressions == []

    def test_throughput_drop_flagged(self):
        """
        A significant throughput drop (>5% by default) should be flagged
        as a regression in the tok_s metric.
        """
        baseline  = self._rows(tok_s=500.0)
        candidate = self._rows(tok_s=400.0)   # 20% drop
        regressions = detect_regressions(baseline, candidate)
        metrics = [r["metric"] for r in regressions]
        assert "tok_s" in metrics

    def test_latency_increase_flagged(self):
        """
        A significant p95 latency increase (>10% by default) should be
        flagged as a regression in the p95_ms metric.
        """
        baseline  = self._rows(p95_ms=800.0)
        candidate = self._rows(p95_ms=950.0)   # ~19% increase
        regressions = detect_regressions(baseline, candidate)
        metrics = [r["metric"] for r in regressions]
        assert "p95_ms" in metrics

    def test_small_change_not_flagged(self):
        """
        A change within the threshold (e.g. 2% throughput drop vs 5%
        threshold) should not be flagged as a regression.
        """
        baseline  = self._rows(tok_s=500.0)
        candidate = self._rows(tok_s=492.0)   # 1.6% drop — under 5% threshold
        regressions = detect_regressions(baseline, candidate)
        metrics = [r["metric"] for r in regressions]
        assert "tok_s" not in metrics

    def test_improvement_not_flagged(self):
        """
        Better metrics in the candidate (higher throughput, lower latency)
        should never be flagged as regressions.
        """
        baseline  = self._rows(tok_s=500.0, p95_ms=800.0)
        candidate = self._rows(tok_s=600.0, p95_ms=600.0)   # better on both
        regressions = detect_regressions(baseline, candidate)
        assert regressions == []

    def test_regression_contains_required_fields(self):
        """
        Each regression dict must contain the fields needed to identify
        and explain the problem: metric, values, degradation%, and location.
        """
        baseline  = self._rows(tok_s=500.0)
        candidate = self._rows(tok_s=300.0)   # 40% drop
        regressions = detect_regressions(baseline, candidate)
        assert len(regressions) > 0
        reg = regressions[0]
        for field in ["metric", "baseline_value", "candidate_value",
                      "degradation_pct", "system", "model", "backend", "concurrency"]:
            assert field in reg, f"Missing field: {field}"

    def test_degradation_pct_is_correct(self):
        """
        The degradation_pct value should accurately reflect the percentage
        drop from baseline to candidate.
        """
        baseline  = self._rows(tok_s=500.0)
        candidate = self._rows(tok_s=400.0)   # exactly 20% drop
        regressions = detect_regressions(baseline, candidate)
        tok_s_reg = next(r for r in regressions if r["metric"] == "tok_s")
        assert abs(tok_s_reg["degradation_pct"] - 20.0) < 0.1

    def test_custom_thresholds(self):
        """
        Custom thresholds should override the defaults. Setting a 0%
        threshold should flag any degradation at all.
        """
        baseline  = self._rows(tok_s=500.0)
        candidate = self._rows(tok_s=499.0)   # 0.2% drop — under default 5% threshold
        # With 0% threshold it should still be flagged
        regressions = detect_regressions(baseline, candidate, thresholds={"tok_s": 0.0})
        metrics = [r["metric"] for r in regressions]
        assert "tok_s" in metrics

    def test_no_baseline_match_skipped(self):
        """
        Candidate rows with no matching baseline row (different system,
        model, or concurrency) should be silently skipped, not flagged.
        """
        baseline  = self._rows(system="nvidia-h100", concurrency=16)
        candidate = self._rows(system="amd-mi300x",  concurrency=16)
        regressions = detect_regressions(baseline, candidate)
        # Different systems — no match possible, should be empty
        assert regressions == []

    def test_empty_baseline_returns_empty(self):
        """
        An empty baseline list should produce no regressions.
        """
        candidate = self._rows(tok_s=500.0)
        regressions = detect_regressions([], candidate)
        assert regressions == []


# ===========================================================================
# add_cost_efficiency
# ===========================================================================

class TestAddCostEfficiency:
    """Tests for add_cost_efficiency() — computing tok/s per dollar."""

    def test_tok_s_per_dollar_computed(self, tmp_path):
        """
        add_cost_efficiency() should add tok_s_per_dollar to each row.
        At $36/hr and 500 tok/s: price_per_second = 0.01,
        tok_s_per_dollar = 500 / 0.01 = 50000.
        """
        f = tmp_path / "run.json"
        write_json(f, make_result(tok_s=500.0))
        rows = load_results([str(f)])
        rows = add_cost_efficiency(rows, price_per_hour=36.0)
        expected = round(500.0 / (36.0 / 3600), 4)
        assert rows[0]["tok_s_per_dollar"] == expected

    def test_zero_tok_s_gives_zero(self, tmp_path):
        """
        When tok_s is 0 (failed run), tok_s_per_dollar should be 0,
        not raise a division error.
        """
        f = tmp_path / "run.json"
        write_json(f, make_result(tok_s=0.0))
        rows = load_results([str(f)])
        rows = add_cost_efficiency(rows, price_per_hour=10.0)
        assert rows[0]["tok_s_per_dollar"] == 0.0

    def test_all_rows_enriched(self, tmp_path):
        """
        Every row in the list should have tok_s_per_dollar added,
        not just the first one.
        """
        results = [make_result(tok_s=float(c * 100), concurrency=c)
                   for c in [1, 4, 8]]
        f = tmp_path / "sweep.json"
        write_json(f, make_sweep_result(results))
        rows = load_results([str(f)])
        rows = add_cost_efficiency(rows, price_per_hour=10.0)
        assert all("tok_s_per_dollar" in r for r in rows)


# ===========================================================================
# generate_markdown_report
# ===========================================================================

class TestGenerateMarkdownReport:
    """Tests for generate_markdown_report() — the human-readable output."""

    def _report(self, tmp_path, results: list = None) -> ComparisonReport:
        """Build a ComparisonReport from a list of make_result() dicts."""
        if results is None:
            results = [make_result(system="nvidia-h100"), make_result(system="amd-mi300x")]
        f = tmp_path / "sweep.json"
        write_json(f, make_sweep_result(results))
        rows = load_results([str(f)])
        return compare_results(rows)

    def test_returns_string(self, tmp_path):
        """generate_markdown_report() must return a str."""
        md = generate_markdown_report(self._report(tmp_path))
        assert isinstance(md, str)

    def test_contains_title(self, tmp_path):
        """
        The report should start with a top-level heading. The default
        title is 'AcceleratorLab Benchmark Report'.
        """
        md = generate_markdown_report(self._report(tmp_path))
        assert "# AcceleratorLab" in md

    def test_custom_title(self, tmp_path):
        """
        A custom title passed to generate_markdown_report() should appear
        as the top-level heading.
        """
        md = generate_markdown_report(self._report(tmp_path), title="My Custom Report")
        assert "# My Custom Report" in md

    def test_contains_executive_summary(self, tmp_path):
        """
        The report must include an Executive Summary section.
        """
        md = generate_markdown_report(self._report(tmp_path))
        assert "Executive Summary" in md

    def test_contains_comparison_table(self, tmp_path):
        """
        The report must include a Full Comparison Table section.
        """
        md = generate_markdown_report(self._report(tmp_path))
        assert "Full Comparison Table" in md

    def test_systems_mentioned(self, tmp_path):
        """
        Each system name should appear somewhere in the report.
        """
        md = generate_markdown_report(self._report(tmp_path))
        assert "nvidia-h100" in md
        assert "amd-mi300x"  in md

    def test_telemetry_section_present_when_available(self, tmp_path):
        """
        The Hardware Telemetry section should be included when any row
        has telemetry_available=True.
        """
        results = [make_result(
            telemetry_available=True, power_mean_w=2400.0, gpu_util_mean_pct=88.0
        )]
        report = self._report(tmp_path, results=results)
        md = generate_markdown_report(report)
        assert "Hardware Telemetry" in md

    def test_telemetry_section_absent_when_unavailable(self, tmp_path):
        """
        The Hardware Telemetry section should be omitted entirely when
        no rows have telemetry data, to keep the report clean.
        """
        results = [make_result(telemetry_available=False)]
        report = self._report(tmp_path, results=results)
        md = generate_markdown_report(report)
        assert "Hardware Telemetry" not in md

    def test_regression_section_present_when_flagged(self, tmp_path):
        """
        The Regressions section should appear when the report has
        regressions flagged.
        """
        report = self._report(tmp_path)
        report.regressions = [{
            "metric": "tok_s", "baseline_value": 500.0, "candidate_value": 300.0,
            "degradation_pct": 40.0, "system": "nvidia-h100",
            "model": "test-model", "backend": "vllm", "concurrency": 16,
        }]
        md = generate_markdown_report(report)
        assert "Regressions" in md

    def test_regression_section_absent_when_none(self, tmp_path):
        """
        The Regressions section should be omitted when no regressions
        were detected, to keep the report clean.
        """
        report = self._report(tmp_path)
        report.regressions = []
        md = generate_markdown_report(report)
        assert "Regressions Detected" not in md

    def test_contains_timestamp(self, tmp_path):
        """
        The report should include a timestamp so the reader knows when
        it was generated.
        """
        md = generate_markdown_report(self._report(tmp_path))
        assert "Generated" in md

    def test_markdown_table_syntax(self, tmp_path):
        """
        The comparison table must use valid GFM markdown table syntax —
        pipe characters and a separator row with dashes.
        """
        md = generate_markdown_report(self._report(tmp_path))
        # GFM tables have separator rows like | --- | --- |
        assert "| -" in md or "|:-" in md or "|-" in md