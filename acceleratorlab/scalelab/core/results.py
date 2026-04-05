"""Pure result-normalization helpers — no Streamlit dependency."""
from __future__ import annotations
from typing import Any, Dict, List


def normalize_results(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in items:
        r = item.get("benchmark_result", item)
        rows.append({
            # ── Software metrics (always present) ────────────────────────────
            "system":            r.get("system",          "unknown"),
            "model":             r.get("model",           "unknown"),
            "backend":           r.get("backend",         "unknown"),
            "concurrency":       int(r.get("concurrency",    1)),
            "tok_s":             float(r.get("tok_s",        0)),
            "ttft_ms":           float(r.get("ttft_ms",      0)),
            "mean_latency_ms":   float(r.get("mean_latency_ms", 0)),
            "p95_ms":            float(r.get("p95_ms",      0)),
            "success_rate":      float(r.get("success_rate", 0)),
            "requests_ok":       int(r.get("requests_ok",   r.get("raw_count", 0))),
            "duration_s":        float(r.get("duration_s",  0)),
            "traffic_pattern":   r.get("traffic_pattern",  "steady"),
            "meets_slo":         bool(r.get("meets_slo",    False)),

            # ── Hardware telemetry (present when nvidia-smi / rocm-smi available) ─
            "telemetry_available": bool(r.get("telemetry_available",  False)),
            "telemetry_vendor":    r.get("telemetry_vendor",          "unknown"),
            "gpu_count":           int(r.get("gpu_count",             0)),
            "telemetry_samples":   int(r.get("telemetry_samples",     0)),
            "telemetry_error":     r.get("telemetry_error",           ""),
            "gpu_util_mean_pct":   float(r.get("gpu_util_mean_pct",   0)),
            "gpu_util_peak_pct":   float(r.get("gpu_util_peak_pct",   0)),
            "vram_used_mean_gb":   float(r.get("vram_used_mean_gb",   0)),
            "vram_used_peak_gb":   float(r.get("vram_used_peak_gb",   0)),
            "vram_total_gb":       float(r.get("vram_total_gb",       0)),
            "power_mean_w":        float(r.get("power_mean_w",        0)),
            "power_peak_w":        float(r.get("power_peak_w",        0)),
            "temp_mean_c":         float(r.get("temp_mean_c",         0)),
            "temp_peak_c":         float(r.get("temp_peak_c",         0)),

            # ── Derived efficiency metric (0 if telemetry unavailable) ────────
            # tok/s per watt — the primary cost-efficiency signal
            "tok_s_per_watt":      round(
                float(r.get("tok_s", 0)) / float(r.get("power_mean_w", 0))
                if float(r.get("power_mean_w", 0)) > 0 else 0.0,
                4
            ),
        })
    return rows