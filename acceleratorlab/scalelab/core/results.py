"""Pure result-normalization helpers — no Streamlit dependency."""
from __future__ import annotations
from typing import Any, Dict, List


def normalize_results(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in items:
        r = item.get("benchmark_result", item)
        rows.append({
            "system":          r.get("system",          "unknown"),
            "model":           r.get("model",           "unknown"),
            "backend":         r.get("backend",         "unknown"),
            "concurrency":     int(r.get("concurrency",    1)),
            "tok_s":           float(r.get("tok_s",        0)),
            "ttft_ms":         float(r.get("ttft_ms",      0)),
            "mean_latency_ms": float(r.get("mean_latency_ms", 0)),
            "p95_ms":          float(r.get("p95_ms",      0)),
            "success_rate":    float(r.get("success_rate", 0)),
            "requests_ok":     int(r.get("requests_ok",   r.get("raw_count", 0))),
            "duration_s":      float(r.get("duration_s",  0)),
            "traffic_pattern": r.get("traffic_pattern", "steady"),
            "meets_slo":       bool(r.get("meets_slo",   False)),
        })
    return rows
