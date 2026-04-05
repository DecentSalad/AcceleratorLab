from __future__ import annotations
import concurrent.futures
import json
import math
import statistics
import time
from typing import Any, Dict, List

import requests

from scalelab.core.models import Scenario
from scalelab.core.telemetry import TelemetryCollector


# ---------------------------------------------------------------------------
# Single-request worker — streaming mode for real TTFT measurement
# ---------------------------------------------------------------------------

def _one_request(
    url: str,
    model: str,
    prompt_tokens: int,
    output_tokens: int,
    api_key: str,
) -> Dict[str, Any]:
    prompt = "hello " * max(1, prompt_tokens // 2)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "stream": True,  # streaming required for real TTFT measurement
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    generated_tokens = 0

    try:
        with requests.post(
            url.rstrip("/") + "/chat/completions",
            json=payload,
            headers=headers,
            stream=True,
            timeout=300,
        ) as r:
            if not r.ok:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "ok": False,
                    "latency_ms": elapsed_ms,
                    "ttft_ms": elapsed_ms,
                    "generated_tokens": 0,
                }

            usage: Dict[str, Any] = {}
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Capture time-to-first-token on the first content chunk
                if ttft_ms is None:
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            ttft_ms = (time.perf_counter() - t0) * 1000.0

                # Accumulate usage from final chunk
                if chunk.get("usage"):
                    usage = chunk["usage"]

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Prefer server-reported completion_tokens; fall back to 0
            generated_tokens = (
                usage.get("completion_tokens")
                or usage.get("generated_tokens")
                or 0
            )

            if ttft_ms is None:
                ttft_ms = elapsed_ms

            return {
                "ok": True,
                "latency_ms": elapsed_ms,
                "ttft_ms": ttft_ms,
                "generated_tokens": generated_tokens,
            }

    except Exception:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "ok": False,
            "latency_ms": elapsed_ms,
            "ttft_ms": elapsed_ms,
            "generated_tokens": 0,
        }


# ---------------------------------------------------------------------------
# Traffic pattern: steady (rate-limited, concurrency-bounded)
# ---------------------------------------------------------------------------

def _steady_rate_requests(
    executor: concurrent.futures.ThreadPoolExecutor,
    url: str, model: str, prompt_tokens: int, output_tokens: int, api_key: str,
    total_requests: int, concurrency: int, deadline: float,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    in_flight = 0
    submitted = 0
    futs: List[concurrent.futures.Future] = []

    def _maybe_submit():
        nonlocal in_flight, submitted
        while in_flight < concurrency and submitted < total_requests and time.perf_counter() < deadline:
            futs.append(executor.submit(
                _one_request, url, model, prompt_tokens, output_tokens, api_key
            ))
            submitted += 1
            in_flight += 1

    _maybe_submit()
    while futs:
        done, _ = concurrent.futures.wait(futs, timeout=1.0,
                                           return_when=concurrent.futures.FIRST_COMPLETED)
        for f in done:
            results.append(f.result())
            in_flight -= 1
            futs.remove(f)
        if time.perf_counter() >= deadline:
            break
        _maybe_submit()
    return results


# ---------------------------------------------------------------------------
# Traffic pattern: burst (all requests fired simultaneously)
# ---------------------------------------------------------------------------

def _burst_requests(
    executor: concurrent.futures.ThreadPoolExecutor,
    url: str, model: str, prompt_tokens: int, output_tokens: int, api_key: str,
    total_requests: int, deadline: float,
) -> List[Dict[str, Any]]:
    futs = [
        executor.submit(_one_request, url, model, prompt_tokens, output_tokens, api_key)
        for _ in range(total_requests)
        if time.perf_counter() < deadline
    ]
    results = []
    for f in concurrent.futures.as_completed(futs):
        results.append(f.result())
        if time.perf_counter() >= deadline:
            break
    return results


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------

def run_openai_compatible_benchmark(scenario: Scenario) -> Dict[str, Any]:
    w = scenario.workload
    deadline = time.perf_counter() + w.duration_s
    start_wall = time.perf_counter()

    # Start hardware telemetry — runs in background for the full benchmark window
    telemetry = TelemetryCollector(vendor=scenario.cluster.accelerator_vendor)
    telemetry.start()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=w.concurrency) as pool:
            pattern = w.traffic_pattern.lower()
            if pattern == "burst":
                results = _burst_requests(
                    pool, w.endpoint, w.model,
                    w.prompt_tokens, w.output_tokens, w.api_key,
                    w.requests, deadline,
                )
            else:
                # "steady" is the default; unknown patterns fall back to steady
                results = _steady_rate_requests(
                    pool, w.endpoint, w.model,
                    w.prompt_tokens, w.output_tokens, w.api_key,
                    w.requests, w.concurrency, deadline,
                )
    finally:
        # Always stop the collector — even if the benchmark raised an exception
        hw = telemetry.stop()

    total_s   = max(0.001, time.perf_counter() - start_wall)
    oks       = [x for x in results if x["ok"]]
    lats      = [x["latency_ms"]       for x in oks] or [0.0]
    ttfts     = [x["ttft_ms"]          for x in oks] or [0.0]
    toks      = [x["generated_tokens"] for x in oks] or [0]

    sorted_lats = sorted(lats)
    p95_idx     = min(len(sorted_lats) - 1, math.ceil(len(sorted_lats) * 0.95) - 1)
    p95         = sorted_lats[p95_idx]
    ttft        = statistics.mean(ttfts)
    mean_lat    = statistics.mean(lats)
    tok_s       = float(sum(toks) / total_s)

    result = {
        "system":            f"{scenario.cluster.accelerator_vendor}-{scenario.cluster.accelerator_arch}",
        "model":             w.model,
        "backend":           w.backend,
        "concurrency":       w.concurrency,
        "requests_sent":     len(results),
        "requests_ok":       len(oks),
        "duration_s":        round(total_s, 2),
        "tok_s":             round(tok_s, 2),
        "ttft_ms":           round(ttft, 2),
        "mean_latency_ms":   round(mean_lat, 2),
        "p95_ms":            round(p95, 2),
        "success_rate":      round(len(oks) / max(1, len(results)), 4),
        "meets_slo":         (ttft <= w.target_ttft_ms and p95 <= w.target_p95_ms),
        "traffic_pattern":   w.traffic_pattern,
    }

    # Merge telemetry metrics into the result dict
    result.update(hw.to_dict())

    return result