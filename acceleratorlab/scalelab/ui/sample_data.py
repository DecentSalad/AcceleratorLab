def load_demo_runs():
    return [
        {"benchmark_result": {"system": "nvidia-b200",   "model": "Llama-3.1-70B", "backend": "vllm",   "concurrency": 32, "tok_s": 6550.0, "ttft_ms": 620,  "mean_latency_ms": 1800, "p95_ms": 2400, "success_rate": 1.0,  "meets_slo": True,  "requests_ok": 200, "duration_s": 62.3, "traffic_pattern": "steady"}},
        {"benchmark_result": {"system": "nvidia-gb200",  "model": "Llama-3.1-70B", "backend": "vllm",   "concurrency": 32, "tok_s": 7200.0, "ttft_ms": 530,  "mean_latency_ms": 1600, "p95_ms": 2100, "success_rate": 1.0,  "meets_slo": True,  "requests_ok": 200, "duration_s": 55.8, "traffic_pattern": "steady"}},
        {"benchmark_result": {"system": "amd-mi325x",    "model": "Llama-3.1-70B", "backend": "sglang", "concurrency": 32, "tok_s": 5900.0, "ttft_ms": 760,  "mean_latency_ms": 2100, "p95_ms": 2800, "success_rate": 1.0,  "meets_slo": True,  "requests_ok": 200, "duration_s": 70.1, "traffic_pattern": "steady"}},
        {"benchmark_result": {"system": "amd-mi355x",    "model": "Llama-3.1-70B", "backend": "sglang", "concurrency": 64, "tok_s": 8000.0, "ttft_ms": 980,  "mean_latency_ms": 3100, "p95_ms": 4900, "success_rate": 0.99, "meets_slo": True,  "requests_ok": 198, "duration_s": 80.4, "traffic_pattern": "burst"}},
        {"benchmark_result": {"system": "nvidia-h100",   "model": "Llama-3.1-70B", "backend": "vllm",   "concurrency": 48, "tok_s": 5100.0, "ttft_ms": 840,  "mean_latency_ms": 2400, "p95_ms": 3800, "success_rate": 1.0,  "meets_slo": True,  "requests_ok": 200, "duration_s": 75.2, "traffic_pattern": "steady"}},
    ]
