from __future__ import annotations
from typing import List
from scalelab.backends.base import BackendAdapter
from scalelab.core.models import Scenario


def _amd_dtype(arch: str) -> str:
    """Mirror of the vLLM dtype helper — same logic, same AMD arch strings."""
    arch_lower = arch.lower().strip()
    if arch_lower.startswith("mi3"):
        return "bfloat16"
    return "float16"


class SGLangAdapter(BackendAdapter):
    name = "sglang"

    def build_server_command(self, scenario: Scenario) -> List[str]:
        l = scenario.launch
        c = scenario.cluster

        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", scenario.workload.model,
            "--host",       "0.0.0.0",
            "--port",       "8000",
            "--tp-size",    str(l.tensor_parallel),
            "--pp-size",    str(l.pipeline_parallel),
        ]

        if l.model_cache_dir:
            cmd += ["--model-cache-dir", l.model_cache_dir]

        # ── Vendor-specific flags ─────────────────────────────────────────────
        vendor = c.accelerator_vendor.lower().strip()

        if vendor == "amd":
            # SGLang uses the same --device flag convention as vLLM
            cmd += ["--device", "rocm"]
            cmd += ["--dtype", _amd_dtype(c.accelerator_arch)]

        # NVIDIA: SGLang defaults to CUDA — no extra flags needed

        cmd += l.extra_args
        return cmd

    def build_healthcheck_url(self, scenario: Scenario) -> str:
        return "http://127.0.0.1:8000/health"