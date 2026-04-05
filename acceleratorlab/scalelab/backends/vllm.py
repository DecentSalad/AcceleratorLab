from __future__ import annotations
from typing import List
from scalelab.backends.base import BackendAdapter
from scalelab.core.models import Scenario


# AMD architectures that support bfloat16 natively.
# All modern MI-series GPUs (MI300X, MI325X, MI355X) prefer bfloat16.
# Older MI200-series can struggle with bfloat16 — float16 is safer there.
_AMD_BFLOAT16_ARCHS = {
    "mi300x", "mi300a", "mi325x", "mi350x", "mi355x",
}


def _amd_dtype(arch: str) -> str:
    """Return the recommended dtype flag value for a given AMD arch string."""
    arch_lower = arch.lower().strip()
    if arch_lower in _AMD_BFLOAT16_ARCHS or arch_lower.startswith("mi3"):
        return "bfloat16"
    # MI200-series and older default to float16 for stability
    return "float16"


class VLLMAdapter(BackendAdapter):
    name = "vllm"

    def build_server_command(self, scenario: Scenario) -> List[str]:
        w = scenario.workload
        l = scenario.launch
        c = scenario.cluster

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model",                  w.model,
            "--host",                   "0.0.0.0",
            "--port",                   "8000",
            "--tensor-parallel-size",   str(l.tensor_parallel),
            "--pipeline-parallel-size", str(l.pipeline_parallel),
        ]

        if l.model_cache_dir:
            cmd += ["--download-dir", l.model_cache_dir]

        # ── Vendor-specific flags ─────────────────────────────────────────────
        vendor = c.accelerator_vendor.lower().strip()

        if vendor == "amd":
            # ROCm backend — required, vLLM defaults to CUDA without this
            cmd += ["--device", "rocm"]
            # Explicit dtype — ROCm auto-detection is less reliable than CUDA
            cmd += ["--dtype", _amd_dtype(c.accelerator_arch)]

        # NVIDIA: no extra flags — vLLM's CUDA defaults are correct

        # User-supplied extra args applied last so they can override anything above
        cmd += l.extra_args
        return cmd

    def build_healthcheck_url(self, scenario: Scenario) -> str:
        return "http://127.0.0.1:8000/health"