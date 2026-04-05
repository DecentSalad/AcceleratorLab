from __future__ import annotations
from typing import List
from scalelab.backends.base import BackendAdapter
from scalelab.core.models import Scenario


def _amd_dtype(arch: str) -> str:
    arch_lower = arch.lower().strip()
    if arch_lower.startswith("mi3"):
        return "bfloat16"
    return "float16"


class TGIAdapter(BackendAdapter):
    name = "tgi"

    def build_server_command(self, scenario: Scenario) -> List[str]:
        l = scenario.launch
        c = scenario.cluster

        cmd = [
            "text-generation-launcher",
            "--model-id",  scenario.workload.model,
            "--hostname",  "0.0.0.0",
            "--port",      "8000",
            # TGI uses --num-shard, not --tensor-parallel-size
            "--num-shard", str(l.tensor_parallel),
        ]

        if l.model_cache_dir:
            cmd += ["--huggingface-hub-cache", l.model_cache_dir]

        # ── Vendor-specific flags ─────────────────────────────────────────────
        vendor = c.accelerator_vendor.lower().strip()

        if vendor == "amd":
            # Explicit dtype — prevents TGI from defaulting to float16 on ROCm,
            # which is suboptimal for MI300X and newer
            cmd += ["--dtype", _amd_dtype(c.accelerator_arch)]
            # CUDA graphs can cause hangs on some ROCm versions — disable by default
            cmd += ["--disable-cuda-graphs"]

        # NVIDIA: TGI's CUDA defaults are correct — no extra flags needed

        cmd += l.extra_args
        return cmd

    def build_healthcheck_url(self, scenario: Scenario) -> str:
        return "http://127.0.0.1:8000/health"