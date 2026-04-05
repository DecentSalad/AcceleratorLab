"""
TensorRT-LLM backend adapter.

Uses the `trtllm-serve` CLI introduced in TRT-LLM 0.12+.  This command
starts an OpenAI-compatible HTTP server directly from a HuggingFace model ID
or a pre-compiled engine directory — no manual trtllm-build step required
for supported models.

Install:  pip install tensorrt-llm  (requires NVIDIA GPU + CUDA 12.x)
Docs:     https://github.com/NVIDIA/TensorRT-LLM

TensorRT-LLM is NVIDIA-only.  If accelerator_vendor is not "nvidia" this
adapter raises a clear error rather than generating a command that will fail
silently on non-CUDA hardware.
"""
from __future__ import annotations
from typing import List
from scalelab.backends.base import BackendAdapter
from scalelab.core.models import Scenario


class TensorRTLLMAdapter(BackendAdapter):
    name = "tensorrt-llm"

    def build_server_command(self, scenario: Scenario) -> List[str]:
        w = scenario.workload
        l = scenario.launch
        c = scenario.cluster

        # TensorRT-LLM only runs on NVIDIA hardware
        vendor = c.accelerator_vendor.lower().strip()
        if vendor != "nvidia":
            raise ValueError(
                f"TensorRT-LLM requires an NVIDIA GPU "
                f"(accelerator_vendor: '{c.accelerator_vendor}'). "
                f"Use vllm or sglang for AMD hardware."
            )

        # trtllm-serve accepts either a HuggingFace model ID or a path to a
        # pre-compiled TensorRT engine directory.
        cmd = [
            "trtllm-serve",
            w.model,                         # HF model ID or engine dir path
            "--host",        "0.0.0.0",
            "--port",        "8000",
            "--tp_size",     str(l.tensor_parallel),
            "--pp_size",     str(l.pipeline_parallel),
            "--max_batch_size", "256",        # reasonable default; override via extra_args
        ]

        # Engine cache / build output directory
        # trtllm-serve uses this as both the source for pre-built engines
        # and the destination when building from a HF model ID
        if l.model_cache_dir:
            cmd += ["--model_cache_dir", l.model_cache_dir]

        # Architecture-specific precision defaults
        # H100, B200, GB200 support fp8 natively — best throughput
        # A100, H800 support bfloat16 — use that instead of fp8
        arch_lower = c.accelerator_arch.lower().strip()
        if arch_lower in {"h100", "h100_sxm", "h100_nvl"}:
            cmd += ["--kv_cache_dtype", "fp8"]
        elif arch_lower in {"b200", "gb200", "b100"}:
            cmd += ["--kv_cache_dtype", "fp8"]
        # A100 and others: omit kv_cache_dtype and let TRT-LLM decide

        # User-supplied extra args override anything above
        cmd += l.extra_args
        return cmd

    def build_healthcheck_url(self, scenario: Scenario) -> str:
        return "http://127.0.0.1:8000/health"