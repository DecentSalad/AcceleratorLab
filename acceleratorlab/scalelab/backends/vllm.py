from __future__ import annotations
from typing import List
from scalelab.backends.base import BackendAdapter
from scalelab.core.models import Scenario

class VLLMAdapter(BackendAdapter):
    name = "vllm"
    def build_server_command(self, scenario: Scenario) -> List[str]:
        w = scenario.workload
        l = scenario.launch
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", w.model,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--tensor-parallel-size", str(l.tensor_parallel),
            "--pipeline-parallel-size", str(l.pipeline_parallel),
        ]
        if l.model_cache_dir:
            cmd += ["--download-dir", l.model_cache_dir]
        cmd += l.extra_args
        return cmd
    def build_healthcheck_url(self, scenario: Scenario) -> str:
        return "http://127.0.0.1:8000/health"
