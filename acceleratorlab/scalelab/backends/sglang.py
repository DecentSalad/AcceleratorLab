from __future__ import annotations
from typing import List

from scalelab.backends.base import BackendAdapter
from scalelab.core.models import Scenario


class SGLangAdapter(BackendAdapter):
    name = "sglang"

    def build_server_command(self, scenario: Scenario) -> List[str]:
        l = scenario.launch
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", scenario.workload.model,
            "--host",       "0.0.0.0",
            "--port",       "8000",
            "--tp-size",    str(l.tensor_parallel),
            "--pp-size",    str(l.pipeline_parallel),
        ]
        cmd += l.extra_args
        return cmd

    def build_healthcheck_url(self, scenario: Scenario) -> str:
        return "http://127.0.0.1:8000/health"
