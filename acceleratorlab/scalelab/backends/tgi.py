from __future__ import annotations
from typing import List

from scalelab.backends.base import BackendAdapter
from scalelab.core.models import Scenario


class TGIAdapter(BackendAdapter):
    name = "tgi"

    def build_server_command(self, scenario: Scenario) -> List[str]:
        l = scenario.launch
        cmd = [
            "text-generation-launcher",
            "--model-id",  scenario.workload.model,
            "--hostname",  "0.0.0.0",
            "--port",      "8000",
            # TGI uses --num-shard (not --tensor-parallel-size)
            "--num-shard", str(l.tensor_parallel),
        ]
        cmd += l.extra_args
        return cmd

    def build_healthcheck_url(self, scenario: Scenario) -> str:
        return "http://127.0.0.1:8000/health"
