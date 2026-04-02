from __future__ import annotations
from typing import List
from scalelab.backends.base import BackendAdapter
from scalelab.core.models import Scenario

class TensorRTLLMAdapter(BackendAdapter):
    name = "tensorrt-llm"
    def build_server_command(self, scenario: Scenario) -> List[str]:
        return ["bash", "-lc", "echo 'Wire your TensorRT-LLM launch command here' && sleep 2"]
    def build_healthcheck_url(self, scenario: Scenario) -> str:
        return "http://127.0.0.1:8000/health"
