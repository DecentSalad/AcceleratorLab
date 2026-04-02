from __future__ import annotations
from typing import List
from scalelab.backends.base import BackendAdapter
from scalelab.core.models import Scenario

class OpenAICompatAdapter(BackendAdapter):
    name = "openai-compat"
    def build_server_command(self, scenario: Scenario) -> List[str]:
        return []
    def build_healthcheck_url(self, scenario: Scenario) -> str:
        endpoint = scenario.workload.endpoint.rstrip("/")
        return endpoint.replace("/v1", "") + "/health"
