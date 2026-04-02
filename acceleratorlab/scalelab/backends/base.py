from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from scalelab.core.models import Scenario

class BackendAdapter(ABC):
    name: str = "base"
    @abstractmethod
    def build_server_command(self, scenario: Scenario) -> List[str]:
        raise NotImplementedError
    @abstractmethod
    def build_healthcheck_url(self, scenario: Scenario) -> str:
        raise NotImplementedError
