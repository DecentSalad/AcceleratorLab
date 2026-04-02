from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class Executor(ABC):
    name: str = "base"
    @abstractmethod
    def launch(self, commands: List[List[str]], env: Dict[str, str] | None = None) -> Dict[str, Any]:
        raise NotImplementedError
