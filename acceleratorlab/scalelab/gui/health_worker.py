"""Quick pre-flight health check worker — used before launching a benchmark."""
from __future__ import annotations
import requests
from PyQt6.QtCore import QThread, pyqtSignal


class HealthCheckWorker(QThread):
    result = pyqtSignal(bool)   # True = server reachable

    def __init__(self, endpoint: str) -> None:
        super().__init__()
        self._endpoint = endpoint.rstrip("/")

    def run(self) -> None:
        base = self._endpoint.replace("/v1", "")
        for path in ["/health", "/v1/models"]:
            try:
                r = requests.get(base + path, timeout=5)
                if r.ok:
                    self.result.emit(True)
                    return
            except Exception:
                pass
        self.result.emit(False)
