from __future__ import annotations
from PyQt6.QtCore import QThread, pyqtSignal
from scalelab.core.models import Scenario
from scalelab.core.orchestrator import execute_scenario


class BenchmarkWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, scenario: dict, launch_servers: bool) -> None:
        super().__init__()
        self._scenario_dict  = scenario
        self._launch_servers = launch_servers

    def run(self) -> None:
        try:
            scenario = Scenario.from_dict(self._scenario_dict)
            result   = execute_scenario(scenario, launch_servers=self._launch_servers)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
