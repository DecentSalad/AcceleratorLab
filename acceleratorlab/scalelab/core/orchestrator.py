from __future__ import annotations
import time
from typing import Any, Dict

import requests

from scalelab.backends.registry import BACKENDS
from scalelab.core.models import Scenario
from scalelab.core.planner import plan_commands
from scalelab.core.traffic import run_openai_compatible_benchmark
from scalelab.executors.local import LocalExecutor
from scalelab.executors.ssh import SSHExecutor
from scalelab.executors.slurm import SlurmExecutor


_HEALTHCHECK_TIMEOUT_S  = 180   # max seconds to wait for server readiness
_HEALTHCHECK_INTERVAL_S = 5     # seconds between probes


def _wait_for_server(url: str, timeout_s: int = _HEALTHCHECK_TIMEOUT_S) -> bool:
    """Poll url until HTTP 200 or timeout. Returns True if server became ready."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(_HEALTHCHECK_INTERVAL_S)
    return False


def execute_scenario(scenario: Scenario, launch_servers: bool = False) -> Dict[str, Any]:
    commands = plan_commands(scenario)
    launch_result: Dict[str, Any] = {"executor": scenario.launch.executor, "skipped": True}

    if launch_servers and commands:
        if scenario.launch.executor == "local":
            launch_result = LocalExecutor().launch(commands, scenario.launch.env)
        elif scenario.launch.executor == "ssh":
            launch_result = SSHExecutor(
                hosts=scenario.cluster.hosts,
                user=scenario.cluster.ssh_user,
            ).launch(commands, scenario.launch.env)
        elif scenario.launch.executor == "slurm":
            launch_result = SlurmExecutor(
                partition=scenario.cluster.slurm_partition,
                account=scenario.cluster.slurm_account,
                nodes=scenario.cluster.nodes,
                gpus_per_node=scenario.cluster.accelerators_per_node,
            ).launch(commands, scenario.launch.env)
        else:
            raise ValueError(f"Unsupported executor: {scenario.launch.executor}")

        # Health-check: wait for the server to become ready before sending traffic
        adapter = BACKENDS.get(scenario.workload.backend)
        if adapter:
            health_url = adapter.build_healthcheck_url(scenario)
            ready = _wait_for_server(health_url)
            launch_result["server_ready"] = ready
            if not ready:
                launch_result["warning"] = (
                    f"Server did not respond at {health_url} "
                    f"within {_HEALTHCHECK_TIMEOUT_S}s — benchmark may fail."
                )

    benchmark_result = run_openai_compatible_benchmark(scenario)
    return {
        "scenario":         scenario.to_dict(),
        "launch_result":    launch_result,
        "benchmark_result": benchmark_result,
    }
