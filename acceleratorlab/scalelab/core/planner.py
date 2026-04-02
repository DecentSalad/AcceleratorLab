from __future__ import annotations
from typing import List

from scalelab.backends.registry import BACKENDS
from scalelab.core.models import Scenario


def plan_commands(scenario: Scenario) -> List[List[str]]:
    adapter = BACKENDS[scenario.workload.backend]

    if scenario.workload.backend == "openai-compat":
        return []

    server_cmd = adapter.build_server_command(scenario)

    if scenario.cluster.nodes <= 1:
        return [server_cmd]

    # Multi-node: determine master address from host list or fall back to first hostname
    hosts = scenario.cluster.hosts
    master_addr = hosts[0] if hosts else "node0"
    master_port = "29500"

    commands = []
    for rank in range(scenario.cluster.nodes):
        node_cmd = server_cmd + [
            "--node-rank",   str(rank),
            "--master-addr", master_addr,
            "--master-port", master_port,
        ]
        commands.append(node_cmd)

    return commands
