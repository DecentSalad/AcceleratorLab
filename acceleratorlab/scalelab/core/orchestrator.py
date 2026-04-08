"""
Orchestrator — coordinates the full benchmark pipeline.

Phase 5 improvements:
  - Distributed health-check: polls every node in cluster.hosts
    independently in parallel rather than a single localhost endpoint
  - Quorum gate: waits for all nodes to be ready before sending traffic
  - Topology metadata attached to launch_result for traceability
  - SSH executor now passes key_file and ssh_options from LaunchConfig env
"""
from __future__ import annotations
import concurrent.futures
import shutil
import time
from typing import Any, Dict, List, Optional

import requests

from scalelab.backends.registry import BACKENDS
from scalelab.core.models import Scenario
from scalelab.core.planner import plan_commands
from scalelab.core.traffic import run_openai_compatible_benchmark
from scalelab.executors.local import LocalExecutor
from scalelab.executors.ssh import SSHExecutor
from scalelab.executors.slurm import SlurmExecutor


_HEALTHCHECK_TIMEOUT_S  = 180
_HEALTHCHECK_INTERVAL_S = 5

_VENDOR_TOOLS = {
    "nvidia": "nvidia-smi",
    "amd":    "rocm-smi",
}


# ---------------------------------------------------------------------------
# Vendor validation (unchanged from Phase 2)
# ---------------------------------------------------------------------------

def _validate_vendor(scenario: Scenario) -> Dict[str, Any]:
    vendor = scenario.cluster.accelerator_vendor.lower().strip()
    tool   = _VENDOR_TOOLS.get(vendor)

    if tool is None:
        return {
            "vendor_validation": "skipped",
            "vendor_validation_note": (
                f"No validation tool known for vendor '{vendor}'. "
                f"Supported: {list(_VENDOR_TOOLS.keys())}"
            ),
        }

    tool_found = shutil.which(tool) is not None

    if tool_found:
        return {
            "vendor_validation": "passed",
            "vendor_validation_note": (
                f"'{tool}' found in PATH — declared vendor '{vendor}' "
                f"is consistent with this machine."
            ),
        }

    other_vendor = "amd" if vendor == "nvidia" else "nvidia"
    other_tool   = _VENDOR_TOOLS[other_vendor]
    other_found  = shutil.which(other_tool) is not None

    note = (
        f"WARNING: '{tool}' not found in PATH but scenario declares "
        f"accelerator_vendor='{vendor}'. "
    )
    if other_found:
        note += (
            f"'{other_tool}' IS present — you may have the wrong vendor "
            f"set in your scenario (should it be '{other_vendor}'?). "
        )
    note += (
        "Benchmark will proceed but hardware telemetry will be unavailable. "
        "If running via SSH/Slurm, this warning is expected on the head node."
    )
    return {"vendor_validation": "warning", "vendor_validation_note": note}


# ---------------------------------------------------------------------------
# Single-node health-check (used by local executor path)
# ---------------------------------------------------------------------------

def _wait_for_server(url: str, timeout_s: int = _HEALTHCHECK_TIMEOUT_S) -> bool:
    """Poll a single URL until HTTP 200 or timeout."""
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


# ---------------------------------------------------------------------------
# Distributed health-check (Phase 5) — polls every node independently
# ---------------------------------------------------------------------------

def _check_one_host(host: str, port: int, path: str, timeout_s: int) -> Dict[str, Any]:
    """
    Poll a single host's health endpoint until HTTP 200 or timeout.
    Returns a structured result so the caller knows which nodes are ready.
    """
    url      = f"http://{host}:{port}{path}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.ok:
                return {"host": host, "ready": True, "url": url}
        except Exception:
            pass
        time.sleep(_HEALTHCHECK_INTERVAL_S)
    return {"host": host, "ready": False, "url": url,
            "error": f"did not respond within {timeout_s}s"}


def _wait_for_all_nodes(
    hosts: List[str],
    port: int = 8000,
    health_path: str = "/health",
    timeout_s: int = _HEALTHCHECK_TIMEOUT_S,
    quorum: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Poll every node's health endpoint in parallel and wait for quorum.

    Parameters
    ----------
    hosts
        List of hostnames to check.
    port
        Port the inference server is listening on.
    health_path
        URL path for the health endpoint (e.g. "/health").
    timeout_s
        Maximum seconds to wait per node.
    quorum
        Minimum number of nodes that must be ready before proceeding.
        Defaults to all nodes (strict quorum).

    Returns
    -------
    Dict with:
        all_ready      — True if quorum was reached
        nodes_ready    — count of ready nodes
        nodes_failed   — count of nodes that timed out
        quorum_reached — True if nodes_ready >= quorum
        details        — per-host result list
    """
    if not hosts:
        # No hosts declared — single-node mode, skip distributed check
        return {"all_ready": True, "nodes_ready": 0, "nodes_failed": 0,
                "quorum_reached": True, "details": []}

    required = quorum if quorum is not None else len(hosts)

    details = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        futs = {
            pool.submit(_check_one_host, host, port, health_path, timeout_s): host
            for host in hosts
        }
        for fut in concurrent.futures.as_completed(futs):
            details.append(fut.result())

    nodes_ready  = sum(1 for d in details if d["ready"])
    nodes_failed = len(details) - nodes_ready

    return {
        "all_ready":      nodes_ready == len(hosts),
        "nodes_ready":    nodes_ready,
        "nodes_failed":   nodes_failed,
        "quorum_reached": nodes_ready >= required,
        "details":        details,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def execute_scenario(scenario: Scenario, launch_servers: bool = False) -> Dict[str, Any]:
    commands     = plan_commands(scenario)
    launch_result: Dict[str, Any] = {
        "executor": scenario.launch.executor,
        "skipped":  True,
    }

    # Attach topology metadata to every result for traceability
    topo = scenario.cluster.topology
    launch_result["topology"] = {
        "rack_id":                    topo.rack_id,
        "switch_group":               topo.switch_group,
        "nvlink_domain":              topo.nvlink_domain,
        "nodes_per_switch":           topo.nodes_per_switch,
        "inter_node_bandwidth_gbps":  topo.inter_node_bandwidth_gbps,
    }

    # Validate declared vendor against detectable hardware on this machine
    launch_result.update(_validate_vendor(scenario))

    if launch_servers and commands:
        executor = scenario.launch.executor

        if executor == "local":
            launch_result.update(
                LocalExecutor().launch(commands, scenario.launch.env)
            )

        elif executor == "ssh":
            # Pull optional SSH config from scenario launch env
            key_file   = scenario.launch.env.pop("SSH_KEY_FILE", None)
            ssh_opts_raw = scenario.launch.env.pop("SSH_OPTIONS", "")
            ssh_options = {}
            if ssh_opts_raw:
                for pair in ssh_opts_raw.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        ssh_options[k.strip()] = v.strip()

            launch_result.update(
                SSHExecutor(
                    hosts=scenario.cluster.hosts,
                    user=scenario.cluster.ssh_user,
                    key_file=key_file,
                    ssh_options=ssh_options,
                ).launch(commands, scenario.launch.env)
            )

        elif executor == "slurm":
            launch_result.update(
                SlurmExecutor(
                    partition=scenario.cluster.slurm_partition,
                    account=scenario.cluster.slurm_account,
                    nodes=scenario.cluster.nodes,
                    gpus_per_node=scenario.cluster.accelerators_per_node,
                ).launch(commands, scenario.launch.env)
            )

        else:
            raise ValueError(f"Unsupported executor: {executor}")

        # ── Health-check: distributed for multi-node, single for local ──
        adapter = BACKENDS.get(scenario.workload.backend)
        if adapter:
            hosts = scenario.cluster.hosts

            if hosts and executor == "ssh":
                # Phase 5: poll every declared host independently
                health_check = _wait_for_all_nodes(
                    hosts=hosts,
                    port=8000,
                    health_path="/health",
                    timeout_s=_HEALTHCHECK_TIMEOUT_S,
                )
                launch_result["health_check"] = health_check
                launch_result["server_ready"] = health_check["quorum_reached"]
                if not health_check["quorum_reached"]:
                    failed = health_check["nodes_failed"]
                    total  = len(hosts)
                    launch_result["warning"] = (
                        f"{failed}/{total} nodes did not respond within "
                        f"{_HEALTHCHECK_TIMEOUT_S}s — benchmark may fail."
                    )
            else:
                # Single-node path: poll localhost as before
                health_url = adapter.build_healthcheck_url(scenario)
                ready      = _wait_for_server(health_url)
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