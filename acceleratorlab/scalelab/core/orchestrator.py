from __future__ import annotations
import shutil
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

# Maps declared vendor string to the CLI tool that should be present
_VENDOR_TOOLS = {
    "nvidia": "nvidia-smi",
    "amd":    "rocm-smi",
}


# ---------------------------------------------------------------------------
# Vendor validation
# ---------------------------------------------------------------------------

def _validate_vendor(scenario: Scenario) -> Dict[str, Any]:
    """
    Check whether the declared accelerator_vendor matches what is detectable
    on the current machine.

    Returns a dict that is merged into launch_result so the discrepancy (if
    any) is recorded in the saved project file without blocking the run.

    We warn rather than abort because:
      - The benchmark tool may be running on a head node that doesn't have
        GPUs but is submitting jobs to GPU nodes via Slurm or SSH.
      - The user may have nvidia-smi installed but CUDA_VISIBLE_DEVICES=""
        or similar configurations.
    """
    vendor = scenario.cluster.accelerator_vendor.lower().strip()
    tool = _VENDOR_TOOLS.get(vendor)

    if tool is None:
        # Unknown vendor — we have no tool to check against, skip validation
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
    else:
        # Check if the OTHER vendor's tool is present — useful diagnostic
        other_vendor = "amd" if vendor == "nvidia" else "nvidia"
        other_tool = _VENDOR_TOOLS[other_vendor]
        other_found = shutil.which(other_tool) is not None

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

        return {
            "vendor_validation": "warning",
            "vendor_validation_note": note,
        }


# ---------------------------------------------------------------------------
# Health-check
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def execute_scenario(scenario: Scenario, launch_servers: bool = False) -> Dict[str, Any]:
    commands = plan_commands(scenario)
    launch_result: Dict[str, Any] = {"executor": scenario.launch.executor, "skipped": True}

    # Validate declared vendor against detectable hardware on this machine
    launch_result.update(_validate_vendor(scenario))

    if launch_servers and commands:
        if scenario.launch.executor == "local":
            launch_result.update(LocalExecutor().launch(commands, scenario.launch.env))
        elif scenario.launch.executor == "ssh":
            launch_result.update(SSHExecutor(
                hosts=scenario.cluster.hosts,
                user=scenario.cluster.ssh_user,
            ).launch(commands, scenario.launch.env))
        elif scenario.launch.executor == "slurm":
            launch_result.update(SlurmExecutor(
                partition=scenario.cluster.slurm_partition,
                account=scenario.cluster.slurm_account,
                nodes=scenario.cluster.nodes,
                gpus_per_node=scenario.cluster.accelerators_per_node,
            ).launch(commands, scenario.launch.env))
        else:
            raise ValueError(f"Unsupported executor: {scenario.launch.executor}")

        # Health-check: wait for server to become ready before sending traffic
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