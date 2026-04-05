"""
Sweep automation — run a benchmark across a grid of parameter combinations.

A sweep takes a base Scenario and a set of parameter ranges, then generates
every combination (the cartesian product) and runs each one as a separate
benchmark. The results are collected into a SweepResult for comparison.

Typical use case: characterise how an accelerator performs as concurrency,
prompt length, and output length vary — the output is a performance surface
rather than a single data point.

Usage (Python)
--------------
    from scalelab.core.sweep import SweepConfig, run_sweep
    from scalelab.core.io import load_scenario

    base = load_scenario("examples/scenario_local.yaml")
    config = SweepConfig(
        concurrency_levels=[1, 4, 8, 16, 32, 64],
        prompt_tokens_values=[512, 1024, 2048],
        output_tokens_values=[128, 256],
    )
    result = run_sweep(base, config)
    print(f"Completed {result.completed_runs}/{result.total_runs} runs")

Usage (CLI)
-----------
    python -m scalelab.cli.run --sweep examples/sweep_local.yaml
"""
from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from scalelab.core.models import Scenario
from scalelab.core.orchestrator import execute_scenario


# ---------------------------------------------------------------------------
# SweepConfig — what parameter ranges to sweep across
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    """
    Defines the parameter ranges for a sweep run.

    Each field is a list of values to try. The sweep generates the full
    cartesian product of all combinations — so 3 concurrency levels ×
    2 prompt shapes × 2 output shapes = 12 benchmark runs.

    Fields
    ------
    name
        Human-readable label for this sweep, used in saved results.
    concurrency_levels
        List of concurrency values to test. Each value sets how many
        simultaneous requests are in flight during that run.
    prompt_tokens_values
        List of prompt token counts to test. Larger prompts exercise
        the prefill phase more heavily.
    output_tokens_values
        List of output token counts to test. Larger outputs exercise
        the decode phase and affect throughput more than latency.
    models
        Optional list of model IDs to sweep across. If empty the base
        scenario's model is used for every run.
    backends
        Optional list of backend names to sweep across. If empty the
        base scenario's backend is used for every run.
    """
    name: str = "sweep"
    concurrency_levels: List[int]  = field(default_factory=lambda: [1, 4, 8, 16, 32, 64])
    prompt_tokens_values: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    output_tokens_values: List[int] = field(default_factory=lambda: [128, 256])
    models: List[str]   = field(default_factory=list)   # empty → use base scenario's model
    backends: List[str] = field(default_factory=list)   # empty → use base scenario's backend

    @classmethod
    def from_dict(cls, d: dict) -> "SweepConfig":
        """Construct a SweepConfig from the 'ranges' section of a sweep YAML."""
        return cls(
            name=d.get("name", "sweep"),
            concurrency_levels=d.get("concurrency",    [1, 4, 8, 16, 32, 64]),
            prompt_tokens_values=d.get("prompt_tokens", [512, 1024, 2048]),
            output_tokens_values=d.get("output_tokens", [128, 256]),
            models=d.get("models",   []),
            backends=d.get("backends", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":               self.name,
            "concurrency":        self.concurrency_levels,
            "prompt_tokens":      self.prompt_tokens_values,
            "output_tokens":      self.output_tokens_values,
            "models":             self.models,
            "backends":           self.backends,
        }

    @property
    def total_combinations(self) -> int:
        """Total number of benchmark runs this sweep will produce."""
        return (
            max(len(self.concurrency_levels),    1) *
            max(len(self.prompt_tokens_values),  1) *
            max(len(self.output_tokens_values),  1) *
            max(len(self.models),                1) *
            max(len(self.backends),              1)
        )


# ---------------------------------------------------------------------------
# SweepResult — the collected output of a completed sweep
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """All benchmark results from a single sweep run."""
    name: str
    config: SweepConfig
    results: List[Dict[str, Any]] = field(default_factory=list)
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":           self.name,
            "config":         self.config.to_dict(),
            "total_runs":     self.total_runs,
            "completed_runs": self.completed_runs,
            "failed_runs":    self.failed_runs,
            "elapsed_s":      round(self.elapsed_s, 2),
            "results":        self.results,
        }

    def summary(self) -> str:
        """One-line human-readable summary of the sweep outcome."""
        rate = (
            f"{self.completed_runs}/{self.total_runs} runs completed"
            f" ({self.failed_runs} failed)"
            f" in {self.elapsed_s:.1f}s"
        )
        return f"[{self.name}] {rate}"


# ---------------------------------------------------------------------------
# Scenario generation — cartesian product across all parameter dimensions
# ---------------------------------------------------------------------------

def generate_sweep_scenarios(
    base: Scenario,
    config: SweepConfig,
) -> Iterator[Scenario]:
    """
    Yield one Scenario per parameter combination in the sweep grid.

    Each yielded Scenario is a deep copy of `base` with only the swept
    fields modified. The base Scenario is never mutated.

    The iteration order is:
        for model in models:
          for backend in backends:
            for concurrency in concurrency_levels:
              for prompt_tokens in prompt_tokens_values:
                for output_tokens in output_tokens_values:
                  yield scenario

    Parameters
    ----------
    base
        Template scenario. cluster and launch config are preserved in
        every generated scenario — only workload parameters change.
    config
        Defines the parameter ranges to sweep across.
    """
    # Resolve dimensions — use single-element lists when the user didn't
    # specify an override so the cartesian product still works uniformly.
    models   = config.models   or [base.workload.model]
    backends = config.backends or [base.workload.backend]

    for model, backend, concurrency, prompt_tokens, output_tokens in itertools.product(
        models,
        backends,
        config.concurrency_levels,
        config.prompt_tokens_values,
        config.output_tokens_values,
    ):
        scenario = copy.deepcopy(base)

        # Override the swept workload fields
        scenario.workload.model          = model
        scenario.workload.backend        = backend
        scenario.workload.concurrency    = concurrency
        scenario.workload.prompt_tokens  = prompt_tokens
        scenario.workload.output_tokens  = output_tokens

        # Give each scenario a descriptive name for logging and saved results
        model_short = model.split("/")[-1]  # e.g. "Llama-3.1-8B-Instruct"
        scenario.name = (
            f"{config.name}"
            f"__model-{model_short}"
            f"__backend-{backend}"
            f"__conc-{concurrency}"
            f"__pt-{prompt_tokens}"
            f"__ot-{output_tokens}"
        )

        yield scenario


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    base_scenario: Scenario,
    config: SweepConfig,
    launch_servers: bool = False,
    on_result: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> SweepResult:
    """
    Run all parameter combinations and return collected results.

    Parameters
    ----------
    base_scenario
        Template scenario — cluster and launch config are reused for every run.
    config
        Parameter ranges to sweep across.
    launch_servers
        Passed through to execute_scenario(). Set True only when using
        local/SSH/Slurm executors that need to start the server themselves.
    on_result
        Optional callback invoked after each run with (run_index, total, result).
        Useful for printing progress or updating a UI.

    Returns
    -------
    SweepResult
        Contains every individual benchmark result plus aggregate counts.
    """
    scenarios = list(generate_sweep_scenarios(base_scenario, config))
    total = len(scenarios)

    sweep = SweepResult(
        name=config.name,
        config=config,
        total_runs=total,
    )

    t_start = time.perf_counter()

    for i, scenario in enumerate(scenarios):
        try:
            result = execute_scenario(scenario, launch_servers=launch_servers)
            result["sweep_run_index"] = i
            result["sweep_total"]     = total
            sweep.results.append(result)
            sweep.completed_runs += 1
        except Exception as exc:
            # A single failed run should not abort the whole sweep.
            # Record the failure and keep going.
            sweep.results.append({
                "scenario":         scenario.to_dict(),
                "sweep_run_index":  i,
                "sweep_total":      total,
                "error":            str(exc),
                "benchmark_result": {},
            })
            sweep.failed_runs += 1

        if on_result is not None:
            on_result(i + 1, total, sweep.results[-1])

    sweep.elapsed_s = time.perf_counter() - t_start
    return sweep