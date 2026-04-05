from __future__ import annotations
import argparse
import json
from scalelab.core.io import load_scenario, load_sweep_file, save_json
from scalelab.core.orchestrator import execute_scenario
from scalelab.core.sweep import run_sweep


def _progress(run_index: int, total: int, result: dict) -> None:
    """Print a one-line progress update after each sweep run."""
    name = result.get("scenario", {}).get("name", f"run-{run_index}")
    ok   = "ok" if result.get("benchmark_result") else "failed"
    print(f"  [{run_index}/{total}] {name} — {ok}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="AcceleratorLab benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Single run:  python -m scalelab.cli.run --scenario examples/scenario_local.yaml\n"
            "Sweep run:   python -m scalelab.cli.run --sweep   examples/sweep_local.yaml"
        ),
    )

    # Mutually exclusive: either run a single scenario or a sweep
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scenario", metavar="PATH",
                      help="Path to a scenario YAML/JSON file for a single benchmark run")
    mode.add_argument("--sweep",    metavar="PATH",
                      help="Path to a sweep YAML/JSON file to run across a parameter grid")

    parser.add_argument("--launch-servers", action="store_true",
                        help="Launch the serving backend before benchmarking")
    parser.add_argument("--output", default=None, metavar="PATH",
                        help="Write result JSON to this path "
                             "(default: benchmark_result.json or sweep_result.json)")

    args = parser.parse_args()

    # ── Single scenario run ───────────────────────────────────────────────────
    if args.scenario:
        output = args.output or "benchmark_result.json"
        scenario = load_scenario(args.scenario)
        result   = execute_scenario(scenario, launch_servers=args.launch_servers)
        save_json(output, result)
        print(json.dumps(result, indent=2))

    # ── Sweep run ─────────────────────────────────────────────────────────────
    else:
        output = args.output or "sweep_result.json"
        base_scenario, config = load_sweep_file(args.sweep)

        print(f"Starting sweep '{config.name}' — {config.total_combinations} runs")
        print(f"  concurrency:   {config.concurrency_levels}")
        print(f"  prompt_tokens: {config.prompt_tokens_values}")
        print(f"  output_tokens: {config.output_tokens_values}")
        if config.models:
            print(f"  models:        {config.models}")
        if config.backends:
            print(f"  backends:      {config.backends}")
        print()

        result = run_sweep(
            base_scenario,
            config,
            launch_servers=args.launch_servers,
            on_result=_progress,
        )

        save_json(output, result.to_dict())
        print()
        print(result.summary())
        print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()