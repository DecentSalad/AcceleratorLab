from __future__ import annotations
import argparse
import json
from scalelab.core.io import load_scenario, load_sweep_file, save_json
from scalelab.core.orchestrator import execute_scenario
from scalelab.core.sweep import run_sweep
from scalelab.core.compare import load_results, compare_results, detect_regressions
from scalelab.core.report import generate_markdown_report


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
            "Single run:   python -m scalelab.cli.run --scenario examples/scenario_local.yaml\n"
            "Sweep run:    python -m scalelab.cli.run --sweep   examples/sweep_local.yaml\n"
            "Compare runs: python -m scalelab.cli.run --compare h100.json mi300x.json\n"
            "With regression check:\n"
            "              python -m scalelab.cli.run --compare new.json --baseline old.json"
        ),
    )

    # Mutually exclusive run modes
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scenario", metavar="PATH",
                      help="Path to a scenario YAML/JSON for a single benchmark run")
    mode.add_argument("--sweep",    metavar="PATH",
                      help="Path to a sweep YAML/JSON to run across a parameter grid")
    mode.add_argument("--compare",  metavar="PATH", nargs="+",
                      help="One or more result JSON files to compare")

    parser.add_argument("--launch-servers", action="store_true",
                        help="Launch the serving backend before benchmarking")
    parser.add_argument("--output", default=None, metavar="PATH",
                        help="Write result to this path "
                             "(default: benchmark_result.json, sweep_result.json, or comparison.md)")
    parser.add_argument("--baseline", default=None, metavar="PATH",
                        help="Baseline result JSON for regression detection "
                             "(only used with --compare)")
    parser.add_argument("--price-per-hour", type=float, default=None, metavar="USD",
                        help="Instance cost in USD/hr for cost-efficiency calculation "
                             "(only used with --compare)")

    args = parser.parse_args()

    # ── Single scenario run ───────────────────────────────────────────────────
    if args.scenario:
        output   = args.output or "benchmark_result.json"
        scenario = load_scenario(args.scenario)
        result   = execute_scenario(scenario, launch_servers=args.launch_servers)
        save_json(output, result)
        print(json.dumps(result, indent=2))

    # ── Sweep run ─────────────────────────────────────────────────────────────
    elif args.sweep:
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

    # ── Compare run ───────────────────────────────────────────────────────────
    else:
        output = args.output or "comparison.md"
        rows   = load_results(args.compare)

        if not rows:
            print("No benchmark results found in the provided files.")
            return

        # Optional cost-efficiency enrichment
        if args.price_per_hour:
            from scalelab.core.compare import add_cost_efficiency
            rows = add_cost_efficiency(rows, args.price_per_hour)

        report = compare_results(rows)

        # Optional regression detection against a baseline file
        if args.baseline:
            baseline_rows = load_results([args.baseline])
            regressions   = detect_regressions(baseline_rows, rows)
            report.regressions = regressions
            if regressions:
                print(f"⚠️  {len(regressions)} regression(s) detected vs baseline.")
            else:
                print("✓  No regressions detected vs baseline.")

        # Print summary table to terminal
        print()
        print(report.summary_table())
        print()

        # Write markdown report
        md = generate_markdown_report(report)
        with open(output, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Report saved to: {output}")
        print(f"Systems: {', '.join(report.systems)}")
        print(f"Total runs compared: {len(report.rows)}")


if __name__ == "__main__":
    main()