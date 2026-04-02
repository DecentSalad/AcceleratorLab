from __future__ import annotations
import argparse, json
from scalelab.core.io import load_scenario, save_json
from scalelab.core.orchestrator import execute_scenario

def main():
    parser = argparse.ArgumentParser(description="Run AcceleratorLab scenario")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--launch-servers", action="store_true")
    parser.add_argument("--output", default="benchmark_result.json")
    args = parser.parse_args()
    scenario = load_scenario(args.scenario)
    result = execute_scenario(scenario, launch_servers=args.launch_servers)
    save_json(args.output, result)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
