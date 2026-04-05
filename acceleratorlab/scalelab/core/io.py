from __future__ import annotations
import json
from pathlib import Path
import yaml
from scalelab.core.models import Scenario


def load_scenario(path: str | Path) -> Scenario:
    """Load a Scenario from a YAML or JSON file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    elif p.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        raise ValueError(f"Unsupported scenario format: {p.suffix}")
    if "scenario" in payload:
        payload = payload["scenario"]
    return Scenario.from_dict(payload)


def load_sweep_file(path: str | Path) -> tuple:
    """
    Load a sweep YAML file and return (base_scenario, sweep_config_dict).

    Sweep files have the structure:
        sweep:
          name: my-sweep
          base_scenario:
            cluster: ...
            workload: ...
            launch: ...
          ranges:
            concurrency: [1, 4, 8, 16]
            prompt_tokens: [512, 1024]
            output_tokens: [128, 256]

    Returns
    -------
    (Scenario, dict)
        The base Scenario object and the raw ranges dict.
        Pass the ranges dict to SweepConfig.from_dict() to build
        the full SweepConfig.
    """
    from scalelab.core.sweep import SweepConfig

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    elif p.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        raise ValueError(f"Unsupported sweep file format: {p.suffix}")

    # Unwrap top-level 'sweep' key if present
    if "sweep" in payload:
        payload = payload["sweep"]

    sweep_name    = payload.get("name", "sweep")
    base_dict     = payload.get("base_scenario", payload.get("scenario", {}))
    ranges_dict   = payload.get("ranges", {})
    ranges_dict["name"] = sweep_name   # carry name through to SweepConfig

    base_scenario = Scenario.from_dict(base_dict)
    config        = SweepConfig.from_dict(ranges_dict)

    return base_scenario, config


def save_json(path: str | Path, payload) -> None:
    """Serialize payload to a JSON file."""
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")