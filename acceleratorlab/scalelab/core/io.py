from __future__ import annotations
import json
from pathlib import Path
import yaml
from scalelab.core.models import Scenario

def load_scenario(path: str | Path) -> Scenario:
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

def save_json(path: str | Path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
