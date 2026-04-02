from __future__ import annotations
import json
from pathlib import Path


def _project_dir() -> Path:
    """Return (and lazily create) the project storage directory.

    Uses ~/.scalelab_projects/ so it lands in the user's home directory
    regardless of the current working directory, and doesn't litter the
    filesystem on Windows or Linux.
    """
    d = Path.home() / ".scalelab_projects"
    d.mkdir(exist_ok=True)
    return d


def list_projects():
    return sorted(p.name for p in _project_dir().glob("*.json"))


def save_project(name: str, payload) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name).strip("._")
    if not safe:
        safe = "project"
    path = _project_dir() / f"{safe}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_project(filename: str):
    return json.loads((_project_dir() / filename).read_text(encoding="utf-8"))
