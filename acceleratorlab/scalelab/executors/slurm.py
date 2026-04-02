from __future__ import annotations
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from scalelab.executors.base import Executor


class SlurmExecutor(Executor):
    name = "slurm"

    def __init__(
        self,
        partition: str,
        account: str = "",
        nodes: int = 1,
        gpus_per_node: int = 8,
    ) -> None:
        self.partition    = partition
        self.account      = account
        self.nodes        = nodes
        self.gpus_per_node = gpus_per_node

    def _build_script(
        self,
        commands: List[List[str]],
        env: Dict[str, str] | None = None,
    ) -> str:
        # shlex.quote each value so single-quotes in env values don't break the script
        env_lines  = [f"export {k}={shlex.quote(v)}" for k, v in (env or {}).items()]
        env_block  = "\n".join(env_lines)
        body       = "\n".join(" ".join(shlex.quote(c) for c in cmd) for cmd in commands)
        account_line = f"#SBATCH --account={self.account}\n" if self.account else ""

        return (
            "#!/bin/bash\n"
            "#SBATCH --job-name=acceleratorlab\n"
            f"#SBATCH --partition={self.partition}\n"
            f"{account_line}"
            f"#SBATCH --nodes={self.nodes}\n"
            f"#SBATCH --gres=gpu:{self.gpus_per_node}\n"
            "#SBATCH --ntasks-per-node=1\n"
            "#SBATCH --output=slurm-%j.out\n\n"
            "set -euo pipefail\n"
            f"{env_block}\n"
            f"{body}\n"
        )

    def launch(
        self,
        commands: List[List[str]],
        env: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        script_text = self._build_script(commands, env)
        script_path = Path("acceleratorlab_slurm_job.sh")
        script_path.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
        )
        return {
            "executor":    self.name,
            "script_path": str(script_path),
            "returncode":  proc.returncode,
            "stdout":      proc.stdout,
            "stderr":      proc.stderr,
            "submitted":   proc.returncode == 0,
        }
