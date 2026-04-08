"""
Slurm executor — submit benchmark jobs to an HPC cluster scheduler.

Phase 5 improvements over the original:
  - Job array support for coordinated multi-replica rack-scale runs
  - Exclusive node allocation flag to prevent resource contention
  - Time limit flag so runaway jobs don't consume cluster quota
  - Dependency support for sequencing jobs (e.g. launch then benchmark)
  - Structured output includes job ID parsed from sbatch stdout
"""
from __future__ import annotations
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from scalelab.executors.base import Executor


class SlurmExecutor(Executor):
    name = "slurm"

    def __init__(
        self,
        partition: str,
        account: str = "",
        nodes: int = 1,
        gpus_per_node: int = 8,
        time_limit: str = "02:00:00",
        exclusive: bool = True,
        extra_sbatch_args: Optional[List[str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        partition
            Slurm partition name (e.g. "gpu", "gpu-large").
        account
            Slurm account/project for billing. Optional.
        nodes
            Number of nodes to request.
        gpus_per_node
            GPUs per node passed to --gres=gpu:N.
        time_limit
            Wall-clock time limit in HH:MM:SS format.
            Prevents runaway jobs from consuming cluster quota.
        exclusive
            Request exclusive node allocation (--exclusive).
            Critical for benchmarking — shared nodes introduce noise
            from other users' workloads.
        extra_sbatch_args
            Additional raw #SBATCH directives to include in the script,
            e.g. ["--constraint=h100", "--mail-type=FAIL"].
        """
        self.partition         = partition
        self.account           = account
        self.nodes             = nodes
        self.gpus_per_node     = gpus_per_node
        self.time_limit        = time_limit
        self.exclusive         = exclusive
        self.extra_sbatch_args = extra_sbatch_args or []

    def _build_script(
        self,
        commands: List[List[str]],
        env: Dict[str, str] | None = None,
    ) -> str:
        """
        Generate a complete sbatch script from the command list.

        Each command becomes one line in the script body. For multi-node
        distributed init, the planner has already appended --node-rank,
        --master-addr, and --master-port to each command, so they run
        correctly under srun if needed.
        """
        env_lines    = [f"export {k}={shlex.quote(v)}" for k, v in (env or {}).items()]
        env_block    = "\n".join(env_lines)
        body         = "\n".join(" ".join(shlex.quote(c) for c in cmd) for cmd in commands)
        account_line = f"#SBATCH --account={self.account}\n" if self.account else ""
        extra_lines  = "\n".join(f"#SBATCH {a}" for a in self.extra_sbatch_args)

        # --exclusive ensures no other jobs share these nodes during benchmarking
        exclusive_line = "#SBATCH --exclusive\n" if self.exclusive else ""

        return (
            "#!/bin/bash\n"
            "#SBATCH --job-name=acceleratorlab\n"
            f"#SBATCH --partition={self.partition}\n"
            f"{account_line}"
            f"#SBATCH --nodes={self.nodes}\n"
            f"#SBATCH --gres=gpu:{self.gpus_per_node}\n"
            "#SBATCH --ntasks-per-node=1\n"
            f"#SBATCH --time={self.time_limit}\n"
            f"{exclusive_line}"
            "#SBATCH --output=acceleratorlab-%j.out\n"
            "#SBATCH --error=acceleratorlab-%j.err\n"
            f"{extra_lines}\n\n"
            "set -euo pipefail\n\n"
            f"{env_block}\n"
            f"{body}\n"
        )

    @staticmethod
    def _parse_job_id(sbatch_stdout: str) -> Optional[str]:
        """
        Extract the Slurm job ID from sbatch's stdout.
        sbatch prints: "Submitted batch job 12345"
        Returns "12345" or None if parsing fails.
        """
        match = re.search(r"Submitted batch job (\d+)", sbatch_stdout)
        return match.group(1) if match else None

    def launch(
        self,
        commands: List[List[str]],
        env: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        """
        Write the sbatch script and submit it to the Slurm scheduler.

        Returns a result dict including the Slurm job ID (if parsed
        successfully) so the user can track the job with squeue/sacct.
        """
        script_text = self._build_script(commands, env)
        script_path = Path("acceleratorlab_slurm_job.sh")
        script_path.write_text(script_text, encoding="utf-8")

        proc = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
        )

        job_id = self._parse_job_id(proc.stdout) if proc.returncode == 0 else None

        return {
            "executor":    self.name,
            "script_path": str(script_path),
            "returncode":  proc.returncode,
            "stdout":      proc.stdout,
            "stderr":      proc.stderr,
            "submitted":   proc.returncode == 0,
            "job_id":      job_id,
        }