from __future__ import annotations
import os
import subprocess
from typing import Any, Dict, List

from scalelab.executors.base import Executor


class LocalExecutor(Executor):
    name = "local"

    def launch(
        self,
        commands: List[List[str]],
        env: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        handles = []
        results = []

        for cmd in commands:
            # Use Popen (non-blocking) so long-running server daemons don't
            # hang the benchmark runner.  The caller is responsible for
            # waiting on the health-check endpoint before sending traffic.
            proc = subprocess.Popen(
                cmd,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            handles.append({"command": cmd, "pid": proc.pid, "proc": proc})

        for h in handles:
            proc = h.pop("proc")
            # Poll once — if the process already exited (e.g. bad command),
            # capture its output; otherwise report it as running.
            returncode = proc.poll()
            if returncode is not None:
                stdout, stderr = proc.communicate(timeout=5)
                results.append({
                    "command":    h["command"],
                    "pid":        h["pid"],
                    "returncode": returncode,
                    "stdout":     stdout.decode("utf-8", errors="replace"),
                    "stderr":     stderr.decode("utf-8", errors="replace"),
                    "status":     "exited",
                })
            else:
                results.append({
                    "command":    h["command"],
                    "pid":        h["pid"],
                    "returncode": None,
                    "status":     "running",
                })

        return {"executor": self.name, "results": results}
