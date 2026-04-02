from __future__ import annotations
import concurrent.futures
import shlex
import subprocess
from typing import Any, Dict, List

from scalelab.executors.base import Executor


class SSHExecutor(Executor):
    name = "ssh"

    def __init__(self, hosts: List[str], user: str = "") -> None:
        self.hosts = hosts
        self.user  = user

    def _launch_one(self, host: str, cmd: List[str]) -> Dict[str, Any]:
        target     = f"{self.user}@{host}" if self.user else host
        # shlex.quote each token so paths/args with spaces survive the SSH shell
        remote_cmd = " ".join(shlex.quote(c) for c in cmd)
        ssh_cmd    = ["ssh", target, remote_cmd]
        proc = subprocess.run(ssh_cmd, capture_output=True, text=True)
        return {
            "host":        host,
            "command":     cmd,
            "ssh_command": ssh_cmd,
            "returncode":  proc.returncode,
            "stdout":      proc.stdout,
            "stderr":      proc.stderr,
        }

    def launch(
        self,
        commands: List[List[str]],
        env: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        if not self.hosts:
            raise ValueError("SSH executor requires one or more hosts")

        # Fan out to all nodes in parallel rather than sequentially
        pairs = [
            (self.hosts[i % len(self.hosts)], cmd)
            for i, cmd in enumerate(commands)
        ]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs)) as pool:
            futs = {
                pool.submit(self._launch_one, host, cmd): (host, cmd)
                for host, cmd in pairs
            }
            for fut in concurrent.futures.as_completed(futs):
                results.append(fut.result())

        return {"executor": self.name, "results": results}
