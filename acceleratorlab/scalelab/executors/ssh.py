"""
SSH executor — launch server commands on remote hosts in parallel.

Phase 5 improvements over the original:
  - Per-connection timeout so a hung SSH never blocks the entire fan-out
  - SSH key file support (avoids password prompts on HPC clusters)
  - Per-host error isolation with structured failure reporting
  - SSH options passthrough for StrictHostKeyChecking, compression, etc.
  - Quorum tracking: reports how many nodes launched successfully
"""
from __future__ import annotations
import concurrent.futures
import shlex
import subprocess
from typing import Any, Dict, List, Optional

from scalelab.executors.base import Executor


# Default SSH connection timeout in seconds.
# Long enough for slow DNS resolution on HPC clusters, short enough to
# fail fast if a host is unreachable rather than blocking for minutes.
_DEFAULT_CONNECT_TIMEOUT = 30


class SSHExecutor(Executor):
    name = "ssh"

    def __init__(
        self,
        hosts: List[str],
        user: str = "",
        key_file: Optional[str] = None,
        connect_timeout: int = _DEFAULT_CONNECT_TIMEOUT,
        ssh_options: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        hosts
            List of hostnames or IP addresses to fan out to.
        user
            SSH username. If empty, SSH uses the current user.
        key_file
            Path to the private key file (e.g. ~/.ssh/id_rsa).
            If None, SSH uses its default key discovery.
        connect_timeout
            Seconds before an individual SSH connection is considered failed.
        ssh_options
            Additional SSH -o options as a dict, e.g.
            {"StrictHostKeyChecking": "no", "Compression": "yes"}.
            StrictHostKeyChecking=no is the most common need on HPC clusters
            where host keys change frequently.
        """
        self.hosts           = hosts
        self.user            = user
        self.key_file        = key_file
        self.connect_timeout = connect_timeout
        self.ssh_options     = ssh_options or {}

    def _build_ssh_prefix(self) -> List[str]:
        """
        Build the base ssh command with all options applied.
        Returns a list of tokens that precede the target and remote command.
        """
        cmd = ["ssh"]

        # Connection timeout — prevents a single unreachable host from
        # blocking the entire parallel fan-out
        cmd += ["-o", f"ConnectTimeout={self.connect_timeout}"]

        # Private key file
        if self.key_file:
            cmd += ["-i", self.key_file]

        # Extra SSH -o options (e.g. StrictHostKeyChecking=no)
        for key, val in self.ssh_options.items():
            cmd += ["-o", f"{key}={val}"]

        return cmd

    def _launch_one(self, host: str, cmd: List[str]) -> Dict[str, Any]:
        """
        Launch a single command on a single remote host.
        Returns a structured result dict regardless of success or failure.
        """
        target     = f"{self.user}@{host}" if self.user else host
        remote_cmd = " ".join(shlex.quote(c) for c in cmd)
        ssh_cmd    = self._build_ssh_prefix() + [target, remote_cmd]

        try:
            proc = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=self.connect_timeout + 10,  # slightly wider than SSH's own timeout
            )
            return {
                "host":        host,
                "command":     cmd,
                "ssh_command": ssh_cmd,
                "returncode":  proc.returncode,
                "stdout":      proc.stdout,
                "stderr":      proc.stderr,
                "status":      "ok" if proc.returncode == 0 else "failed",
            }
        except subprocess.TimeoutExpired:
            return {
                "host":        host,
                "command":     cmd,
                "ssh_command": ssh_cmd,
                "returncode":  -1,
                "stdout":      "",
                "stderr":      f"SSH connection to {host} timed out after {self.connect_timeout}s",
                "status":      "timeout",
            }
        except Exception as exc:
            return {
                "host":        host,
                "command":     cmd,
                "ssh_command": ssh_cmd,
                "returncode":  -1,
                "stdout":      "",
                "stderr":      str(exc),
                "status":      "error",
            }

    def launch(
        self,
        commands: List[List[str]],
        env: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        """
        Fan out all commands to their target hosts in parallel.

        Commands are paired with hosts by index (command 0 → host 0,
        command 1 → host 1, wrapping if there are more commands than hosts).
        All SSH connections are initiated simultaneously — this is a
        correctness requirement for distributed init, not just a performance
        optimization (see Phase 2 docs for why).

        Returns a dict containing per-host results plus quorum counts.
        """
        if not self.hosts:
            raise ValueError("SSH executor requires one or more hosts")

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

        # Quorum summary — how many nodes launched successfully
        ok_count      = sum(1 for r in results if r.get("status") == "ok")
        failed_count  = len(results) - ok_count
        all_nodes_ok  = ok_count == len(results)

        return {
            "executor":       self.name,
            "results":        results,
            "nodes_launched": ok_count,
            "nodes_failed":   failed_count,
            "quorum_reached": all_nodes_ok,
        }