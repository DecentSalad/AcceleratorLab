"""
Tests for Phase 5 — Rack-Scale Orchestration.

Run with:  pytest tests/test_rack.py -v

Covers:
  - TopologyConfig construction and serialisation
  - ClusterConfig correctly nests TopologyConfig
  - Scenario round-trip with topology fields
  - SSHExecutor: timeout, key file, quorum tracking, per-host error isolation
  - SlurmExecutor: exclusive flag, time limit, job ID parsing
  - Distributed health-check: all-ready, partial failure, quorum logic
  - Topology metadata attached to launch_result in execute_scenario()

No real cluster, SSH connection, or GPU required — all network calls
and subprocess calls are mocked.
"""
import pytest
from unittest.mock import patch, MagicMock, call
from typing import Any, Dict

from scalelab.core.models import (
    TopologyConfig, ClusterConfig, WorkloadConfig, LaunchConfig, Scenario
)
from scalelab.executors.ssh import SSHExecutor
from scalelab.executors.slurm import SlurmExecutor
from scalelab.core.orchestrator import (
    _validate_vendor, _wait_for_all_nodes, execute_scenario
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scenario(
    hosts=None,
    executor="local",
    vendor="nvidia",
    topology_kwargs=None,
) -> Scenario:
    topo = TopologyConfig(**(topology_kwargs or {}))
    return Scenario(
        name="test",
        cluster=ClusterConfig(
            accelerator_vendor=vendor,
            accelerator_arch="h100",
            nodes=len(hosts) if hosts else 1,
            hosts=hosts or [],
            topology=topo,
        ),
        workload=WorkloadConfig(
            model="test-model",
            backend="openai-compat",
            endpoint="http://127.0.0.1:8000/v1",
        ),
        launch=LaunchConfig(executor=executor),
    )


# ===========================================================================
# TopologyConfig
# ===========================================================================

class TestTopologyConfig:
    """Tests for the new TopologyConfig dataclass."""

    def test_default_values_are_empty(self):
        """
        A default TopologyConfig should have empty/zero values so it
        can be safely included in every result without adding noise
        when topology hasn't been configured.
        """
        topo = TopologyConfig()
        assert topo.rack_id                   == ""
        assert topo.switch_group              == ""
        assert topo.nvlink_domain             == ""
        assert topo.nodes_per_switch          == 0
        assert topo.inter_node_bandwidth_gbps == 0.0

    def test_explicit_values_stored(self):
        """
        Values passed to the constructor should be stored and retrievable.
        """
        topo = TopologyConfig(
            rack_id="rack-07",
            switch_group="switch-A",
            nvlink_domain="domain-0",
            nodes_per_switch=8,
            inter_node_bandwidth_gbps=400.0,
        )
        assert topo.rack_id                   == "rack-07"
        assert topo.switch_group              == "switch-A"
        assert topo.nvlink_domain             == "domain-0"
        assert topo.nodes_per_switch          == 8
        assert topo.inter_node_bandwidth_gbps == 400.0

    def test_to_dict_has_all_keys(self):
        """
        to_dict() must include every field so topology travels with results.
        """
        topo = TopologyConfig(rack_id="rack-01")
        d    = topo.to_dict()
        for key in ["rack_id", "switch_group", "nvlink_domain",
                    "nodes_per_switch", "inter_node_bandwidth_gbps"]:
            assert key in d

    def test_from_dict_roundtrip(self):
        """
        from_dict(to_dict()) should reproduce an equivalent TopologyConfig.
        """
        original = TopologyConfig(
            rack_id="rack-07",
            switch_group="switch-A",
            nvlink_domain="domain-0",
            nodes_per_switch=8,
            inter_node_bandwidth_gbps=400.0,
        )
        restored = TopologyConfig.from_dict(original.to_dict())
        assert restored.rack_id                   == original.rack_id
        assert restored.switch_group              == original.switch_group
        assert restored.nvlink_domain             == original.nvlink_domain
        assert restored.nodes_per_switch          == original.nodes_per_switch
        assert restored.inter_node_bandwidth_gbps == original.inter_node_bandwidth_gbps

    def test_from_dict_empty_uses_defaults(self):
        """
        from_dict({}) should produce the same defaults as the no-arg constructor.
        No KeyError should be raised for missing keys.
        """
        topo = TopologyConfig.from_dict({})
        assert topo.rack_id == ""
        assert topo.nodes_per_switch == 0


# ===========================================================================
# ClusterConfig with topology
# ===========================================================================

class TestClusterConfigTopology:
    """Tests for ClusterConfig's nested TopologyConfig."""

    def test_default_cluster_has_empty_topology(self):
        """
        A ClusterConfig created with no topology argument should have a
        default (empty) TopologyConfig, not None.
        """
        cluster = ClusterConfig()
        assert isinstance(cluster.topology, TopologyConfig)
        assert cluster.topology.rack_id == ""

    def test_topology_stored_in_cluster(self):
        """
        A TopologyConfig passed to ClusterConfig should be accessible
        via cluster.topology.
        """
        topo    = TopologyConfig(rack_id="rack-99")
        cluster = ClusterConfig(topology=topo)
        assert cluster.topology.rack_id == "rack-99"

    def test_to_dict_includes_topology(self):
        """
        ClusterConfig.to_dict() must include the topology sub-dict.
        This ensures topology travels with every saved result.
        """
        cluster = ClusterConfig(topology=TopologyConfig(rack_id="rack-01"))
        d       = cluster.to_dict()
        assert "topology" in d
        assert d["topology"]["rack_id"] == "rack-01"

    def test_from_dict_restores_topology(self):
        """
        ClusterConfig built from a dict that includes a topology section
        should have the correct nested TopologyConfig.
        """
        d = {
            "accelerator_vendor": "nvidia",
            "accelerator_arch":   "h100",
            "topology": {
                "rack_id":       "rack-05",
                "switch_group":  "sw-B",
                "nvlink_domain": "dom-1",
            }
        }
        cluster = ClusterConfig.from_dict(d)
        assert cluster.topology.rack_id      == "rack-05"
        assert cluster.topology.switch_group == "sw-B"

    def test_from_dict_without_topology_uses_defaults(self):
        """
        A ClusterConfig dict with no topology key should produce a
        ClusterConfig with an empty default topology — not raise KeyError.
        """
        cluster = ClusterConfig.from_dict({"accelerator_vendor": "amd"})
        assert isinstance(cluster.topology, TopologyConfig)
        assert cluster.topology.rack_id == ""

    def test_scenario_roundtrip_with_topology(self):
        """
        A Scenario with topology metadata should survive to_dict() /
        from_dict() with the topology fields intact.
        """
        topo = TopologyConfig(
            rack_id="rack-07",
            nvlink_domain="domain-0",
            inter_node_bandwidth_gbps=400.0,
        )
        s1 = make_scenario(topology_kwargs={
            "rack_id": "rack-07",
            "nvlink_domain": "domain-0",
            "inter_node_bandwidth_gbps": 400.0,
        })
        s2 = Scenario.from_dict(s1.to_dict())
        assert s2.cluster.topology.rack_id                   == "rack-07"
        assert s2.cluster.topology.nvlink_domain             == "domain-0"
        assert s2.cluster.topology.inter_node_bandwidth_gbps == 400.0


# ===========================================================================
# SSHExecutor — Phase 5 improvements
# ===========================================================================

class TestSSHExecutorPhase5:
    """Tests for Phase 5 SSH executor improvements."""

    def test_connect_timeout_in_command(self):
        """
        The SSH command must include -o ConnectTimeout=N so a single
        unreachable host cannot block the whole parallel fan-out indefinitely.
        """
        ex  = SSHExecutor(hosts=["node-01"], connect_timeout=15)
        prefix = ex._build_ssh_prefix()
        assert "ConnectTimeout=15" in " ".join(prefix)

    def test_key_file_in_command(self):
        """
        When key_file is provided, -i path must appear in the SSH prefix.
        This allows passwordless SSH into HPC clusters from the head node.
        """
        ex     = SSHExecutor(hosts=["node-01"], key_file="/home/user/.ssh/id_rsa")
        prefix = ex._build_ssh_prefix()
        assert "-i" in prefix
        assert "/home/user/.ssh/id_rsa" in prefix

    def test_no_key_file_no_i_flag(self):
        """
        When key_file is None, -i must not appear in the SSH prefix.
        """
        ex     = SSHExecutor(hosts=["node-01"])
        prefix = ex._build_ssh_prefix()
        assert "-i" not in prefix

    def test_ssh_options_included(self):
        """
        Extra SSH -o options should appear in the SSH prefix.
        StrictHostKeyChecking=no is essential on clusters where host keys
        change with each job allocation.
        """
        ex     = SSHExecutor(hosts=["node-01"],
                             ssh_options={"StrictHostKeyChecking": "no"})
        prefix = ex._build_ssh_prefix()
        assert "StrictHostKeyChecking=no" in " ".join(prefix)

    def test_quorum_tracking_all_success(self):
        """
        When all hosts respond successfully, quorum_reached should be True
        and nodes_launched should equal the number of hosts.
        """
        ex = SSHExecutor(hosts=["node-01", "node-02"])

        def fake_launch_one(host, cmd):
            return {"host": host, "returncode": 0, "stdout": "", "stderr": "",
                    "status": "ok", "command": cmd, "ssh_command": []}

        with patch.object(ex, "_launch_one", side_effect=fake_launch_one):
            result = ex.launch([["echo", "hi"], ["echo", "hi"]])

        assert result["quorum_reached"]   is True
        assert result["nodes_launched"]   == 2
        assert result["nodes_failed"]     == 0

    def test_quorum_tracking_partial_failure(self):
        """
        When one host fails, quorum_reached should be False,
        nodes_failed should be 1, and the result should still contain
        all per-host details (not just the failed one).
        """
        ex = SSHExecutor(hosts=["node-01", "node-02"])

        def fake_launch_one(host, cmd):
            status = "ok" if host == "node-01" else "timeout"
            rc     = 0   if host == "node-01" else -1
            return {"host": host, "returncode": rc, "stdout": "", "stderr": "",
                    "status": status, "command": cmd, "ssh_command": []}

        with patch.object(ex, "_launch_one", side_effect=fake_launch_one):
            result = ex.launch([["echo", "hi"], ["echo", "hi"]])

        assert result["quorum_reached"] is False
        assert result["nodes_launched"] == 1
        assert result["nodes_failed"]   == 1
        assert len(result["results"])   == 2

    def test_timeout_returns_structured_result(self):
        """
        A subprocess.TimeoutExpired during SSH should return a structured
        result dict with status="timeout" rather than raising an exception.
        One failed host must not abort the other parallel SSH connections.
        """
        ex = SSHExecutor(hosts=["node-01"], connect_timeout=1)

        with patch("subprocess.run", side_effect=__import__("subprocess").TimeoutExpired(
            cmd=["ssh"], timeout=1
        )):
            result = ex._launch_one("node-01", ["echo", "hi"])

        assert result["status"]     == "timeout"
        assert result["returncode"] == -1
        assert "timed out" in result["stderr"].lower()

    def test_empty_hosts_raises(self):
        """
        Calling launch() with no hosts must raise ValueError immediately
        rather than silently doing nothing.
        """
        ex = SSHExecutor(hosts=[])
        with pytest.raises(ValueError, match="requires one or more hosts"):
            ex.launch([["echo", "hi"]])


# ===========================================================================
# SlurmExecutor — Phase 5 improvements
# ===========================================================================

class TestSlurmExecutorPhase5:
    """Tests for Phase 5 Slurm executor improvements."""

    def test_exclusive_flag_in_script(self):
        """
        --exclusive must appear in the sbatch script when exclusive=True.
        Exclusive allocation prevents other jobs from sharing nodes during
        benchmarking, which would introduce measurement noise.
        """
        ex     = SlurmExecutor(partition="gpu", exclusive=True)
        script = ex._build_script([["echo", "hi"]])
        assert "--exclusive" in script

    def test_no_exclusive_when_disabled(self):
        """
        --exclusive must not appear when exclusive=False.
        Some clusters don't allow exclusive allocation.
        """
        ex     = SlurmExecutor(partition="gpu", exclusive=False)
        script = ex._build_script([["echo", "hi"]])
        assert "--exclusive" not in script

    def test_time_limit_in_script(self):
        """
        --time must appear in the sbatch script with the configured value.
        Without a time limit, runaway jobs accumulate on the scheduler.
        """
        ex     = SlurmExecutor(partition="gpu", time_limit="04:00:00")
        script = ex._build_script([["echo", "hi"]])
        assert "--time=04:00:00" in script

    def test_error_file_in_script(self):
        """
        The script should include --error to capture stderr from each node
        into a separate file for post-run diagnosis.
        """
        ex     = SlurmExecutor(partition="gpu")
        script = ex._build_script([["echo", "hi"]])
        assert "--error=" in script

    def test_extra_sbatch_args_included(self):
        """
        Extra SBATCH directives passed via extra_sbatch_args should appear
        verbatim in the generated script.
        """
        ex = SlurmExecutor(
            partition="gpu",
            extra_sbatch_args=["--constraint=h100", "--mail-type=FAIL"],
        )
        script = ex._build_script([["echo", "hi"]])
        assert "--constraint=h100" in script
        assert "--mail-type=FAIL"   in script

    def test_parse_job_id_success(self):
        """
        _parse_job_id() should extract the numeric job ID from sbatch stdout.
        The standard sbatch output is "Submitted batch job 12345".
        """
        job_id = SlurmExecutor._parse_job_id("Submitted batch job 99999\n")
        assert job_id == "99999"

    def test_parse_job_id_failure_returns_none(self):
        """
        _parse_job_id() should return None when the output doesn't match
        the expected pattern, rather than raising an exception.
        """
        assert SlurmExecutor._parse_job_id("ERROR: something went wrong") is None
        assert SlurmExecutor._parse_job_id("") is None

    def test_job_id_in_launch_result(self):
        """
        A successful sbatch submission should include the parsed job_id
        in the launch result dict so the user can track the job.
        """
        ex = SlurmExecutor(partition="gpu")

        mock_proc       = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout     = "Submitted batch job 42\n"
        mock_proc.stderr     = ""

        with patch("subprocess.run", return_value=mock_proc):
            result = ex.launch([["echo", "hi"]])

        assert result["submitted"] is True
        assert result["job_id"]    == "42"

    def test_failed_submission_job_id_is_none(self):
        """
        When sbatch returns non-zero, job_id should be None — the job
        was not submitted so there is no ID to return.
        """
        ex = SlurmExecutor(partition="gpu")

        mock_proc            = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout     = "error: Invalid partition\n"
        mock_proc.stderr     = "error: Invalid partition\n"

        with patch("subprocess.run", return_value=mock_proc):
            result = ex.launch([["echo", "hi"]])

        assert result["submitted"] is False
        assert result["job_id"]    is None


# ===========================================================================
# Distributed health-check
# ===========================================================================

class TestDistributedHealthCheck:
    """Tests for _wait_for_all_nodes() — Phase 5's distributed health gate."""

    def _mock_requests(self, ready_hosts):
        """
        Return a mock for requests.get that returns HTTP 200 for hosts in
        ready_hosts and raises ConnectionError for all others.
        """
        import requests as req

        def fake_get(url, timeout=5):
            host = url.split("//")[1].split(":")[0]
            if host in ready_hosts:
                r       = MagicMock()
                r.ok    = True
                r.status_code = 200
                return r
            raise req.exceptions.ConnectionError(f"Cannot connect to {host}")

        return fake_get

    def test_all_nodes_ready(self):
        """
        When every host responds with HTTP 200, all_ready and
        quorum_reached should both be True.
        """
        hosts = ["node-01", "node-02", "node-03"]
        with patch("scalelab.core.orchestrator.requests.get",
                   side_effect=self._mock_requests(set(hosts))):
            result = _wait_for_all_nodes(hosts, timeout_s=1)

        assert result["all_ready"]      is True
        assert result["quorum_reached"] is True
        assert result["nodes_ready"]    == 3
        assert result["nodes_failed"]   == 0

    def test_one_node_fails(self):
        """
        When one node times out, all_ready should be False.
        quorum_reached should also be False when default (all) quorum is used.
        """
        hosts = ["node-01", "node-02", "node-03"]
        with patch("scalelab.core.orchestrator.requests.get",
                   side_effect=self._mock_requests({"node-01", "node-02"})):
            result = _wait_for_all_nodes(hosts, timeout_s=1)

        assert result["all_ready"]      is False
        assert result["quorum_reached"] is False
        assert result["nodes_ready"]    == 2
        assert result["nodes_failed"]   == 1

    def test_partial_quorum_satisfied(self):
        """
        When quorum=2 and 2 of 3 nodes are ready, quorum_reached should
        be True even though all_ready is False.
        This allows benchmarking to proceed when one node is slow to start.
        """
        hosts = ["node-01", "node-02", "node-03"]
        with patch("scalelab.core.orchestrator.requests.get",
                   side_effect=self._mock_requests({"node-01", "node-02"})):
            result = _wait_for_all_nodes(hosts, timeout_s=1, quorum=2)

        assert result["quorum_reached"] is True
        assert result["all_ready"]      is False

    def test_details_contain_all_hosts(self):
        """
        The details list should contain one entry per host so the caller
        can identify exactly which nodes failed.
        """
        hosts = ["node-01", "node-02"]
        with patch("scalelab.core.orchestrator.requests.get",
                   side_effect=self._mock_requests({"node-01"})):
            result = _wait_for_all_nodes(hosts, timeout_s=1)

        assert len(result["details"]) == 2
        detail_hosts = {d["host"] for d in result["details"]}
        assert detail_hosts == {"node-01", "node-02"}

    def test_empty_hosts_returns_ready(self):
        """
        With no hosts declared (single-node local mode), the distributed
        health-check should return all_ready=True without making any
        HTTP requests.
        """
        result = _wait_for_all_nodes(hosts=[])
        assert result["all_ready"]      is True
        assert result["quorum_reached"] is True
        assert result["nodes_ready"]    == 0


# ===========================================================================
# Topology metadata in execute_scenario
# ===========================================================================

class TestTopologyInResults:
    """
    Tests that topology metadata is correctly attached to launch_result
    in execute_scenario(), ensuring it travels with every saved result.
    """

    def _run_scenario(self, topology_kwargs=None):
        """Run execute_scenario() with a mocked benchmark and return launch_result."""
        s = make_scenario(topology_kwargs=topology_kwargs or {})

        fake_benchmark = {
            "system": "nvidia-h100", "tok_s": 100.0, "ttft_ms": 200.0,
            "p95_ms": 400.0, "success_rate": 1.0, "meets_slo": True,
            "telemetry_available": False, "traffic_pattern": "steady",
        }
        with patch("scalelab.core.orchestrator.run_openai_compatible_benchmark",
                   return_value=fake_benchmark):
            result = execute_scenario(s, launch_servers=False)

        return result["launch_result"]

    def test_topology_key_present(self):
        """
        launch_result must always contain a 'topology' key, even when
        topology fields are all empty.
        """
        lr = self._run_scenario()
        assert "topology" in lr

    def test_topology_rack_id_propagated(self):
        """
        The rack_id from the scenario's topology config must appear in
        the launch_result topology dict.
        """
        lr = self._run_scenario({"rack_id": "rack-42"})
        assert lr["topology"]["rack_id"] == "rack-42"

    def test_topology_bandwidth_propagated(self):
        """
        inter_node_bandwidth_gbps must be carried into the result so
        comparison reports can filter or group by interconnect speed.
        """
        lr = self._run_scenario({"inter_node_bandwidth_gbps": 400.0})
        assert lr["topology"]["inter_node_bandwidth_gbps"] == 400.0

    def test_topology_all_fields_present(self):
        """
        The topology dict in launch_result must include all five fields,
        even when they're empty, so downstream consumers don't need to
        check for missing keys.
        """
        lr = self._run_scenario()
        for key in ["rack_id", "switch_group", "nvlink_domain",
                    "nodes_per_switch", "inter_node_bandwidth_gbps"]:
            assert key in lr["topology"], f"Missing topology field: {key}"