from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TopologyConfig:
    """
    Describes the physical placement and interconnect topology of the cluster.

    This is metadata that travels with every result so comparisons can be
    filtered or grouped by topology. For example: "show me all results where
    nodes share the same NVLink domain" vs "results across separate racks".

    Fields
    ------
    rack_id
        Identifier for the physical rack (e.g. "rack-07"). Nodes in the
        same rack share top-of-rack switch bandwidth.
    switch_group
        Identifier for the network switch group. Nodes in the same switch
        group have lower inter-node latency than nodes crossing to a
        different switch.
    nvlink_domain
        Identifier for the NVLink / NVSwitch domain. Nodes in the same
        NVLink domain can use GPU-direct RDMA across nodes (NVLink 4.0+).
        For AMD this maps to the Infinity Fabric domain.
    nodes_per_switch
        How many nodes share a single top-of-rack switch. Affects
        available inter-node bandwidth per node under load.
    inter_node_bandwidth_gbps
        Measured or rated bandwidth between nodes in GB/s.
        E.g. 400 for 400GbE, 3200 for NVLink 4.0.
    """
    rack_id: str = ""
    switch_group: str = ""
    nvlink_domain: str = ""
    nodes_per_switch: int = 0
    inter_node_bandwidth_gbps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TopologyConfig":
        return cls(
            rack_id=d.get("rack_id", ""),
            switch_group=d.get("switch_group", ""),
            nvlink_domain=d.get("nvlink_domain", ""),
            nodes_per_switch=int(d.get("nodes_per_switch", 0)),
            inter_node_bandwidth_gbps=float(d.get("inter_node_bandwidth_gbps", 0.0)),
        )


@dataclass
class ClusterConfig:
    accelerator_vendor: str = "nvidia"
    accelerator_arch: str = "b200"
    nodes: int = 1
    accelerators_per_node: int = 8
    interconnect: str = "ethernet"
    ssh_user: str = ""
    hosts: List[str] = field(default_factory=list)
    slurm_partition: str = ""
    slurm_account: str = ""
    # Phase 5: topology metadata for rack-scale placement awareness
    topology: TopologyConfig = field(default_factory=TopologyConfig)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # asdict handles nested dataclasses automatically
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ClusterConfig":
        """
        Build a ClusterConfig from a dict, handling the nested topology
        sub-section gracefully (absent = empty TopologyConfig).
        """
        topo_dict = d.pop("topology", {}) if isinstance(d, dict) else {}
        # Remove topology before passing **d to the constructor
        clean = {k: v for k, v in d.items() if k != "topology"}
        return cls(
            **{k: v for k, v in clean.items()
               if k in ClusterConfig.__dataclass_fields__
               and k != "topology"},
            topology=TopologyConfig.from_dict(topo_dict or {}),
        )


@dataclass
class WorkloadConfig:
    name: str = "chat-assistant"
    model: str = "meta-llama/Llama-3.1-70B-Instruct"
    backend: str = "vllm"
    traffic_pattern: str = "steady"
    prompt_tokens: int = 2048
    output_tokens: int = 256
    concurrency: int = 64
    requests: int = 200
    duration_s: int = 300
    target_ttft_ms: int = 1500
    target_p95_ms: int = 5000
    endpoint: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LaunchConfig:
    executor: str = "local"
    model_cache_dir: str = ""
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    extra_args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    nodes_per_replica: int = 1
    replicas: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Scenario:
    name: str
    cluster: ClusterConfig
    workload: WorkloadConfig
    launch: LaunchConfig

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Scenario":
        cluster_dict  = payload.get("cluster", {})
        return cls(
            name=payload.get("name", "scenario"),
            cluster=ClusterConfig.from_dict(dict(cluster_dict)),
            workload=WorkloadConfig(**payload.get("workload", {})),
            launch=LaunchConfig(**payload.get("launch", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":     self.name,
            "cluster":  self.cluster.to_dict(),
            "workload": self.workload.to_dict(),
            "launch":   self.launch.to_dict(),
        }