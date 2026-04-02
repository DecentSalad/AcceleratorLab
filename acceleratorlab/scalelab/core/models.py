from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

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
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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
        return cls(
            name=payload.get("name", "scenario"),
            cluster=ClusterConfig(**payload.get("cluster", {})),
            workload=WorkloadConfig(**payload.get("workload", {})),
            launch=LaunchConfig(**payload.get("launch", {})),
        )
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cluster": self.cluster.to_dict(),
            "workload": self.workload.to_dict(),
            "launch": self.launch.to_dict(),
        }
