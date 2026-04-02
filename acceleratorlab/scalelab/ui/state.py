from __future__ import annotations
from copy import deepcopy

import streamlit as st
import yaml

from scalelab.core.projects import (
    list_projects as core_list_projects,
    load_project as core_load_project,
    save_project as core_save_project,
)

DEFAULT_SCENARIO = {
    "name": "my-multi-node-benchmark",
    "cluster": {
        "accelerator_vendor": "nvidia",
        "accelerator_arch": "b200",
        "nodes": 2,
        "accelerators_per_node": 8,
        "interconnect": "infiniband",
        "ssh_user": "",
        "hosts": [],
        "slurm_partition": "gpu",
        "slurm_account": "",
    },
    "workload": {
        "name": "chat-assistant",
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "backend": "vllm",
        "traffic_pattern": "steady",
        "prompt_tokens": 2048,
        "output_tokens": 256,
        "concurrency": 64,
        "requests": 64,
        "duration_s": 300,
        "target_ttft_ms": 1500,
        "target_p95_ms": 5000,
        "endpoint": "http://127.0.0.1:8000/v1",
        "api_key": "EMPTY",
    },
    "launch": {
        "executor": "slurm",
        "model_cache_dir": "",
        "tensor_parallel": 8,
        "pipeline_parallel": 1,
        "extra_args": [],
        "env": {},
        "nodes_per_replica": 1,
        "replicas": 1,
    },
}


def init_state():
    if "scenario" not in st.session_state:
        st.session_state["scenario"] = deepcopy(DEFAULT_SCENARIO)
    if "run_history" not in st.session_state:
        st.session_state["run_history"] = []
    if "loaded_results" not in st.session_state:
        st.session_state["loaded_results"] = []
    if "project_name" not in st.session_state:
        st.session_state["project_name"] = DEFAULT_SCENARIO["name"]


def scenario_yaml_text():
    return yaml.safe_dump({"scenario": st.session_state["scenario"]}, sort_keys=False)


def friendly_summary(s):
    c = s["cluster"]
    w = s["workload"]
    return (
        f"{c['nodes']} node(s) × {c['accelerators_per_node']} accelerator(s)/node · "
        f"{w['backend']} serving · {w['model']} · concurrency {w['concurrency']} · "
        f"pattern: {w['traffic_pattern']}"
    )


def normalize_results(items):
    rows = []
    for item in items:
        r = item.get("benchmark_result", item)
        rows.append({
            "system":          r.get("system",          "unknown"),
            "model":           r.get("model",           "unknown"),
            "backend":         r.get("backend",         "unknown"),
            "concurrency":     int(r.get("concurrency",    1)),
            "tok_s":           float(r.get("tok_s",        0)),
            "ttft_ms":         float(r.get("ttft_ms",      0)),
            "mean_latency_ms": float(r.get("mean_latency_ms", 0)),
            "p95_ms":          float(r.get("p95_ms",      0)),
            "success_rate":    float(r.get("success_rate", 0)),
            "requests_ok":     int(r.get("requests_ok",   r.get("raw_count", 0))),
            "duration_s":      float(r.get("duration_s",  0)),
            "traffic_pattern": r.get("traffic_pattern", "steady"),
            "meets_slo":       bool(r.get("meets_slo",   False)),
        })
    return rows


def append_run(item):
    """Append a single result item to run_history (deduplication source of truth)."""
    st.session_state["run_history"].extend(normalize_results([item]))


def save_project(name):
    payload = {
        "project_name":   name,
        "scenario":       st.session_state["scenario"],
        "run_history":    st.session_state["run_history"],
        "loaded_results": st.session_state["loaded_results"],
    }
    return str(core_save_project(name, payload))


def list_projects():
    return core_list_projects()


def load_project(filename):
    payload = core_load_project(filename)
    st.session_state["project_name"]   = payload.get("project_name",   "project")
    st.session_state["scenario"]       = payload.get("scenario",       deepcopy(DEFAULT_SCENARIO))
    st.session_state["run_history"]    = payload.get("run_history",    [])
    st.session_state["loaded_results"] = payload.get("loaded_results", [])
