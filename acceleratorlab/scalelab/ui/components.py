from __future__ import annotations
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from scalelab.ui.state import append_run, friendly_summary, normalize_results


def render_header():
    st.title("AcceleratorLab Console Pro")
    st.caption("Multi-node AI inference benchmark — planner, launcher, and results console.")


def render_help_panel():
    st.markdown("### Help")
    with st.expander("What does multi-node mean here?"):
        st.write(
            "A single benchmark scenario launches or models serving across multiple hosts "
            "or Slurm nodes. The planner generates one per-rank command per node, including "
            "--master-addr and --master-port for distributed init."
        )
    with st.expander("How is TTFT measured?"):
        st.write(
            "Requests are sent with stream=True. Time-to-first-token is the wall-clock "
            "elapsed time from request dispatch to the first SSE chunk containing content. "
            "This is a real measurement, not a proxy."
        )
    with st.expander("What traffic patterns are available?"):
        st.write(
            "steady: requests are submitted in a concurrency-bounded sliding window, "
            "pacing naturally to the server's throughput. "
            "burst: all requests fire simultaneously for maximum stress. "
            "Both patterns respect the duration_s cap."
        )
    with st.expander("Can it run any model on any accelerator?"):
        st.write(
            "The tool is extensible. Real production support depends on the serving backend "
            "and model recipe. Use extra_args to pass vendor-specific flags (e.g. ROCm flags "
            "for AMD, --dtype for precision)."
        )
    with st.expander("Best target OS?"):
        st.write(
            "Linux is the best choice for real multi-node execution. "
            "Windows and macOS are fine for UI, planning, and result review."
        )


def render_workload_builder():
    s = st.session_state["scenario"]
    c = s["cluster"]
    w = s["workload"]
    l = s["launch"]

    st.subheader("Scenario")
    s["name"] = st.text_input("Scenario name", value=s["name"])

    a, b, c1 = st.columns(3)
    c["accelerator_vendor"] = a.selectbox(
        "Vendor", ["nvidia", "amd", "other"],
        index=["nvidia", "amd", "other"].index(c["accelerator_vendor"]),
    )
    c["accelerator_arch"] = b.text_input("Accelerator architecture", value=c["accelerator_arch"])
    interconnects = ["ethernet", "infiniband", "nvlink", "xgmi", "other"]
    c["interconnect"] = c1.selectbox(
        "Interconnect", interconnects,
        index=interconnects.index(c["interconnect"]) if c["interconnect"] in interconnects else 0,
    )

    d, e, f = st.columns(3)
    c["nodes"]                  = d.number_input("Node count",             min_value=1,   max_value=1024, value=int(c["nodes"]))
    c["accelerators_per_node"]  = e.number_input("Accelerators per node",  min_value=1,   max_value=64,   value=int(c["accelerators_per_node"]))
    l["executor"]               = f.selectbox("Executor", ["local", "ssh", "slurm"],
                                              index=["local", "ssh", "slurm"].index(l["executor"]))

    g, h = st.columns(2)
    backends = ["vllm", "sglang", "tgi", "openai-compat", "tensorrt-llm"]
    w["backend"] = g.selectbox("Serving backend", backends,
                               index=backends.index(w["backend"]))
    w["model"]   = h.text_input("Model", value=w["model"])

    i, j, k = st.columns(3)
    w["prompt_tokens"] = i.slider("Prompt tokens",  128,   32768, int(w["prompt_tokens"]), step=128)
    w["output_tokens"] = j.slider("Output tokens",   16,    4096, int(w["output_tokens"]), step=16)
    w["concurrency"]   = k.slider("Concurrency",      1,    1024, int(w["concurrency"]),   step=1)

    m, n, o = st.columns(3)
    w["requests"]       = m.number_input("Request count",     min_value=1,  max_value=100000, value=int(w["requests"]))
    w["duration_s"]     = n.number_input("Duration cap (s)",  min_value=10, max_value=86400,  value=int(w["duration_s"]))
    patterns = ["steady", "burst"]
    w["traffic_pattern"] = o.selectbox(
        "Traffic pattern", patterns,
        index=patterns.index(w["traffic_pattern"]) if w["traffic_pattern"] in patterns else 0,
    )

    p1, p2, p3 = st.columns(3)
    w["target_ttft_ms"] = p1.number_input("Target TTFT (ms)", min_value=1, max_value=60000,  value=int(w["target_ttft_ms"]))
    w["target_p95_ms"]  = p2.number_input("Target p95 (ms)",  min_value=1, max_value=120000, value=int(w["target_p95_ms"]))

    q1, q2 = st.columns(2)
    w["endpoint"] = q1.text_input("OpenAI-compatible endpoint", value=w["endpoint"])
    w["api_key"]  = q2.text_input("API key", value=w["api_key"], type="password")

    st.info(friendly_summary(s))


def render_executor_panel():
    s = st.session_state["scenario"]
    c = s["cluster"]
    l = s["launch"]

    st.subheader("Launch details")
    a, b = st.columns(2)
    l["tensor_parallel"]   = a.number_input("Tensor parallel",   min_value=1, max_value=64, value=int(l["tensor_parallel"]))
    l["pipeline_parallel"] = b.number_input("Pipeline parallel", min_value=1, max_value=64, value=int(l["pipeline_parallel"]))

    if l["executor"] == "ssh":
        c["ssh_user"]  = st.text_input("SSH user", value=c.get("ssh_user", ""))
        hosts_raw      = st.text_area("Hosts (one per line)", value="\n".join(c.get("hosts", [])))
        c["hosts"]     = [x.strip() for x in hosts_raw.splitlines() if x.strip()]

    if l["executor"] == "slurm":
        c["slurm_partition"] = st.text_input("Slurm partition", value=c.get("slurm_partition", "gpu"))
        c["slurm_account"]   = st.text_input("Slurm account",   value=c.get("slurm_account",   ""))

    extra        = st.text_area("Extra backend args (space separated)", value=" ".join(l.get("extra_args", [])))
    l["extra_args"] = [x for x in extra.split() if x]

    model_cache  = st.text_input("Model cache dir (optional)", value=l.get("model_cache_dir", ""))
    l["model_cache_dir"] = model_cache


def render_results_review():
    st.subheader("Results")
    uploaded = st.file_uploader(
        "Upload benchmark result JSON", type=["json"], accept_multiple_files=True
    )
    if uploaded:
        loaded = []
        for f in uploaded:
            payload = json.load(f)
            if isinstance(payload, list):
                loaded.extend(payload)
            else:
                loaded.append(payload)
        st.session_state["loaded_results"] = loaded
        # Append to run_history via the deduplication helper
        for item in loaded:
            append_run(item)

    # run_history is the single source of truth for the dashboard
    rows = st.session_state.get("run_history", [])
    if not rows:
        st.info("No results yet. Run a scenario or upload a result JSON.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    best = df.sort_values(["meets_slo", "tok_s"], ascending=[False, False]).iloc[0]
    st.success(
        f"Best result: {best['system']} / {best['model']} / "
        f"{best['tok_s']:.1f} tok/s / p95 {best['p95_ms']:.0f} ms"
    )


def render_dashboard():
    # Use run_history as the single source of truth — no double-counting
    rows = st.session_state.get("run_history", [])
    if not rows:
        st.info("Load or generate results to populate the dashboard.")
        return

    df = pd.DataFrame(rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best tok/s",    f"{df['tok_s'].max():.1f}")
    c2.metric("Avg tok/s",     f"{df['tok_s'].mean():.1f}")
    c3.metric("Avg p95",       f"{df['p95_ms'].mean():.0f} ms")
    c4.metric("SLO pass rate", f"{100 * df['meets_slo'].mean():.1f}%")

    # Peak throughput by system
    fig, ax = plt.subplots(figsize=(8, 4.5))
    by_system = df.groupby("system", as_index=False).agg(best_tok_s=("tok_s", "max"))
    ax.bar(by_system["system"], by_system["best_tok_s"])
    ax.set_title("Peak throughput by system")
    ax.set_ylabel("Tokens / sec")
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig)

    # p95 latency by concurrency per system
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    for system in sorted(df["system"].unique()):
        sub = df[df["system"] == system].sort_values("concurrency")
        ax2.plot(sub["concurrency"], sub["p95_ms"], marker="o", label=system)
    ax2.set_title("p95 latency by concurrency")
    ax2.set_xlabel("Concurrency")
    ax2.set_ylabel("p95 latency (ms)")
    ax2.legend()
    st.pyplot(fig2)

    # TTFT by system
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    by_ttft = df.groupby("system", as_index=False).agg(mean_ttft=("ttft_ms", "mean"))
    ax3.barh(by_ttft["system"], by_ttft["mean_ttft"])
    ax3.set_title("Mean TTFT by system")
    ax3.set_xlabel("Mean TTFT (ms)")
    st.pyplot(fig3)
