from __future__ import annotations
import json

import streamlit as st

from scalelab.core.models import Scenario
from scalelab.core.orchestrator import execute_scenario
from scalelab.ui.components import (
    render_dashboard,
    render_executor_panel,
    render_header,
    render_help_panel,
    render_results_review,
    render_workload_builder,
)
from scalelab.ui.sample_data import load_demo_runs
from scalelab.ui.state import (
    append_run,
    init_state,
    list_projects,
    load_project,
    save_project,
    scenario_yaml_text,
)

st.set_page_config(
    page_title="AcceleratorLab Console Pro",
    page_icon="📊",
    layout="wide",
)
init_state()
render_header()

with st.sidebar:
    st.subheader("Projects")
    st.session_state["project_name"] = st.text_input(
        "Project name", value=st.session_state["project_name"]
    )
    a, b = st.columns(2)
    with a:
        if st.button("Save", use_container_width=True):
            path = save_project(st.session_state["project_name"])
            st.success(f"Saved: {path}")
    with b:
        projects = list_projects()
        selected = st.selectbox("Load", [""] + projects, label_visibility="collapsed")
        if selected:
            load_project(selected)
            st.success(f"Loaded {selected}")
            st.rerun()

    if st.button("Load demo results", use_container_width=True):
        demo = load_demo_runs()
        st.session_state["loaded_results"] = demo
        for item in demo:
            append_run(item)
        st.success("Demo results loaded.")

    render_help_panel()

tab1, tab2, tab3, tab4 = st.tabs(["Plan", "Launch & Benchmark", "Results", "Export"])

with tab1:
    render_workload_builder()
    render_executor_panel()

with tab2:
    st.subheader("Launch & benchmark")
    launch_servers = st.checkbox(
        "Launch serving commands via executor",
        value=False,
        help="When enabled, the executor will start the serving backend before running traffic.",
    )
    if st.button("Run scenario now", use_container_width=True):
        with st.spinner("Running benchmark…"):
            scenario = Scenario.from_dict(st.session_state["scenario"])
            result   = execute_scenario(scenario, launch_servers=launch_servers)
        st.session_state["last_run_result"] = result
        st.session_state["loaded_results"]  = [result] + st.session_state.get("loaded_results", [])
        append_run(result)
        st.success("Scenario completed.")
        st.json(result)

with tab3:
    render_results_review()
    render_dashboard()

with tab4:
    st.subheader("Export")
    yaml_text = scenario_yaml_text()
    st.code(yaml_text, language="yaml")
    st.download_button(
        "Download scenario YAML", yaml_text,
        "scenario.yaml", "text/yaml",
        use_container_width=True,
    )
    last_run = st.session_state.get("last_run_result")
    if last_run:
        st.download_button(
            "Download last run JSON",
            json.dumps(last_run, indent=2),
            "last_run.json", "application/json",
            use_container_width=True,
        )
