# AcceleratorLab Console Pro

A benchmark framework for AI inference servers across heterogeneous GPU accelerator fleets.
Supports NVIDIA and AMD deployments from single desktop servers to multi-node Slurm clusters.

> **Version:** 0.4.0

---

## What It Does

AcceleratorLab runs structured load tests against OpenAI-compatible AI inference servers and
measures real performance metrics — throughput, time-to-first-token (TTFT), and p95 latency.
It supports multiple serving backends, multiple execution environments, and two traffic patterns.

**Backends:** vLLM, SGLang, TGI, OpenAI-compatible API, TensorRT-LLM (placeholder)  
**Executors:** Local (Popen), SSH (parallel fan-out), Slurm (sbatch)  
**Traffic patterns:** Steady (concurrency-bounded sliding window), Burst (all-at-once stress test)  
**Interfaces:** Native PyQt6 desktop GUI, Streamlit web console, CLI

---

## Requirements

- Python 3.11+
- See `requirements.txt` for Python dependencies

```bash
pip install -r requirements.txt
```

**For the desktop GUI only** — Linux systems may need Qt platform libraries:

```bash
sudo apt-get install libxcb-cursor0 libxcb-xinerama0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxkbcommon-x11-0 libfontconfig1 libdbus-1-3
```

---

## Interfaces

### Desktop GUI (recommended for end users)

Native PyQt6 app with model picker, accelerator target picker, live charts, and no browser required.

```bash
python -m scalelab.gui.app
```

### Web Console (Streamlit)

Browser-based operator dashboard.

```bash
streamlit run scalelab/ui/app.py
```

### CLI

Run a scenario directly from a YAML or JSON file.

```bash
# Benchmark only — target a server already running
python -m scalelab.cli.run --scenario examples/scenario_local.yaml

# Launch server + benchmark in one step
python -m scalelab.cli.run --scenario examples/scenario_slurm.yaml --launch-servers

# Save results to a specific file
python -m scalelab.cli.run --scenario examples/scenario_local.yaml --output results.json
```

---

## Scenario Files

All benchmark configuration lives in a YAML (or JSON) scenario file. Two examples are provided
in `examples/`:

| File | Description |
|---|---|
| `scenario_local.yaml` | Single node, openai-compat backend, local executor |
| `scenario_slurm.yaml` | 4-node AMD cluster, vLLM backend, Slurm executor |

**Scenario structure:**

```yaml
scenario:
  name: my-benchmark
  cluster:
    accelerator_vendor: nvidia   # nvidia | amd
    accelerator_arch: h100
    nodes: 1
    accelerators_per_node: 8
    interconnect: ethernet       # ethernet | infiniband | nvlink
    hosts: []                    # required for SSH executor
    slurm_partition: gpu         # required for Slurm executor
    slurm_account: ""
  workload:
    model: meta-llama/Llama-3.1-8B-Instruct
    backend: vllm                # vllm | sglang | tgi | openai-compat | tensorrt-llm
    traffic_pattern: steady      # steady | burst
    prompt_tokens: 1024
    output_tokens: 128
    concurrency: 16
    requests: 100
    duration_s: 120
    target_ttft_ms: 1500         # SLO threshold
    target_p95_ms: 4000          # SLO threshold
    endpoint: http://127.0.0.1:8000/v1
    api_key: EMPTY
  launch:
    executor: local              # local | ssh | slurm
    tensor_parallel: 1
    pipeline_parallel: 1
    extra_args: []
    env: {}
```

---

## Traffic Patterns

| Pattern | Behaviour |
|---|---|
| `steady` | Concurrency-bounded sliding window — new requests submit as completions land |
| `burst` | All requests fire simultaneously — maximum stress test |

Both patterns respect the `duration_s` cap.

---

## Metrics

| Metric | How Measured |
|---|---|
| **TTFT** | Wall-clock from request dispatch to first SSE content chunk (real streaming measurement) |
| **Throughput (tok/s)** | `usage.completion_tokens` from server response / total elapsed seconds |
| **p95 latency** | Ceiling-based index: `ceil(N × 0.95) - 1` |
| **Success rate** | Completed requests / total sent |
| **SLO pass/fail** | Compared against `target_ttft_ms` and `target_p95_ms` in the scenario |

---

## Building a Standalone Executable

Produces a single binary that runs without Python installed on the target machine.

**Linux:**
```bash
chmod +x build_linux.sh
./build_linux.sh
# Output: dist/AcceleratorLab
```

**Windows** (run on a Windows machine):
```bat
build_windows.bat
# Output: dist\AcceleratorLab.exe
```

Build time: 2–5 minutes. Output size: ~60–90 MB.

---

## Project Layout

```
acceleratorlab/
├── examples/                   Scenario YAML files
├── docs/                       Installation and operation guide (PDF)
├── scalelab/
│   ├── backends/               Backend adapters (vLLM, SGLang, TGI, etc.)
│   ├── cli/                    CLI entry point
│   ├── core/                   Scenario models, orchestrator, traffic engine, results
│   ├── executors/              Local, SSH, and Slurm executors
│   ├── gui/                    PyQt6 desktop GUI
│   └── ui/                     Streamlit web console
├── acceleratorlab.spec         PyInstaller build spec
├── build_linux.sh
├── build_windows.bat
└── requirements.txt
```

---

## Supported Operating Systems

| OS | CLI | Streamlit UI | Desktop GUI | Multi-node |
|---|---|---|---|---|
| Linux | ✓ | ✓ | ✓ | ✓ (recommended) |
| Windows 10/11 | ✓ | ✓ | ✓ | Requires OpenSSH |
| macOS | ✓ | ✓ | ✓ | Limited |

Linux is the recommended OS for real multi-node benchmarking.

---

## Known Limitations (v0.4.0)

- **TensorRT-LLM** launch command is a placeholder — not yet implemented
- **Hardware telemetry** (GPU utilization, power draw, memory bandwidth) is not collected
- API keys are stored in plaintext YAML — do not commit scenario files with real keys
- No retry logic in the traffic engine for transient server errors
- Cluster topology (hostnames, Slurm partitions) must be configured manually

---

## Documentation

Full installation and operation guide: [`docs/AcceleratorLab_Installation_and_Operation_Guide.pdf`](docs/AcceleratorLab_Installation_and_Operation_Guide.pdf)
