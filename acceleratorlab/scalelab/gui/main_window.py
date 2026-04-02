from __future__ import annotations
import json
from copy import deepcopy
from PyQt6.QtCore    import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QStatusBar, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)
import yaml

from scalelab.core.projects      import list_projects, load_project, save_project
from scalelab.core.results       import normalize_results
from scalelab.ui.sample_data     import load_demo_runs
from scalelab.gui.charts         import ChartCanvas
from scalelab.gui.worker         import BenchmarkWorker
from scalelab.gui.health_worker  import HealthCheckWorker
from scalelab.gui.model_picker   import ModelPickerDialog
from scalelab.gui.target_picker  import TargetPickerDialog
from scalelab.gui.server_setup   import ServerSetupDialog

DEFAULT = {
    "name": "my-benchmark",
    "cluster": {
        "accelerator_vendor": "nvidia", "accelerator_arch": "h100",
        "nodes": 1, "accelerators_per_node": 8,
        "interconnect": "ethernet",
        "ssh_user": "", "hosts": [],
        "slurm_partition": "gpu", "slurm_account": "",
    },
    "workload": {
        "name": "chat-assistant",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "backend": "vllm", "traffic_pattern": "steady",
        "prompt_tokens": 1024, "output_tokens": 128,
        "concurrency": 16, "requests": 100, "duration_s": 120,
        "target_ttft_ms": 1500, "target_p95_ms": 4000,
        "endpoint": "http://127.0.0.1:8000/v1", "api_key": "EMPTY",
    },
    "launch": {
        "executor": "local", "model_cache_dir": "",
        "tensor_parallel": 1, "pipeline_parallel": 1,
        "extra_args": [], "env": {}, "nodes_per_replica": 1, "replicas": 1,
    },
}


def _lbl(text, style="color:#8892c8;font-size:12px;"):
    l = QLabel(text); l.setStyleSheet(style); return l

def _spin(lo, hi, val):
    s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setMinimumWidth(110); return s

def _combo(items, cur=""):
    c = QComboBox(); c.addItems(items)
    if cur in items: c.setCurrentText(cur)
    return c

def _group(title):
    g = QGroupBox(title)
    fl = QFormLayout(); fl.setSpacing(9); fl.setContentsMargins(12, 20, 12, 12)
    g.setLayout(fl); return g

def _metric(val_str, key_str, color="#d8dce8"):
    w = QWidget()
    w.setStyleSheet("background:#131825;border:1px solid #1e2540;border-radius:8px;")
    v = QVBoxLayout(w); v.setContentsMargins(14, 10, 14, 10); v.setSpacing(1)
    vl = QLabel(val_str)
    vl.setStyleSheet(f"font-size:22px;font-weight:300;color:{color};border:none;")
    kl = QLabel(key_str)
    kl.setStyleSheet("font-size:9px;color:#3a4060;letter-spacing:0.8px;border:none;")
    v.addWidget(vl); v.addWidget(kl); return w


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._scenario   = deepcopy(DEFAULT)
        self._history: list[dict] = []
        self._worker: BenchmarkWorker | None = None
        self._health_worker: HealthCheckWorker | None = None
        self._target_info: dict = {}
        self.setWindowTitle("AcceleratorLab Console Pro")
        self.setMinimumSize(1100, 740)
        self.resize(1360, 900)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "Ready  —  Choose a model and target on the Configure tab, "
            "then click Run Benchmark.")
        self._build_ui()

    # ── UI skeleton ──────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        hl   = QHBoxLayout(root); hl.setSpacing(0); hl.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(root)

        sidebar = self._build_sidebar()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(218)
        hl.addWidget(sidebar)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_configure_tab(), "  ⚙   Configure  ")
        self._tabs.addTab(self._build_run_tab(),       "  ▶   Run Benchmark  ")
        self._tabs.addTab(self._build_results_tab(),   "  📊  Results  ")
        self._tabs.addTab(self._build_export_tab(),    "  💾  Export  ")
        hl.addWidget(self._tabs, 1)

    # ── Sidebar ──────────────────────────────────────────────────────

    def _build_sidebar(self):
        sb = QWidget(); v = QVBoxLayout(sb)
        v.setContentsMargins(14, 18, 14, 18); v.setSpacing(8)

        tl = QLabel("AcceleratorLab"); tl.setObjectName("app_title"); v.addWidget(tl)
        vl = QLabel("Console Pro  v0.4.0"); vl.setObjectName("app_ver"); v.addWidget(vl)
        v.addWidget(self._divider())

        v.addWidget(_lbl("PROJECT",
            "color:#3a4060;font-size:10px;font-weight:700;letter-spacing:1.2px;"))
        self._proj_name = QLineEdit("my-benchmark"); v.addWidget(self._proj_name)

        row = QHBoxLayout(); row.setSpacing(6)
        btn_save = QPushButton("Save"); btn_save.setObjectName("btn_accent")
        btn_save.setToolTip("Save scenario and results to ~/.scalelab_projects/")
        btn_save.clicked.connect(self._save_project); row.addWidget(btn_save)
        self._proj_combo = QComboBox()
        self._proj_combo.setPlaceholderText("Load…")
        self._refresh_projects()
        self._proj_combo.currentTextChanged.connect(self._load_project)
        row.addWidget(self._proj_combo, 1); v.addLayout(row)

        v.addSpacing(4)
        btn_demo = QPushButton("Load demo results"); btn_demo.setObjectName("btn_green")
        btn_demo.setToolTip("Populate the Results tab with sample data")
        btn_demo.clicked.connect(self._load_demo); v.addWidget(btn_demo)

        v.addWidget(self._divider()); v.addSpacing(2)
        v.addWidget(_lbl("HOW TO START",
            "color:#3a4060;font-size:10px;font-weight:700;letter-spacing:1.2px;"))

        steps = [
            ("1  Choose a model",    "Browse 30+ open-source models on the Configure tab."),
            ("2  Choose a target",   "Pick a local GPU or cloud instance preset."),
            ("3  Set endpoint",      "Point to your running inference server URL."),
            ("4  Run",               "Go to Run Benchmark — if no server is running, "
                                     "the app will guide you through setting one up."),
            ("5  Compare",           "Charts appear on the Results tab."),
        ]
        for step, tip in steps:
            sl = QLabel(step)
            sl.setStyleSheet("color:#5b8af5;font-size:11px;font-weight:700;border:none;")
            tl2 = QLabel(tip); tl2.setWordWrap(True)
            tl2.setStyleSheet("color:#3a4060;font-size:10px;border:none;padding-bottom:4px;")
            v.addWidget(sl); v.addWidget(tl2)

        v.addStretch(); return sb

    @staticmethod
    def _divider():
        d = QWidget(); d.setFixedHeight(1)
        d.setStyleSheet("background:#1a2035;"); return d

    def _refresh_projects(self):
        self._proj_combo.blockSignals(True)
        self._proj_combo.clear(); self._proj_combo.addItem("")
        for p in list_projects(): self._proj_combo.addItem(p)
        self._proj_combo.blockSignals(False)

    # ── Tab 1: Configure ─────────────────────────────────────────────

    def _build_configure_tab(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        v = QVBoxLayout(content); v.setContentsMargins(20, 16, 20, 20); v.setSpacing(12)

        hint = QLabel(
            "Choose your model and target hardware, configure traffic settings, "
            "then go to Run Benchmark. If you don't have an inference server running yet, "
            "the app will guide you through setting one up.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#4a5270;font-size:11px;")
        v.addWidget(hint)

        # Model picker
        mg = _group("AI Model")
        ml = mg.layout()
        self._model_display = QLabel(self._scenario["workload"]["model"])
        self._model_display.setStyleSheet(
            "font-size:13px;color:#5b8af5;"
            "font-family:'Cascadia Code','Consolas','Courier New',monospace;border:none;")
        self._model_display.setWordWrap(True)
        btn_model = QPushButton("Browse models…"); btn_model.setObjectName("btn_accent")
        btn_model.clicked.connect(self._pick_model)
        self._backend_combo = _combo(
            ["vllm", "sglang", "tgi", "openai-compat", "tensorrt-llm"],
            self._scenario["workload"]["backend"])
        ml.addRow(_lbl("Selected model:"), self._model_display)
        ml.addRow("", btn_model)
        ml.addRow(_lbl("Serving backend:"), self._backend_combo)
        v.addWidget(mg)

        # Target picker
        ag = _group("Accelerator Target")
        al = ag.layout()
        self._target_display = QLabel("No target selected — click Choose Accelerator")
        self._target_display.setStyleSheet("font-size:12px;color:#f5a623;border:none;")
        btn_target = QPushButton("Choose accelerator…"); btn_target.setObjectName("btn_accent")
        btn_target.clicked.connect(self._pick_target)
        al.addRow(_lbl("Target:"), self._target_display)
        al.addRow("", btn_target)
        self._vendor_combo   = _combo(["nvidia", "amd", "other"],
                                       self._scenario["cluster"]["accelerator_vendor"])
        self._arch_edit      = QLineEdit(self._scenario["cluster"]["accelerator_arch"])
        self._nodes_spin     = _spin(1, 1024, self._scenario["cluster"]["nodes"])
        self._gpus_spin      = _spin(1, 64,   self._scenario["cluster"]["accelerators_per_node"])
        self._intercon_combo = _combo(["ethernet","infiniband","nvlink","xgmi","other"],
                                       self._scenario["cluster"]["interconnect"])
        al.addRow(_lbl("Vendor:"),         self._vendor_combo)
        al.addRow(_lbl("Architecture:"),   self._arch_edit)
        al.addRow(_lbl("Nodes:"),          self._nodes_spin)
        al.addRow(_lbl("GPUs per node:"),  self._gpus_spin)
        al.addRow(_lbl("Interconnect:"),   self._intercon_combo)
        v.addWidget(ag)

        # Endpoint
        eg = _group("Server Endpoint")
        el = eg.layout()
        self._endpoint_edit = QLineEdit(self._scenario["workload"]["endpoint"])
        self._endpoint_edit.setToolTip(
            "The OpenAI-compatible API endpoint to benchmark\n"
            "Ollama: http://localhost:11434/v1\n"
            "vLLM:   http://127.0.0.1:8000/v1\n"
            "LM Studio: http://localhost:1234/v1")
        self._apikey_edit = QLineEdit(self._scenario["workload"]["api_key"])
        self._apikey_edit.setEchoMode(QLineEdit.EchoMode.Password)
        el.addRow(_lbl("Endpoint URL:"), self._endpoint_edit)
        el.addRow(_lbl("API key:"),      self._apikey_edit)
        v.addWidget(eg)

        # Traffic
        tg = _group("Traffic Settings")
        tl = tg.layout()
        self._prompt_spin   = _spin(128,    32768,  self._scenario["workload"]["prompt_tokens"])
        self._output_spin   = _spin(16,     4096,   self._scenario["workload"]["output_tokens"])
        self._conc_spin     = _spin(1,      1024,   self._scenario["workload"]["concurrency"])
        self._req_spin      = _spin(1,      100000, self._scenario["workload"]["requests"])
        self._dur_spin      = _spin(10,     86400,  self._scenario["workload"]["duration_s"])
        self._pattern_combo = _combo(["steady", "burst"],
                                      self._scenario["workload"]["traffic_pattern"])
        tl.addRow(_lbl("Input tokens:"),        self._prompt_spin)
        tl.addRow(_lbl("Output tokens:"),       self._output_spin)
        tl.addRow(_lbl("Concurrent requests:"), self._conc_spin)
        tl.addRow(_lbl("Total requests:"),      self._req_spin)
        tl.addRow(_lbl("Duration cap (s):"),    self._dur_spin)
        tl.addRow(_lbl("Traffic pattern:"),     self._pattern_combo)
        v.addWidget(tg)

        # SLOs
        sg = _group("Performance Targets (SLO)")
        sl2 = sg.layout()
        self._ttft_spin = _spin(1, 60000,  self._scenario["workload"]["target_ttft_ms"])
        self._p95_spin  = _spin(1, 120000, self._scenario["workload"]["target_p95_ms"])
        sl2.addRow(_lbl("Target TTFT (ms):"), self._ttft_spin)
        sl2.addRow(_lbl("Target p95 (ms):"),  self._p95_spin)
        v.addWidget(sg)

        # Launch
        lg = _group("Launch Settings")
        ll = lg.layout()
        self._executor_combo = _combo(["local","ssh","slurm"],
                                       self._scenario["launch"]["executor"])
        self._tp_spin   = _spin(1, 64, self._scenario["launch"]["tensor_parallel"])
        self._pp_spin   = _spin(1, 64, self._scenario["launch"]["pipeline_parallel"])
        self._extra_edit = QLineEdit(" ".join(self._scenario["launch"]["extra_args"]))
        self._extra_edit.setPlaceholderText("e.g. --dtype bfloat16 --device rocm")
        ll.addRow(_lbl("Executor:"),           self._executor_combo)
        ll.addRow(_lbl("Tensor parallel:"),    self._tp_spin)
        ll.addRow(_lbl("Pipeline parallel:"),  self._pp_spin)
        ll.addRow(_lbl("Extra backend args:"), self._extra_edit)
        v.addWidget(lg)

        scroll.setWidget(content); return scroll

    def _pick_model(self):
        dlg = ModelPickerDialog(
            current_id=self._scenario["workload"]["model"], parent=self)
        if dlg.exec():
            self._scenario["workload"]["model"] = dlg.selected_id
            self._model_display.setText(dlg.selected_id)
            self.statusBar().showMessage(f"Model set: {dlg.selected_id}")

    def _pick_target(self):
        dlg = TargetPickerDialog(current=self._target_info, parent=self)
        if dlg.exec():
            t = dlg.result_target
            self._target_info = t
            self._scenario["cluster"]["accelerator_vendor"]    = t.get("vendor", "nvidia")
            self._scenario["cluster"]["accelerator_arch"]      = t.get("arch", "unknown")
            self._scenario["cluster"]["accelerators_per_node"] = t.get("accelerators_per_node", 8)
            label = t.get("instance_label", "selected target")
            tag   = "☁" if t.get("target_type") == "cloud" else "💻"
            self._target_display.setText(
                f"{tag}  {label}  —  "
                f"{t.get('vendor','').upper()} {t.get('arch','')}")
            self._target_display.setStyleSheet("font-size:12px;color:#3ecf8e;border:none;")
            self._vendor_combo.setCurrentText(t.get("vendor", "nvidia"))
            self._arch_edit.setText(t.get("arch", ""))
            self._gpus_spin.setValue(t.get("accelerators_per_node", 8))
            self.statusBar().showMessage(f"Target set: {label}")

    # ── Tab 2: Run ───────────────────────────────────────────────────

    def _build_run_tab(self):
        w = QWidget()
        v = QVBoxLayout(w); v.setContentsMargins(28, 24, 28, 24); v.setSpacing(14)

        title = QLabel("Run Benchmark")
        title.setStyleSheet("font-size:20px;font-weight:300;color:#d8dce8;")
        v.addWidget(title)

        hint = QLabel(
            "AcceleratorLab will check whether an inference server is running at your "
            "configured endpoint. If none is found, it will guide you through setting one up. "
            "Once the server is ready, click Run to start the benchmark.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#4a5270;font-size:11px;")
        v.addWidget(hint)

        self._summary_lbl = QLabel()
        self._summary_lbl.setWordWrap(True)
        self._summary_lbl.setStyleSheet(
            "background:#131825;border:1px solid #1e2540;border-radius:8px;"
            "padding:14px 16px;color:#8892c8;font-size:13px;")
        self._update_summary()
        v.addWidget(self._summary_lbl)

        self._launch_check = QCheckBox(
            "Launch serving backend via executor before benchmarking")
        self._launch_check.setToolTip(
            "When checked, AcceleratorLab starts the model server for you using the executor, "
            "then waits for it to become ready before sending traffic.")
        v.addWidget(self._launch_check)

        self._run_btn = QPushButton("Run Benchmark Now")
        self._run_btn.setObjectName("btn_run")
        self._run_btn.clicked.connect(self._preflight_check)
        v.addWidget(self._run_btn)

        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet("color:#4a5270;font-size:11px;")
        self._progress_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._progress_lbl)

        self._result_preview = QTextEdit()
        self._result_preview.setObjectName("code")
        self._result_preview.setReadOnly(True)
        self._result_preview.setPlaceholderText("Benchmark results will appear here…")
        self._result_preview.setMaximumHeight(260)
        v.addWidget(self._result_preview)
        v.addStretch(); return w

    def _update_summary(self):
        w = self._scenario["workload"]
        c = self._scenario["cluster"]
        target = self._target_info.get(
            "instance_label",
            f"{c['accelerator_vendor']} {c['accelerator_arch']}")
        self._summary_lbl.setText(
            f"Model:   <b style='color:#d8dce8'>{w['model']}</b><br>"
            f"Target:  <b style='color:#d8dce8'>{target}</b><br>"
            f"Backend: {w['backend']}  ·  Pattern: {w['traffic_pattern']}  ·  "
            f"Concurrency: {w['concurrency']}  ·  Requests: {w['requests']}  ·  "
            f"Duration cap: {w['duration_s']}s<br>"
            f"Endpoint: <span style='color:#9ab0e8'>{w['endpoint']}</span>")

    # ── Pre-flight health check ───────────────────────────────────────

    def _preflight_check(self):
        self._collect_scenario()
        self._update_summary()
        self._run_btn.setEnabled(False)
        self._progress_lbl.setText("Checking server connection…")
        self.statusBar().showMessage("Checking server connection…")
        endpoint = self._scenario["workload"]["endpoint"]
        self._health_worker = HealthCheckWorker(endpoint)
        self._health_worker.result.connect(self._on_health_result)
        self._health_worker.start()

    def _on_health_result(self, reachable: bool):
        self._run_btn.setEnabled(True)
        self._progress_lbl.setText("")
        if reachable:
            self._launch_benchmark()
        else:
            self._show_server_setup()

    def _show_server_setup(self):
        vram_gb  = self._target_info.get("vram_gb",  0)
        gpu_name = self._target_info.get("instance_label", "")
        endpoint = self._scenario["workload"]["endpoint"]

        dlg = ServerSetupDialog(
            endpoint=endpoint,
            vram_gb=vram_gb,
            gpu_name=gpu_name,
            parent=self,
        )
        # If user picks a different endpoint in the dialog, update our fields
        dlg.endpoint_updated.connect(self._apply_suggested_endpoint)

        result = dlg.exec()
        if result == QDialog.DialogCode.Accepted:
            # User clicked "Run benchmark now" after confirming connection
            self._launch_benchmark()
        else:
            # User clicked "Proceed anyway"
            self._launch_benchmark()

    def _apply_suggested_endpoint(self, endpoint: str, api_key: str):
        self._endpoint_edit.setText(endpoint)
        self._apikey_edit.setText(api_key)
        self._scenario["workload"]["endpoint"] = endpoint
        self._scenario["workload"]["api_key"]  = api_key
        self._update_summary()
        self.statusBar().showMessage(f"Endpoint updated to {endpoint}")

    def _launch_benchmark(self):
        self._run_btn.setEnabled(False)
        self._progress_lbl.setText("Benchmark running… this may take a minute or two.")
        self.statusBar().showMessage("Benchmark in progress…")
        self._worker = BenchmarkWorker(self._scenario, self._launch_check.isChecked())
        self._worker.finished.connect(self._on_run_finished)
        self._worker.error.connect(self._on_run_error)
        self._worker.start()

    def _on_run_finished(self, result):
        self._run_btn.setEnabled(True)
        self._progress_lbl.setText("")
        br = result.get("benchmark_result", {})
        self._result_preview.setPlainText(json.dumps(result, indent=2))
        rows = normalize_results([result])
        self._history.extend(rows)
        self._last_result = result
        self._refresh_results_tab()
        slo = "SLO MET" if br.get("meets_slo") else "SLO MISSED"
        self.statusBar().showMessage(
            f"Done  ·  {br.get('tok_s', 0):.0f} tok/s  ·  "
            f"TTFT {br.get('ttft_ms', 0):.0f} ms  ·  {slo}")
        self._tabs.setCurrentIndex(2)

    def _on_run_error(self, msg):
        self._run_btn.setEnabled(True)
        self._progress_lbl.setText("")
        QMessageBox.critical(self, "Benchmark error", msg)
        self.statusBar().showMessage(f"Error: {msg[:80]}")

    # ── Tab 3: Results ───────────────────────────────────────────────

    def _build_results_tab(self):
        w = QWidget()
        v = QVBoxLayout(w); v.setContentsMargins(20, 16, 20, 20); v.setSpacing(12)

        self._metric_row = QHBoxLayout(); self._metric_row.setSpacing(10)
        self._m_toks = _metric("—", "BEST TOK/S")
        self._m_ttft = _metric("—", "AVG TTFT ms")
        self._m_p95  = _metric("—", "AVG P95 ms")
        self._m_slo  = _metric("—", "SLO PASS RATE")
        for m in [self._m_toks, self._m_ttft, self._m_p95, self._m_slo]:
            self._metric_row.addWidget(m)
        v.addLayout(self._metric_row)

        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels(
            ["System","Model","Backend","Concurrency","tok/s","TTFT ms","p95 ms","SLO"])
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMaximumHeight(200)
        v.addWidget(self._table)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._chart_toks = ChartCanvas(5.5, 3.5)
        self._chart_p95  = ChartCanvas(5.5, 3.5)
        splitter.addWidget(self._chart_toks)
        splitter.addWidget(self._chart_p95)
        v.addWidget(splitter, 1)

        self._chart_ttft = ChartCanvas(12, 3)
        v.addWidget(self._chart_ttft)
        return w

    def _refresh_results_tab(self):
        rows = self._history
        if not rows: return
        import statistics
        toks  = [r["tok_s"]    for r in rows]
        ttfts = [r["ttft_ms"]  for r in rows]
        p95s  = [r["p95_ms"]   for r in rows]
        slos  = [r["meets_slo"] for r in rows]

        best_c = "#3ecf8e" if max(toks) > 0 else "#d8dce8"
        self._m_toks.findChildren(QLabel)[0].setText(f"{max(toks):,.0f}")
        self._m_toks.findChildren(QLabel)[0].setStyleSheet(
            f"font-size:22px;font-weight:300;color:{best_c};border:none;")
        self._m_ttft.findChildren(QLabel)[0].setText(
            f"{statistics.mean(ttfts):,.0f}")
        self._m_p95.findChildren(QLabel)[0].setText(
            f"{statistics.mean(p95s):,.0f}")
        rate = sum(slos) / max(len(slos), 1) * 100
        sc = "#3ecf8e" if rate >= 95 else "#f5a623" if rate >= 70 else "#e05c5c"
        self._m_slo.findChildren(QLabel)[0].setText(f"{rate:.0f}%")
        self._m_slo.findChildren(QLabel)[0].setStyleSheet(
            f"font-size:22px;font-weight:300;color:{sc};border:none;")

        self._table.setRowCount(0)
        for r in rows:
            row = self._table.rowCount(); self._table.insertRow(row)
            for col, val in enumerate([
                r["system"], r["model"], r["backend"],
                str(r["concurrency"]),
                f"{r['tok_s']:.0f}", f"{r['ttft_ms']:.0f}",
                f"{r['p95_ms']:.0f}",
                "✓" if r["meets_slo"] else "✗",
            ]):
                item = QTableWidgetItem(val)
                if col == 7:
                    from PyQt6.QtGui import QColor
                    item.setForeground(
                        QColor("#3ecf8e" if val == "✓" else "#e05c5c"))
                self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()

        by_system: dict = {}
        for r in rows:
            s = r["system"]
            by_system.setdefault(s, {"toks":[], "p95s":[], "ttfts":[], "concs":[]})
            by_system[s]["toks"].append(r["tok_s"])
            by_system[s]["p95s"].append(r["p95_ms"])
            by_system[s]["ttfts"].append(r["ttft_ms"])
            by_system[s]["concs"].append(r["concurrency"])

        systems    = list(by_system.keys())
        best_toks  = [max(by_system[s]["toks"]) for s in systems]
        self._chart_toks.bar(systems, best_toks, "Peak throughput by system", "Tokens / sec")

        p95_series = {s: (by_system[s]["concs"], by_system[s]["p95s"]) for s in systems}
        self._chart_p95.lines(p95_series, "p95 latency by concurrency",
                               "Concurrency", "p95 ms")

        mean_ttfts = [
            sum(by_system[s]["ttfts"]) / len(by_system[s]["ttfts"]) for s in systems]
        self._chart_ttft.hbar(systems, mean_ttfts, "Mean TTFT by system", "TTFT ms")

    # ── Tab 4: Export ─────────────────────────────────────────────────

    def _build_export_tab(self):
        w = QWidget()
        v = QVBoxLayout(w); v.setContentsMargins(20, 16, 20, 20); v.setSpacing(12)
        v.addWidget(_lbl("Scenario YAML",
            "font-size:14px;font-weight:600;color:#d8dce8;"))
        self._yaml_preview = QTextEdit(); self._yaml_preview.setObjectName("code")
        self._yaml_preview.setReadOnly(True); self._yaml_preview.setMaximumHeight(220)
        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        btn_refresh = QPushButton("Refresh YAML")
        btn_refresh.clicked.connect(self._refresh_yaml)
        btn_save_yaml = QPushButton("Save YAML…"); btn_save_yaml.setObjectName("btn_accent")
        btn_save_yaml.clicked.connect(self._save_yaml)
        btn_row.addWidget(btn_refresh); btn_row.addWidget(btn_save_yaml); btn_row.addStretch()
        v.addWidget(self._yaml_preview); v.addLayout(btn_row)
        v.addWidget(self._divider())
        v.addWidget(_lbl("Last run result JSON",
            "font-size:14px;font-weight:600;color:#d8dce8;"))
        self._json_preview = QTextEdit(); self._json_preview.setObjectName("code")
        self._json_preview.setReadOnly(True); self._json_preview.setMaximumHeight(220)
        self._json_preview.setPlaceholderText("No runs yet.")
        btn_row2 = QHBoxLayout(); btn_row2.setSpacing(8)
        btn_save_json = QPushButton("Save JSON…"); btn_save_json.setObjectName("btn_accent")
        btn_save_json.clicked.connect(self._save_json)
        btn_row2.addWidget(btn_save_json); btn_row2.addStretch()
        v.addWidget(self._json_preview); v.addLayout(btn_row2)
        v.addStretch()
        self._refresh_yaml(); return w

    def _refresh_yaml(self):
        self._collect_scenario()
        self._yaml_preview.setPlainText(
            yaml.safe_dump({"scenario": self._scenario}, sort_keys=False))

    def _save_yaml(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save scenario YAML", "scenario.yaml", "YAML (*.yaml *.yml)")
        if path:
            with open(path, "w") as f:
                yaml.safe_dump({"scenario": self._scenario}, f, sort_keys=False)
            self.statusBar().showMessage(f"Saved: {path}")

    def _save_json(self):
        if not hasattr(self, "_last_result"):
            QMessageBox.information(self, "No result", "Run a benchmark first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result JSON", "benchmark_result.json", "JSON (*.json)")
        if path:
            with open(path, "w") as f:
                json.dump(self._last_result, f, indent=2)
            self.statusBar().showMessage(f"Saved: {path}")

    # ── Project management ────────────────────────────────────────────

    def _collect_scenario(self):
        w = self._scenario["workload"]
        c = self._scenario["cluster"]
        l = self._scenario["launch"]
        w["model"]           = self._model_display.text()
        w["backend"]         = self._backend_combo.currentText()
        w["endpoint"]        = self._endpoint_edit.text()
        w["api_key"]         = self._apikey_edit.text()
        w["prompt_tokens"]   = self._prompt_spin.value()
        w["output_tokens"]   = self._output_spin.value()
        w["concurrency"]     = self._conc_spin.value()
        w["requests"]        = self._req_spin.value()
        w["duration_s"]      = self._dur_spin.value()
        w["traffic_pattern"] = self._pattern_combo.currentText()
        w["target_ttft_ms"]  = self._ttft_spin.value()
        w["target_p95_ms"]   = self._p95_spin.value()
        c["accelerator_vendor"]    = self._vendor_combo.currentText()
        c["accelerator_arch"]      = self._arch_edit.text()
        c["nodes"]                 = self._nodes_spin.value()
        c["accelerators_per_node"] = self._gpus_spin.value()
        c["interconnect"]          = self._intercon_combo.currentText()
        l["executor"]          = self._executor_combo.currentText()
        l["tensor_parallel"]   = self._tp_spin.value()
        l["pipeline_parallel"] = self._pp_spin.value()
        l["extra_args"]        = [x for x in self._extra_edit.text().split() if x]

    def _save_project(self):
        self._collect_scenario()
        name = self._proj_name.text().strip() or "project"
        save_project(name, {
            "project_name": name,
            "scenario":     self._scenario,
            "run_history":  self._history,
        })
        self._refresh_projects()
        self.statusBar().showMessage(f"Project saved: {name}")

    def _load_project(self, filename):
        if not filename: return
        try:
            p = load_project(filename)
            self._scenario = p.get("scenario", deepcopy(DEFAULT))
            self._history  = p.get("run_history", [])
            self._proj_name.setText(p.get("project_name", "project"))
            self._populate_from_scenario()
            self._refresh_results_tab()
            self.statusBar().showMessage(f"Loaded: {filename}")
        except Exception as e:
            QMessageBox.warning(self, "Load error", str(e))

    def _populate_from_scenario(self):
        w = self._scenario["workload"]
        c = self._scenario["cluster"]
        l = self._scenario["launch"]
        self._model_display.setText(w["model"])
        self._backend_combo.setCurrentText(w["backend"])
        self._endpoint_edit.setText(w["endpoint"])
        self._apikey_edit.setText(w["api_key"])
        self._prompt_spin.setValue(w["prompt_tokens"])
        self._output_spin.setValue(w["output_tokens"])
        self._conc_spin.setValue(w["concurrency"])
        self._req_spin.setValue(w["requests"])
        self._dur_spin.setValue(w["duration_s"])
        self._pattern_combo.setCurrentText(w["traffic_pattern"])
        self._ttft_spin.setValue(w["target_ttft_ms"])
        self._p95_spin.setValue(w["target_p95_ms"])
        self._vendor_combo.setCurrentText(c["accelerator_vendor"])
        self._arch_edit.setText(c["accelerator_arch"])
        self._nodes_spin.setValue(c["nodes"])
        self._gpus_spin.setValue(c["accelerators_per_node"])
        self._intercon_combo.setCurrentText(c.get("interconnect", "ethernet"))
        self._executor_combo.setCurrentText(l["executor"])
        self._tp_spin.setValue(l["tensor_parallel"])
        self._pp_spin.setValue(l["pipeline_parallel"])
        self._extra_edit.setText(" ".join(l.get("extra_args", [])))
        self._update_summary()

    def _load_demo(self):
        demo = load_demo_runs()
        rows = normalize_results(demo)
        self._history.extend(rows)
        self._refresh_results_tab()
        self.statusBar().showMessage("Demo results loaded — see Results tab.")
        self._tabs.setCurrentIndex(2)


# Fix missing QDialog import used in _show_server_setup
from PyQt6.QtWidgets import QDialog
