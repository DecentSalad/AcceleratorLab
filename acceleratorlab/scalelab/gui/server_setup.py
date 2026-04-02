"""
Server Setup Assistant.
Shown automatically when Run Benchmark is clicked and no server is reachable.
Covers Ollama (recommended for Windows), vLLM, and LM Studio.
VRAM-aware model recommendations are shown when GPU info is available.
"""
from __future__ import annotations
import subprocess
import sys
from typing import Any

import requests
from PyQt6.QtCore    import Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui     import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFrame, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)


# ── VRAM-based model recommendations ────────────────────────────────────────

def _recommended_models(vram_gb: int) -> list[tuple[str, str]]:
    """Return list of (ollama_tag, display_name) sorted best-first for VRAM."""
    if vram_gb >= 80:
        return [
            ("qwen2.5:72b",          "Qwen 2.5 72B  (full precision)"),
            ("llama3.1:70b",         "Llama 3.1 70B  (full precision)"),
            ("qwen2.5:32b",          "Qwen 2.5 32B"),
        ]
    if vram_gb >= 32:
        return [
            ("qwen2.5:32b",          "Qwen 2.5 32B"),
            ("llama3.1:70b-instruct-q4_K_M", "Llama 3.1 70B  (4-bit)"),
            ("qwen2.5:14b",          "Qwen 2.5 14B"),
        ]
    if vram_gb >= 16:
        return [
            ("qwen2.5:14b",          "Qwen 2.5 14B"),
            ("phi4",                 "Phi-4 14B"),
            ("llama3.1:8b",          "Llama 3.1 8B"),
        ]
    if vram_gb >= 8:
        return [
            ("llama3.1:8b",          "Llama 3.1 8B  (recommended)"),
            ("mistral:7b",           "Mistral 7B"),
            ("gemma3:4b",            "Gemma 3 4B"),
        ]
    # Low VRAM or unknown
    return [
        ("llama3.2:3b",          "Llama 3.2 3B"),
        ("llama3.2:1b",          "Llama 3.2 1B"),
        ("phi3.5:mini",          "Phi-3.5 Mini 3.8B"),
    ]


# ── Connection test worker ───────────────────────────────────────────────────

class ConnectionTestWorker(QThread):
    result = pyqtSignal(bool, str)   # (success, message)

    def __init__(self, endpoint: str) -> None:
        super().__init__()
        self._endpoint = endpoint.rstrip("/")

    def run(self) -> None:
        # Try /health first, then /v1/models as fallback
        for path in ["/health", "/v1/models", ""]:
            url = self._endpoint.replace("/v1", "") + path
            try:
                r = requests.get(url, timeout=6)
                if r.ok:
                    self.result.emit(True, f"Connected successfully at {url}")
                    return
            except Exception:
                pass
        self.result.emit(False, f"Could not reach {self._endpoint}")


# ── Shared UI helpers ────────────────────────────────────────────────────────

def _h(text: str, size: int = 13, color: str = "#d8dce8", bold: bool = True) -> QLabel:
    lbl = QLabel(text)
    weight = "700" if bold else "400"
    lbl.setStyleSheet(f"font-size:{size}px;font-weight:{weight};color:{color};border:none;")
    lbl.setWordWrap(True)
    return lbl


def _p(text: str, color: str = "#8892c8") -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"font-size:11px;color:{color};border:none;")
    lbl.setWordWrap(True)
    return lbl


def _cmd_row(cmd: str, label: str = "") -> QWidget:
    """A copyable command block with a Copy button."""
    w = QWidget()
    w.setStyleSheet("background:#1c2030;border:1px solid #252e48;border-radius:5px;")
    h = QHBoxLayout(w); h.setContentsMargins(10, 6, 6, 6); h.setSpacing(8)
    txt = QLabel(cmd)
    txt.setStyleSheet(
        "font-family:'Cascadia Code','Consolas','Courier New',monospace;"
        "font-size:11px;color:#9ab0e8;border:none;")
    txt.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    h.addWidget(txt, 1)
    btn = QPushButton("Copy")
    btn.setFixedWidth(52)
    btn.setStyleSheet(
        "QPushButton{background:#252e48;border:none;border-radius:4px;"
        "padding:3px 8px;font-size:10px;color:#8892c8;}"
        "QPushButton:hover{background:#3a4570;color:#d8dce8;}")
    btn.clicked.connect(lambda: QApplication.clipboard().setText(cmd))
    h.addWidget(btn)
    return w


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "font-size:10px;font-weight:700;color:#3a4060;"
        "letter-spacing:1.2px;border:none;padding-top:8px;")
    return lbl


def _link_btn(label: str, url: str) -> QPushButton:
    btn = QPushButton(label)
    btn.setObjectName("btn_accent")
    btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(url)))
    return btn


def _divider() -> QWidget:
    d = QWidget(); d.setFixedHeight(1)
    d.setStyleSheet("background:#1a2035;"); return d


# ── Ollama tab ───────────────────────────────────────────────────────────────

class OllamaTab(QWidget):
    endpoint_chosen = pyqtSignal(str, str)  # (endpoint, api_key)

    def __init__(self, vram_gb: int = 0, parent=None):
        super().__init__(parent)
        self._vram_gb = vram_gb
        self._build()

    def _build(self):
        v = QVBoxLayout(self); v.setContentsMargins(4, 12, 4, 4); v.setSpacing(8)

        v.addWidget(_h("Ollama — recommended for Windows and beginners", 13))
        v.addWidget(_p(
            "Ollama is a single installer that downloads and runs models locally. "
            "It works natively on Windows, macOS, and Linux without Docker or WSL2. "
            "It automatically uses your GPU if one is detected."))

        v.addWidget(_divider())

        is_windows = sys.platform == "win32"
        is_linux   = sys.platform.startswith("linux")

        if is_windows:
            v.addWidget(_section_label("STEP 1 — DOWNLOAD AND INSTALL"))
            v.addWidget(_p("Click the button below to open the Ollama download page. "
                           "Download the installer and run it. It takes about a minute."))
            row = QHBoxLayout(); row.setSpacing(8)
            row.addWidget(_link_btn("Open ollama.com/download", "https://ollama.com/download"))
            row.addStretch()
            v.addLayout(row)
        elif is_linux:
            v.addWidget(_section_label("STEP 1 — INSTALL"))
            v.addWidget(_p("Run this one command in a terminal:"))
            v.addWidget(_cmd_row("curl -fsSL https://ollama.com/install.sh | sh"))
        else:
            v.addWidget(_section_label("STEP 1 — INSTALL"))
            v.addWidget(_link_btn("Open ollama.com/download", "https://ollama.com/download"))

        v.addWidget(_section_label("STEP 2 — DOWNLOAD A MODEL"))
        v.addWidget(_p(
            "After installing, open a Command Prompt (Windows) or terminal (Linux) "
            "and run one of these commands to download a model. "
            + (f"Based on your {self._vram_gb} GB GPU, we recommend:"
               if self._vram_gb > 0 else "Recommended starting models:")))

        models = _recommended_models(self._vram_gb)
        for tag, name in models[:3]:
            row = QHBoxLayout(); row.setSpacing(0)
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet("font-size:11px;color:#6b7294;border:none;min-width:220px;")
            row.addWidget(name_lbl)
            row.addWidget(_cmd_row(f"ollama pull {tag}"), 1)
            container = QWidget(); container.setLayout(row)
            v.addWidget(container)

        v.addWidget(_section_label("STEP 3 — START OLLAMA"))
        v.addWidget(_p(
            "On Windows, Ollama starts automatically after install. "
            "On Linux, run this in a terminal (leave it running):"))
        v.addWidget(_cmd_row("ollama serve"))

        v.addWidget(_section_label("STEP 4 — SET ENDPOINT IN ACCELERATORLAB"))
        v.addWidget(_p("Use these settings on the Configure tab:"))
        v.addWidget(_cmd_row("http://localhost:11434/v1"))

        row2 = QHBoxLayout(); row2.setSpacing(8)
        use_btn = QPushButton("Use Ollama endpoint")
        use_btn.setObjectName("btn_green")
        use_btn.clicked.connect(
            lambda: self.endpoint_chosen.emit("http://localhost:11434/v1", "ollama"))
        row2.addWidget(use_btn); row2.addStretch()
        v.addLayout(row2)

        # Try to detect if Ollama is already installed
        try:
            subprocess.check_output(["ollama", "--version"],
                                     stderr=subprocess.DEVNULL, timeout=4)
            installed_lbl = QLabel("Ollama is already installed on this machine.")
            installed_lbl.setStyleSheet(
                "font-size:11px;font-weight:700;color:#3ecf8e;border:none;"
                "background:#0f2a1e;border:1px solid #1a6a42;border-radius:4px;padding:6px 10px;")
            v.addWidget(installed_lbl)
        except Exception:
            pass

        v.addStretch()


# ── vLLM tab ─────────────────────────────────────────────────────────────────

class VLLMTab(QWidget):
    endpoint_chosen = pyqtSignal(str, str)

    def __init__(self, vram_gb: int = 0, parent=None):
        super().__init__(parent)
        self._vram_gb = vram_gb
        self._build()

    def _build(self):
        v = QVBoxLayout(self); v.setContentsMargins(4, 12, 4, 4); v.setSpacing(8)

        v.addWidget(_h("vLLM — best performance, Linux recommended", 13))
        v.addWidget(_p(
            "vLLM delivers the highest inference throughput and is the primary backend "
            "AcceleratorLab is designed around. On Linux it installs with a single pip command. "
            "On Windows it requires WSL2 (Windows Subsystem for Linux) or Docker."))

        v.addWidget(_divider())

        if sys.platform == "win32":
            v.addWidget(_section_label("WINDOWS — WSL2 REQUIRED"))
            v.addWidget(_p(
                "vLLM does not run natively on Windows. You need WSL2 installed first. "
                "Click below to open Microsoft's WSL2 installation guide:"))
            row = QHBoxLayout(); row.setSpacing(8)
            row.addWidget(_link_btn(
                "Open WSL2 install guide",
                "https://learn.microsoft.com/en-us/windows/wsl/install"))
            row.addStretch()
            v.addLayout(row)
            v.addWidget(_p(
                "Once WSL2 is set up, open a WSL2 terminal and follow the Linux steps below."))
            v.addWidget(_divider())

        v.addWidget(_section_label("STEP 1 — INSTALL VLLM"))
        v.addWidget(_cmd_row("pip install vllm"))

        v.addWidget(_section_label("STEP 2 — START THE SERVER"))
        models = _recommended_models(self._vram_gb)
        first_hf = {
            "llama3.1:8b":          "meta-llama/Llama-3.1-8B-Instruct",
            "llama3.1:70b":         "meta-llama/Llama-3.1-70B-Instruct",
            "llama3.1:70b-instruct-q4_K_M": "meta-llama/Llama-3.1-70B-Instruct",
            "qwen2.5:72b":          "Qwen/Qwen2.5-72B-Instruct",
            "qwen2.5:32b":          "Qwen/Qwen2.5-32B-Instruct",
            "qwen2.5:14b":          "Qwen/Qwen2.5-14B-Instruct",
            "phi4":                 "microsoft/phi-4",
            "llama3.2:3b":          "meta-llama/Llama-3.2-3B-Instruct",
            "llama3.2:1b":          "meta-llama/Llama-3.2-1B-Instruct",
            "mistral:7b":           "mistralai/Mistral-7B-Instruct-v0.3",
            "gemma3:4b":            "google/gemma-3-4b-it",
            "phi3.5:mini":          "microsoft/Phi-3.5-mini-instruct",
        }.get(models[0][0], "meta-llama/Llama-3.1-8B-Instruct")

        v.addWidget(_p(f"Replace the model ID with the one you want. "
                       f"Based on your hardware we suggest: {models[0][1]}"))
        v.addWidget(_cmd_row(
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {first_hf} --port 8000"))

        v.addWidget(_section_label("STEP 3 — ENDPOINT IN ACCELERATORLAB"))
        v.addWidget(_cmd_row("http://127.0.0.1:8000/v1"))

        row2 = QHBoxLayout(); row2.setSpacing(8)
        use_btn = QPushButton("Use vLLM endpoint")
        use_btn.setObjectName("btn_accent")
        use_btn.clicked.connect(
            lambda: self.endpoint_chosen.emit("http://127.0.0.1:8000/v1", "EMPTY"))
        row2.addWidget(use_btn); row2.addStretch()
        v.addLayout(row2)
        v.addStretch()


# ── LM Studio tab ────────────────────────────────────────────────────────────

class LMStudioTab(QWidget):
    endpoint_chosen = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        v = QVBoxLayout(self); v.setContentsMargins(4, 12, 4, 4); v.setSpacing(8)

        v.addWidget(_h("LM Studio — graphical interface for running models", 13))
        v.addWidget(_p(
            "LM Studio is a desktop app with a graphical model browser and chat interface. "
            "It can also run a local API server compatible with AcceleratorLab. "
            "Works on Windows, macOS, and Linux."))

        v.addWidget(_divider())

        v.addWidget(_section_label("STEP 1 — DOWNLOAD LM STUDIO"))
        row = QHBoxLayout(); row.setSpacing(8)
        row.addWidget(_link_btn("Open lmstudio.ai", "https://lmstudio.ai"))
        row.addStretch()
        v.addLayout(row)

        v.addWidget(_section_label("STEP 2 — DOWNLOAD A MODEL"))
        v.addWidget(_p(
            "Open LM Studio, click the magnifying glass icon on the left, "
            "search for a model (e.g. Llama 3.1 8B), and click Download."))

        v.addWidget(_section_label("STEP 3 — START THE LOCAL SERVER"))
        v.addWidget(_p(
            "Click the server icon (looks like this: <->) on the left sidebar. "
            "Select your downloaded model from the dropdown at the top. "
            "Click Start Server. The server starts on port 1234 by default."))

        v.addWidget(_section_label("STEP 4 — ENDPOINT IN ACCELERATORLAB"))
        v.addWidget(_cmd_row("http://localhost:1234/v1"))

        row2 = QHBoxLayout(); row2.setSpacing(8)
        use_btn = QPushButton("Use LM Studio endpoint")
        use_btn.setObjectName("btn_accent")
        use_btn.clicked.connect(
            lambda: self.endpoint_chosen.emit("http://localhost:1234/v1", "lm-studio"))
        row2.addWidget(use_btn); row2.addStretch()
        v.addLayout(row2)
        v.addStretch()


# ── Main dialog ──────────────────────────────────────────────────────────────

class ServerSetupDialog(QDialog):
    """
    Shown when no inference server is detected at the configured endpoint.
    User can follow setup instructions, test the connection, or proceed anyway.
    """
    endpoint_updated = pyqtSignal(str, str)  # (endpoint, api_key) if user changes it

    def __init__(self, endpoint: str, vram_gb: int = 0,
                 gpu_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("No inference server detected")
        self.setMinimumSize(720, 580)
        self.resize(800, 660)
        self._endpoint  = endpoint
        self._vram_gb   = vram_gb
        self._gpu_name  = gpu_name
        self._test_worker: ConnectionTestWorker | None = None
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 18, 20, 18)
        v.setSpacing(12)

        # ── Header ──
        title = QLabel("No inference server detected")
        title.setStyleSheet("font-size:18px;font-weight:700;color:#e05c5c;")
        v.addWidget(title)

        explanation = QLabel(
            "AcceleratorLab sends benchmark traffic to a running AI model server "
            "at the endpoint URL you configured. No server was found at "
            f"<b style='color:#9ab0e8'>{self._endpoint}</b>. "
            "Choose one of the options below to get a server running on this machine, "
            "then click Test Connection before running the benchmark.")
        explanation.setWordWrap(True)
        explanation.setStyleSheet("font-size:12px;color:#8892c8;border:none;")
        v.addWidget(explanation)

        if self._gpu_name:
            gpu_lbl = QLabel(
                f"Detected GPU: {self._gpu_name}"
                + (f"  ({self._vram_gb} GB VRAM)" if self._vram_gb else ""))
            gpu_lbl.setStyleSheet(
                "font-size:11px;color:#3ecf8e;font-weight:600;"
                "background:#0f2a1e;border:1px solid #1a6a42;"
                "border-radius:4px;padding:5px 10px;")
            v.addWidget(gpu_lbl)

        # ── Tabs ──
        tabs = QTabWidget()
        ollama_tab = OllamaTab(vram_gb=self._vram_gb, parent=self)
        vllm_tab   = VLLMTab(vram_gb=self._vram_gb,   parent=self)
        lms_tab    = LMStudioTab(parent=self)

        scroll_ollama = self._wrap_scroll(ollama_tab)
        scroll_vllm   = self._wrap_scroll(vllm_tab)
        scroll_lms    = self._wrap_scroll(lms_tab)

        tabs.addTab(scroll_ollama, "  Ollama  (recommended)  ")
        tabs.addTab(scroll_vllm,   "  vLLM  ")
        tabs.addTab(scroll_lms,    "  LM Studio  ")

        ollama_tab.endpoint_chosen.connect(self._on_endpoint_chosen)
        vllm_tab.endpoint_chosen.connect(self._on_endpoint_chosen)
        lms_tab.endpoint_chosen.connect(self._on_endpoint_chosen)

        v.addWidget(tabs, 1)

        # ── Connection test row ──
        test_row = QHBoxLayout(); test_row.setSpacing(10)
        self._test_btn = QPushButton("Test connection")
        self._test_btn.setObjectName("btn_accent")
        self._test_btn.clicked.connect(self._test_connection)
        test_row.addWidget(self._test_btn)
        self._test_status = QLabel("Click Test Connection after starting your server.")
        self._test_status.setStyleSheet("font-size:11px;color:#4a5270;border:none;")
        test_row.addWidget(self._test_status, 1)
        v.addLayout(test_row)

        # ── Bottom buttons ──
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        btn_row.addStretch()
        proceed = QPushButton("Proceed anyway (results will be zero)")
        proceed.setStyleSheet(
            "QPushButton{background:#1c2030;border:1px solid #252e48;"
            "border-radius:6px;padding:7px 18px;color:#4a5270;font-size:11px;}"
            "QPushButton:hover{color:#d8dce8;border-color:#5b8af5;}")
        proceed.clicked.connect(self.reject)
        btn_row.addWidget(proceed)
        self._run_btn = QPushButton("Connection confirmed — run benchmark")
        self._run_btn.setObjectName("btn_run")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._run_btn)
        v.addLayout(btn_row)

    @staticmethod
    def _wrap_scroll(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(widget)
        return scroll

    def _on_endpoint_chosen(self, endpoint: str, api_key: str) -> None:
        self._endpoint = endpoint
        self.endpoint_updated.emit(endpoint, api_key)
        self._test_status.setText(
            f"Endpoint set to {endpoint} — click Test Connection to verify.")
        self._test_status.setStyleSheet("font-size:11px;color:#f5a623;border:none;")

    def _test_connection(self) -> None:
        self._test_btn.setEnabled(False)
        self._test_btn.setText("Testing…")
        self._test_status.setText("Connecting…")
        self._test_status.setStyleSheet("font-size:11px;color:#4a5270;border:none;")
        self._test_worker = ConnectionTestWorker(self._endpoint)
        self._test_worker.result.connect(self._on_test_result)
        self._test_worker.start()

    def _on_test_result(self, success: bool, msg: str) -> None:
        self._test_btn.setEnabled(True)
        self._test_btn.setText("Test connection")
        if success:
            self._test_status.setText(f"Connected — server is ready.")
            self._test_status.setStyleSheet(
                "font-size:11px;font-weight:700;color:#3ecf8e;border:none;")
            self._run_btn.setEnabled(True)
            self._run_btn.setText("Run benchmark now")
        else:
            self._test_status.setText(
                "Not reachable yet — check the server is running and try again.")
            self._test_status.setStyleSheet(
                "font-size:11px;color:#e05c5c;border:none;")
            self._run_btn.setEnabled(False)
