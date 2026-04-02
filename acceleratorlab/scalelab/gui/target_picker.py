from __future__ import annotations
import subprocess
from PyQt6.QtCore    import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QWidget, QScrollArea, QFrame, QSpinBox,
    QSizePolicy, QFormLayout, QLineEdit, QComboBox,
)

CLOUD_PRESETS = [
    ("AWS",        "p4d.24xlarge",          "A100 40GB SXM",    8, 40,  "nvidia", "a100",
     "High-memory SXM A100s. Standard for large model inference on AWS."),
    ("AWS",        "p4de.24xlarge",         "A100 80GB SXM",    8, 80,  "nvidia", "a100",
     "80GB A100 variant. More headroom for larger models."),
    ("AWS",        "p5.48xlarge",           "H100 80GB SXM",    8, 80,  "nvidia", "h100",
     "AWS flagship H100 instance. Best inference throughput on AWS."),
    ("AWS",        "p5e.48xlarge",          "H200 141GB SXM",   8, 141, "nvidia", "h200",
     "H200 with 141GB HBM3e — highest memory bandwidth on AWS."),
    ("GCP",        "a2-highgpu-8g",         "A100 40GB SXM",    8, 40,  "nvidia", "a100",
     "Google standard A100 instance. Good price-performance."),
    ("GCP",        "a3-highgpu-8g",         "H100 80GB SXM",    8, 80,  "nvidia", "h100",
     "Google H100 offering. NVLink interconnect between GPUs."),
    ("GCP",        "a3-megagpu-8g",         "H100 80GB SXM",    8, 80,  "nvidia", "h100",
     "A3 Mega: 3.2Tbps inter-node NVLink for multi-node workloads."),
    ("GCP",        "a4-highgpu-8g",         "B200 180GB SXM",   8, 180, "nvidia", "b200",
     "Latest Blackwell B200 on GCP. Highest throughput available."),
    ("Azure",      "ND96asr_v4",            "A100 80GB SXM",    8, 80,  "nvidia", "a100",
     "Azure 80GB A100 node. 400Gbps InfiniBand HDR."),
    ("Azure",      "ND96isr_H100_v5",       "H100 80GB SXM",    8, 80,  "nvidia", "h100",
     "Azure H100 instance. High-bandwidth NVLink fabric."),
    ("CoreWeave",  "HGX A100 80GB x8",      "A100 80GB SXM",    8, 80,  "nvidia", "a100",
     "CoreWeave popular A100 cluster. Competitive pricing."),
    ("CoreWeave",  "HGX H100 80GB x8",      "H100 80GB SXM",    8, 80,  "nvidia", "h100",
     "H100 SXM at scale. Good availability for burst workloads."),
    ("CoreWeave",  "HGX H200 141GB x8",     "H200 141GB SXM",   8, 141, "nvidia", "h200",
     "H200 on CoreWeave — excellent for memory-bound 70B+ models."),
    ("Lambda",     "1x A100 40GB",          "A100 40GB PCIe",   1, 40,  "nvidia", "a100",
     "Single A100 — good for 7-13B model development and testing."),
    ("Lambda",     "8x A100 40GB",          "A100 40GB SXM",    8, 40,  "nvidia", "a100",
     "Lambda flagship 8x A100 node. Popular for 70B models."),
    ("Lambda",     "8x H100 80GB",          "H100 80GB SXM",    8, 80,  "nvidia", "h100",
     "Lambda H100 cluster. Best for latency-sensitive production."),
    ("Vultr",      "MI300X x8",             "AMD MI300X 192GB", 8, 192, "amd",    "mi300x",
     "192GB HBM3 per GPU. Exceptional for large context windows."),
    ("FluidStack", "MI325X x8",             "AMD MI325X 256GB", 8, 256, "amd",    "mi325x",
     "Latest AMD MI325X — highest memory capacity available."),
    ("FluidStack", "MI355X x8",             "AMD MI355X 288GB", 8, 288, "amd",    "mi355x",
     "Frontier AMD GPU. Competitive with H200 on many workloads."),
]

PROVIDERS = ["All"] + sorted({p[0] for p in CLOUD_PRESETS})


class DetectWorker(QThread):
    done = pyqtSignal(list)

    def run(self):
        gpus = []
        # NVIDIA
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=8,
            ).decode()
            for i, line in enumerate(out.strip().splitlines()):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    name = parts[0]
                    try:
                        vram = int(parts[1]) // 1024
                    except ValueError:
                        vram = 0
                    drv = parts[2] if len(parts) > 2 else "unknown"
                    gpus.append({"idx": i, "vendor": "nvidia",
                                 "name": name, "vram_gb": vram,
                                 "driver": drv, "source": "nvidia-smi"})
        except Exception:
            pass
        # AMD
        if not gpus:
            try:
                out = subprocess.check_output(
                    ["rocm-smi", "--showproductname", "--csv"],
                    stderr=subprocess.DEVNULL, timeout=8,
                ).decode()
                for i, line in enumerate(out.strip().splitlines()[1:]):
                    if line.strip():
                        parts = line.split(",")
                        name = parts[-1].strip() if parts else "AMD GPU"
                        gpus.append({"idx": i, "vendor": "amd",
                                     "name": name, "vram_gb": 0,
                                     "driver": "ROCm", "source": "rocm-smi"})
            except Exception:
                pass
        self.done.emit(gpus)


class LocalGPUCard(QFrame):
    # Signal carries the gpu info dict index
    clicked = pyqtSignal(int)

    def __init__(self, gpu: dict, index: int, parent=None):
        super().__init__(parent)
        self.gpu_info  = gpu
        self._index    = index
        self._selected = False
        self.setObjectName("card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(80)

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(3)

        top = QHBoxLayout()
        name_l = QLabel(f"GPU {gpu['idx']}  —  {gpu['name']}")
        name_l.setStyleSheet(
            "font-size:13px;font-weight:700;color:#d8dce8;border:none;")
        top.addWidget(name_l, 1)
        tag_id = "tag_nvidia" if gpu["vendor"] == "nvidia" else "tag_amd"
        tag = QLabel(gpu["vendor"].upper())
        tag.setObjectName(tag_id)
        top.addWidget(tag)
        v.addLayout(top)

        sub_parts = []
        if gpu["vram_gb"]:
            sub_parts.append(f"{gpu['vram_gb']} GB VRAM")
        sub_parts.append(f"Driver: {gpu['driver']}")
        sub_parts.append(f"Detected via {gpu['source']}")
        sub = QLabel("  ·  ".join(sub_parts))
        sub.setStyleSheet("font-size:11px;color:#4a5270;border:none;")
        v.addWidget(sub)

    def set_selected(self, sel: bool):
        self._selected = sel
        self.setObjectName("card_selected" if sel else "card")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, e):
        self.clicked.emit(self._index)
        super().mousePressEvent(e)


class CloudPresetCard(QFrame):
    clicked = pyqtSignal(int)

    def __init__(self, preset: tuple, index: int, parent=None):
        super().__init__(parent)
        self.preset    = preset
        self._index    = index
        self._selected = False
        self.setObjectName("card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(100)

        provider, inst, gpu, cnt, vram, vendor, arch, desc = preset
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(3)

        top = QHBoxLayout()
        inst_l = QLabel(inst)
        inst_l.setStyleSheet(
            "font-size:13px;font-weight:700;color:#d8dce8;border:none;")
        top.addWidget(inst_l, 1)
        prov_l = QLabel(provider)
        prov_l.setObjectName("tag_cloud")
        top.addWidget(prov_l)
        v.addLayout(top)

        gpu_l = QLabel(f"{cnt}x  {gpu}  ·  {cnt * vram} GB total VRAM")
        gpu_l.setStyleSheet(
            "font-size:11px;color:#5b8af5;font-weight:600;border:none;")
        v.addWidget(gpu_l)

        desc_l = QLabel(desc)
        desc_l.setWordWrap(True)
        desc_l.setStyleSheet("font-size:11px;color:#8892c8;border:none;")
        v.addWidget(desc_l, 1)

    def set_selected(self, sel: bool):
        self._selected = sel
        self.setObjectName("card_selected" if sel else "card")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, e):
        self.clicked.emit(self._index)
        super().mousePressEvent(e)


class TargetPickerDialog(QDialog):
    def __init__(self, current: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose accelerator target")
        self.setMinimumSize(780, 600)
        self.resize(880, 680)
        self.result_target = current or {}
        self._local_cards: list[LocalGPUCard]    = []
        self._cloud_cards: list[CloudPresetCard] = []
        self._active_local_idx: int  = -1
        self._active_cloud_idx: int  = -1
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(10)

        title = QLabel("Choose accelerator target")
        title.setStyleSheet("font-size:18px;font-weight:700;color:#d8dce8;")
        v.addWidget(title)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_local_tab(),  "  Local GPU  ")
        self._tabs.addTab(self._build_cloud_tab(),  "  Cloud Instance  ")
        v.addWidget(self._tabs, 1)

        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        btn_row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(cancel)
        ok = QPushButton("Use selected target")
        ok.setObjectName("btn_accent")
        ok.clicked.connect(self._accept)
        btn_row.addWidget(ok)
        v.addLayout(btn_row)

    # ── Local tab ────────────────────────────────────────────────────

    def _build_local_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w); v.setContentsMargins(0, 12, 0, 0); v.setSpacing(8)

        hint = QLabel(
            "Click Detect GPUs to scan this machine using nvidia-smi or rocm-smi. "
            "If no GPUs are found, use the manual override below.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#4a5270;font-size:11px;")
        v.addWidget(hint)

        self._detect_btn = QPushButton("  Detect GPUs on this machine")
        self._detect_btn.setObjectName("btn_accent")
        self._detect_btn.clicked.connect(self._detect_gpus)
        v.addWidget(self._detect_btn)

        self._detect_status = QLabel("Click Detect to scan for GPUs.")
        self._detect_status.setStyleSheet("color:#4a5270;font-size:11px;")
        v.addWidget(self._detect_status)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._local_list = QWidget()
        self._local_vl   = QVBoxLayout(self._local_list)
        self._local_vl.setSpacing(8)
        self._local_vl.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._local_list)
        v.addWidget(scroll, 1)

        # Manual override
        form_w = QWidget()
        form_w.setStyleSheet(
            "background:#131825;border:1px solid #1e2540;border-radius:6px;")
        fl = QFormLayout(form_w)
        fl.setContentsMargins(14, 10, 14, 10); fl.setSpacing(8)
        manual_title = QLabel("Manual override (optional)")
        manual_title.setStyleSheet(
            "font-size:12px;font-weight:700;color:#6b7294;")
        fl.addRow(manual_title)
        self._manual_vendor = QComboBox()
        self._manual_vendor.addItems(["nvidia", "amd", "other"])
        fl.addRow(QLabel("Vendor:"), self._manual_vendor)
        self._manual_arch = QLineEdit()
        self._manual_arch.setPlaceholderText("e.g. h100, mi300x, b200")
        fl.addRow(QLabel("Architecture:"), self._manual_arch)
        self._manual_count = QSpinBox()
        self._manual_count.setRange(1, 64); self._manual_count.setValue(8)
        fl.addRow(QLabel("GPU count:"), self._manual_count)
        v.addWidget(form_w)
        return w

    def _detect_gpus(self):
        self._detect_btn.setEnabled(False)
        self._detect_btn.setText("  Detecting…")
        self._detect_status.setText("Scanning for GPUs…")
        self._detect_worker = DetectWorker()
        self._detect_worker.done.connect(self._on_detected)
        self._detect_worker.start()

    def _on_detected(self, gpus: list):
        self._detect_btn.setEnabled(True)
        self._detect_btn.setText("  Detect GPUs on this machine")

        # Clear old cards
        for card in self._local_cards:
            self._local_vl.removeWidget(card)
            card.deleteLater()
        self._local_cards.clear()
        self._active_local_idx = -1

        if gpus:
            self._detect_status.setText(f"Found {len(gpus)} GPU(s) — click one to select it:")
            for i, gpu in enumerate(gpus):
                card = LocalGPUCard(gpu, i, self._local_list)
                card.clicked.connect(self._on_local_clicked)  # signal, not parent chain
                self._local_cards.append(card)
                self._local_vl.addWidget(card)
        else:
            self._detect_status.setText(
                "No GPUs detected (nvidia-smi and rocm-smi not found or returned no devices). "
                "Use the manual override below.")

    def _on_local_clicked(self, index: int):
        if self._active_local_idx >= 0:
            self._local_cards[self._active_local_idx].set_selected(False)
        self._local_cards[index].set_selected(True)
        self._active_local_idx = index

    # ── Cloud tab ─────────────────────────────────────────────────────

    def _build_cloud_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w); v.setContentsMargins(0, 12, 0, 0); v.setSpacing(8)

        hint = QLabel(
            "Select a cloud instance preset. Vendor, GPU architecture, and count "
            "will be set automatically.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#4a5270;font-size:11px;")
        v.addWidget(hint)

        row = QHBoxLayout(); row.setSpacing(10)
        self._cloud_search = QLineEdit()
        self._cloud_search.setPlaceholderText("Search providers or instance types…")
        self._cloud_search.textChanged.connect(self._filter_cloud)
        row.addWidget(self._cloud_search, 1)
        self._provider_combo = QComboBox()
        self._provider_combo.addItems(PROVIDERS)
        self._provider_combo.setFixedWidth(140)
        self._provider_combo.currentTextChanged.connect(self._filter_cloud)
        row.addWidget(QLabel("Provider:")); row.addWidget(self._provider_combo)
        v.addLayout(row)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._cloud_list = QWidget()
        self._cloud_vl   = QVBoxLayout(self._cloud_list)
        self._cloud_vl.setSpacing(8)
        self._cloud_vl.setAlignment(Qt.AlignmentFlag.AlignTop)

        for i, preset in enumerate(CLOUD_PRESETS):
            card = CloudPresetCard(preset, i, self._cloud_list)
            card.clicked.connect(self._on_cloud_clicked)  # signal, not parent chain
            self._cloud_cards.append(card)
            self._cloud_vl.addWidget(card)

        scroll.setWidget(self._cloud_list)
        v.addWidget(scroll, 1)
        return w

    def _filter_cloud(self):
        query    = self._cloud_search.text().lower()
        provider = self._provider_combo.currentText()
        for card in self._cloud_cards:
            p = card.preset
            match_prov  = (provider == "All") or (p[0] == provider)
            match_query = (not query) or any(
                query in s.lower() for s in [p[0], p[1], p[2], p[7]])
            card.setVisible(match_prov and match_query)

    def _on_cloud_clicked(self, index: int):
        if self._active_cloud_idx >= 0:
            self._cloud_cards[self._active_cloud_idx].set_selected(False)
        self._cloud_cards[index].set_selected(True)
        self._active_cloud_idx = index

    # ── Accept ────────────────────────────────────────────────────────

    def _accept(self):
        tab = self._tabs.currentIndex()
        if tab == 0:
            if self._active_local_idx >= 0:
                g = self._local_cards[self._active_local_idx].gpu_info
                self.result_target = {
                    "vendor":                g["vendor"],
                    "arch":                  self._arch_from_name(g["name"]),
                    "accelerators_per_node": 1,
                    "nodes":                 1,
                    "target_type":           "local",
                    "instance_label":        g["name"],
                }
            elif self._manual_arch.text().strip():
                self.result_target = {
                    "vendor":                self._manual_vendor.currentText(),
                    "arch":                  self._manual_arch.text().strip(),
                    "accelerators_per_node": self._manual_count.value(),
                    "nodes":                 1,
                    "target_type":           "local",
                    "instance_label":
                        f"{self._manual_vendor.currentText()} "
                        f"{self._manual_arch.text().strip()}",
                }
            else:
                self._detect_status.setText(
                    "Please detect a GPU or fill in the manual override first.")
                return
        else:
            if self._active_cloud_idx < 0:
                return
            p = self._cloud_cards[self._active_cloud_idx].preset
            self.result_target = {
                "vendor":                p[5],
                "arch":                  p[6],
                "accelerators_per_node": p[3],
                "nodes":                 1,
                "target_type":           "cloud",
                "instance_label":        f"{p[0]}  {p[1]}",
            }
        self.accept()

    @staticmethod
    def _arch_from_name(name: str) -> str:
        name_l = name.lower()
        for kw, arch in [
            ("h200","h200"), ("h100","h100"), ("b200","b200"), ("gb200","gb200"),
            ("a100","a100"), ("a10","a10"),   ("v100","v100"), ("t4","t4"),
            ("mi355","mi355x"), ("mi325","mi325x"), ("mi300","mi300x"),
            ("mi250","mi250x"), ("3090","rtx3090"), ("4090","rtx4090"),
        ]:
            if kw in name_l:
                return arch
        return name.replace(" ", "_").lower()
