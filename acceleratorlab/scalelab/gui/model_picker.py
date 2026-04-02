from __future__ import annotations
from PyQt6.QtCore    import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QWidget, QFrame, QComboBox,
    QGridLayout, QSizePolicy,
)

MODELS = [
    ("Llama 3.x", "Llama 3.2  1B",   "meta-llama/Llama-3.2-1B-Instruct",        "1B",       "~3 GB",   "Ultra-light. Runs on any modern GPU or even CPU."),
    ("Llama 3.x", "Llama 3.2  3B",   "meta-llama/Llama-3.2-3B-Instruct",        "3B",       "~6 GB",   "Strong reasoning at very low memory. Great for edge devices."),
    ("Llama 3.x", "Llama 3.1  8B",   "meta-llama/Llama-3.1-8B-Instruct",        "8B",       "~16 GB",  "Best small all-rounder. The go-to for single-GPU development."),
    ("Llama 3.x", "Llama 3.1  70B",  "meta-llama/Llama-3.1-70B-Instruct",       "70B",      "~140 GB", "Production-quality reasoning. Requires multiple high-end GPUs."),
    ("Llama 3.x", "Llama 3.1  405B", "meta-llama/Llama-3.1-405B-Instruct",      "405B",     "~810 GB", "Frontier open model. Needs a full multi-node cluster."),
    ("Mistral",   "Mistral 7B",       "mistralai/Mistral-7B-Instruct-v0.3",      "7B",       "~14 GB",  "Fast, efficient instruction-following. Excellent throughput."),
    ("Mistral",   "Mistral Nemo 12B", "mistralai/Mistral-Nemo-Instruct-2407",    "12B",      "~24 GB",  "Strong multilingual model. Good accuracy-to-size ratio."),
    ("Mistral",   "Mixtral 8x7B",     "mistralai/Mixtral-8x7B-Instruct-v0.1",   "8x7B MoE", "~90 GB",  "Mixture-of-Experts. High throughput with selective expert activation."),
    ("Mistral",   "Mixtral 8x22B",    "mistralai/Mixtral-8x22B-Instruct-v0.1",  "8x22B MoE","~280 GB", "State-of-the-art MoE. Near-GPT-4 quality on many benchmarks."),
    ("Qwen",      "Qwen 2.5  7B",     "Qwen/Qwen2.5-7B-Instruct",               "7B",       "~14 GB",  "Excellent code and math. Strong multilingual support."),
    ("Qwen",      "Qwen 2.5  14B",    "Qwen/Qwen2.5-14B-Instruct",              "14B",      "~28 GB",  "Great balance of quality and resource use."),
    ("Qwen",      "Qwen 2.5  32B",    "Qwen/Qwen2.5-32B-Instruct",              "32B",      "~64 GB",  "Near-70B quality at lower cost. Popular in production."),
    ("Qwen",      "Qwen 2.5  72B",    "Qwen/Qwen2.5-72B-Instruct",              "72B",      "~144 GB", "Top open-source 70B-class model across reasoning benchmarks."),
    ("Qwen",      "Qwen 3  30B-A3B",  "Qwen/Qwen3-30B-A3B",                     "30B MoE",  "~60 GB",  "Efficient MoE — activates only 3B params per token."),
    ("DeepSeek",  "DeepSeek-R1  7B",  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B","7B",       "~14 GB",  "Reasoning-focused. Distilled from the full R1 model."),
    ("DeepSeek",  "DeepSeek-R1  14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B","14B",     "~28 GB",  "Strong chain-of-thought reasoning at medium size."),
    ("DeepSeek",  "DeepSeek-R1  70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B","70B",    "~140 GB", "Best open reasoning model in the 70B class."),
    ("DeepSeek",  "DeepSeek-V3",      "deepseek-ai/DeepSeek-V3",                "671B MoE", "~700 GB", "Cutting-edge MoE. Requires large multi-node cluster."),
    ("Gemma",     "Gemma 3  1B",      "google/gemma-3-1b-it",                   "1B",       "~2 GB",   "Google's smallest instruction model. Runs on CPU."),
    ("Gemma",     "Gemma 3  4B",      "google/gemma-3-4b-it",                   "4B",       "~8 GB",   "Excellent for single-GPU workstations."),
    ("Gemma",     "Gemma 3  12B",     "google/gemma-3-12b-it",                  "12B",      "~24 GB",  "Strong all-round model. Good coding and reasoning."),
    ("Gemma",     "Gemma 3  27B",     "google/gemma-3-27b-it",                  "27B",      "~54 GB",  "Google's top open model. Competitive with larger open LLMs."),
    ("Phi",       "Phi-4  14B",       "microsoft/phi-4",                         "14B",      "~28 GB",  "Microsoft's flagship small model. Strong reasoning."),
    ("Phi",       "Phi-3.5  Mini 3.8B","microsoft/Phi-3.5-mini-instruct",        "3.8B",     "~8 GB",   "Very capable for its size. Fast and lightweight."),
    ("Phi",       "Phi-3  Medium 14B","microsoft/Phi-3-medium-128k-instruct",    "14B",      "~28 GB",  "128k context window. Good for long-document tasks."),
    ("Falcon",    "Falcon 3  7B",     "tiiuae/Falcon3-7B-Instruct",             "7B",       "~14 GB",  "TII efficient instruction model. Good multilingual support."),
    ("Falcon",    "Falcon 3  10B",    "tiiuae/Falcon3-10B-Instruct",            "10B",      "~20 GB",  "Strong reasoning and code at medium size."),
    ("Falcon",    "Falcon 2  11B",    "tiiuae/falcon-11b",                      "11B",      "~22 GB",  "Multimodal-capable base model from TII."),
    ("Code",      "Qwen 2.5 Coder 7B","Qwen/Qwen2.5-Coder-7B-Instruct",        "7B",       "~14 GB",  "Best-in-class small code model. Great for agentic coding."),
    ("Code",      "Qwen 2.5 Coder 32B","Qwen/Qwen2.5-Coder-32B-Instruct",      "32B",      "~64 GB",  "Top open code model. Beats GPT-4 on several benchmarks."),
    ("Code",      "DeepSeek-Coder-V2","deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct","16B MoE","~32 GB","MoE code model with 128k context. Strong across 338 languages."),
]

FAMILIES = ["All"] + sorted({m[0] for m in MODELS})
FAMILY_COLORS = {
    "Llama 3.x": "#5b8af5", "Mistral": "#a78bfa", "Qwen": "#3ecf8e",
    "DeepSeek":  "#e05c5c",  "Gemma":   "#f5a623", "Phi":  "#5cc8fa",
    "Falcon":    "#f59b23",  "Code":    "#3ecf8e",
}


class ModelCard(QFrame):
    clicked = pyqtSignal(str)   # emits hf_id

    def __init__(self, family, name, hf_id, size, vram, desc, parent=None):
        super().__init__(parent)
        self.hf_id     = hf_id
        self._selected = False
        self.setObjectName("card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(3)

        top = QHBoxLayout(); top.setSpacing(8)
        name_lbl = QLabel(name)
        name_lbl.setStyleSheet("font-size:13px;font-weight:700;color:#d8dce8;border:none;")
        top.addWidget(name_lbl, 1)
        col = FAMILY_COLORS.get(family, "#5b8af5")
        fam_lbl = QLabel(family)
        fam_lbl.setStyleSheet(
            f"font-size:9px;font-weight:700;color:{col};"
            f"background:transparent;border:1px solid {col};"
            "border-radius:3px;padding:1px 6px;")
        top.addWidget(fam_lbl)
        v.addLayout(top)

        id_lbl = QLabel(hf_id)
        id_lbl.setStyleSheet(
            "font-size:10px;color:#4a5270;"
            "font-family:'Cascadia Code','Consolas','Courier New',monospace;border:none;")
        v.addWidget(id_lbl)

        desc_lbl = QLabel(desc)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("font-size:11px;color:#8892c8;border:none;")
        v.addWidget(desc_lbl, 1)

        bot = QHBoxLayout(); bot.setSpacing(12)
        for label, value in [("Params", size), ("Min VRAM", vram)]:
            pair = QHBoxLayout(); pair.setSpacing(4)
            kl = QLabel(label + ":"); kl.setStyleSheet("font-size:9px;color:#3a4060;border:none;")
            vl = QLabel(value);       vl.setStyleSheet("font-size:10px;font-weight:700;color:#6b7294;border:none;")
            pair.addWidget(kl); pair.addWidget(vl)
            bot.addLayout(pair)
        bot.addStretch()
        v.addLayout(bot)

    def set_selected(self, sel: bool) -> None:
        self._selected = sel
        self.setObjectName("card_selected" if sel else "card")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        self.clicked.emit(self.hf_id)
        super().mousePressEvent(event)


class ModelPickerDialog(QDialog):
    def __init__(self, current_id: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose a model")
        self.setMinimumSize(800, 620)
        self.resize(900, 700)
        self.selected_id   = current_id
        self._cards: list[ModelCard] = []
        self._active_card: ModelCard | None = None
        self._build()
        # Pre-select current model if it matches a card
        for card in self._cards:
            if card.hf_id == current_id:
                card.set_selected(True)
                self._active_card = card
                break

    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(10)

        title = QLabel("Choose a model")
        title.setStyleSheet("font-size:18px;font-weight:700;color:#d8dce8;")
        v.addWidget(title)

        hint = QLabel(
            "Select a model to benchmark. The HuggingFace model ID will be passed "
            "to the serving backend. Ensure weights are cached locally or the server "
            "can download them.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#4a5270;font-size:11px;")
        v.addWidget(hint)

        # Filter row
        row = QHBoxLayout(); row.setSpacing(10)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search models…")
        self._search.textChanged.connect(self._filter)
        row.addWidget(self._search, 1)
        self._family_combo = QComboBox()
        self._family_combo.addItems(FAMILIES)
        self._family_combo.setFixedWidth(160)
        self._family_combo.currentTextChanged.connect(self._filter)
        row.addWidget(QLabel("Family:"))
        row.addWidget(self._family_combo)
        v.addLayout(row)

        # Scroll area with a static grid — cards are placed once and never moved
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(10)
        grid.setContentsMargins(0, 0, 0, 0)

        for i, (fam, name, hf_id, size, vram, desc) in enumerate(MODELS):
            card = ModelCard(fam, name, hf_id, size, vram, desc, grid_widget)
            card.clicked.connect(self._on_card_clicked)
            self._cards.append(card)
            grid.addWidget(card, i // 2, i % 2)   # fixed position — never moved

        scroll.setWidget(grid_widget)
        v.addWidget(scroll, 1)

        # Custom ID entry
        cust_row = QHBoxLayout(); cust_row.setSpacing(8)
        cust_row.addWidget(QLabel("Custom HuggingFace ID:"))
        self._custom_edit = QLineEdit()
        self._custom_edit.setPlaceholderText("org/model-name")
        known_ids = {m[2] for m in MODELS}
        self._custom_edit.setText(
            self.selected_id if self.selected_id not in known_ids else "")
        cust_row.addWidget(self._custom_edit, 1)
        v.addLayout(cust_row)

        # Buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        btn_row.addStretch()
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        btn_row.addWidget(cancel)
        ok = QPushButton("Use selected model"); ok.setObjectName("btn_accent")
        ok.clicked.connect(self._accept)
        btn_row.addWidget(ok)
        v.addLayout(btn_row)

    def _on_card_clicked(self, hf_id: str) -> None:
        if self._active_card:
            self._active_card.set_selected(False)
        for card in self._cards:
            if card.hf_id == hf_id:
                card.set_selected(True)
                self._active_card = card
                self.selected_id  = hf_id
                break

    def _filter(self) -> None:
        """Show/hide cards in place — no layout manipulation."""
        query  = self._search.text().lower()
        family = self._family_combo.currentText()
        for i, card in enumerate(self._cards):
            m = MODELS[i]
            match_fam   = (family == "All") or (m[0] == family)
            match_query = (not query) or any(
                query in s.lower() for s in [m[1], m[2], m[0], m[5]])
            card.setVisible(match_fam and match_query)

    def _accept(self) -> None:
        custom = self._custom_edit.text().strip()
        if custom:
            self.selected_id = custom
        if self.selected_id:
            self.accept()
        else:
            self._search.setFocus()
