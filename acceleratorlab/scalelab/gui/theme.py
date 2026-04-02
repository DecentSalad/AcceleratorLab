DARK = """
QMainWindow, QDialog { background:#0f1420; }
QWidget { background:#0f1420; color:#d8dce8;
  font-family:"Segoe UI","Ubuntu","Helvetica Neue",sans-serif; font-size:13px; }
QScrollArea, QScrollArea>QWidget>QWidget { background:#0f1420; border:none; }
#sidebar { background:#090d18; border-right:1px solid #1a2035; }
#sidebar QLabel#app_title { font-size:16px; font-weight:700; color:#5b8af5; }
#sidebar QLabel#app_ver   { font-size:10px; color:#3a4060; }
QGroupBox { background:#131825; border:1px solid #1e2540; border-radius:8px;
  margin-top:14px; padding:10px 12px 12px; font-size:11px; font-weight:600; color:#6b7294; }
QGroupBox::title { subcontrol-origin:margin; subcontrol-position:top left;
  left:12px; top:-1px; padding:0 6px; background:#0f1420;
  color:#6b7294; font-size:10px; font-weight:700; letter-spacing:1.2px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
  background:#1c2030; border:1px solid #252e48; border-radius:5px;
  padding:5px 8px; color:#d8dce8; selection-background-color:#2c4db8; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus { border-color:#5b8af5; }
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color:#344060; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button
  { background:#252e48; border:none; width:18px; }
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background:#3a4570; }
QComboBox { background:#1c2030; border:1px solid #252e48; border-radius:5px;
  padding:5px 8px; color:#d8dce8; min-width:120px; }
QComboBox:hover { border-color:#344060; } QComboBox:focus { border-color:#5b8af5; }
QComboBox::drop-down { border:none; width:24px; }
QComboBox QAbstractItemView { background:#1c2030; border:1px solid #252e48;
  selection-background-color:#2c4db8; color:#d8dce8; outline:none; }
QSlider::groove:horizontal { height:4px; background:#252e48; border-radius:2px; }
QSlider::handle:horizontal { background:#5b8af5; border:none; width:14px; height:14px;
  margin:-5px 0; border-radius:7px; }
QSlider::handle:horizontal:hover { background:#7ba4f8; }
QSlider::sub-page:horizontal { background:#5b8af5; border-radius:2px; }
QCheckBox { spacing:8px; color:#d8dce8; }
QCheckBox::indicator { width:16px; height:16px; border:1px solid #252e48;
  border-radius:3px; background:#1c2030; }
QCheckBox::indicator:checked { background:#5b8af5; border-color:#5b8af5; }
QPushButton { background:#1c2030; border:1px solid #252e48; border-radius:6px;
  padding:7px 18px; color:#d8dce8; font-weight:500; }
QPushButton:hover { background:#252b40; border-color:#5b8af5; color:#fff; }
QPushButton:pressed { background:#1a2030; } QPushButton:disabled { color:#3a4060; }
QPushButton#btn_run { background:#2c4db8; border:none; border-radius:8px;
  padding:14px 28px; font-size:15px; font-weight:700; color:#fff; min-height:48px; }
QPushButton#btn_run:hover { background:#5b8af5; }
QPushButton#btn_run:disabled { background:#1a2035; color:#3a4060; }
QPushButton#btn_accent { background:#1a2e78; border:1px solid #2c4db8; color:#8ab2ff; }
QPushButton#btn_accent:hover { background:#2c4db8; color:#fff; }
QPushButton#btn_green { background:#0f2a1e; border:1px solid #1a6a42; color:#3ecf8e; }
QPushButton#btn_green:hover { background:#163822; }
QPushButton#btn_pick { background:#1c2030; border:1px solid #344060;
  border-radius:6px; padding:8px 14px; color:#8892c8; text-align:left; }
QPushButton#btn_pick:hover { border-color:#5b8af5; color:#d8dce8; }
QTabWidget::pane { background:#0f1420; border:none; border-top:1px solid #1a2035; }
QTabBar::tab { background:#090d18; color:#4a5270; padding:10px 24px;
  border:none; border-bottom:2px solid transparent; font-size:12px;
  font-weight:600; min-width:140px; }
QTabBar::tab:hover { color:#d8dce8; background:#0f1420; }
QTabBar::tab:selected { color:#5b8af5; border-bottom:2px solid #5b8af5; background:#0f1420; }
QTableWidget { background:#131825; alternate-background-color:#1c2030;
  gridline-color:#1a2035; border:1px solid #1a2035; border-radius:6px;
  selection-background-color:#2c4db8; }
QHeaderView::section { background:#1c2030; color:#6b7294; padding:8px 10px;
  border:none; border-bottom:1px solid #252e48; font-size:10px;
  font-weight:700; letter-spacing:0.8px; }
QTableWidget::item { padding:6px 10px; }
QScrollBar:vertical, QScrollBar:horizontal { background:#0f1420; width:8px; height:8px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal
  { background:#252e48; border-radius:4px; min-height:24px; min-width:24px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background:#3a4570; }
QScrollBar::add-line, QScrollBar::sub-line { height:0; width:0; }
QStatusBar { background:#090d18; color:#4a5270; font-size:11px; border-top:1px solid #1a2035; }
QToolTip { background:#1c2030; border:1px solid #252e48; color:#d8dce8;
  padding:6px 10px; border-radius:4px; font-size:12px; }
QDialog { background:#0f1420; }
QFrame#card { background:#131825; border:1px solid #1e2540; border-radius:8px; }
QFrame#card_selected { background:#0e1e3a; border:2px solid #5b8af5; border-radius:8px; }
QFrame#card_hover { background:#1a2235; border:1px solid #344060; border-radius:8px; }
QLabel#hint { color:#4a5270; font-size:11px; }
QLabel#section { color:#6b7294; font-size:10px; font-weight:700; letter-spacing:1.2px; }
QLabel#val_big { font-size:22px; font-weight:300; color:#d8dce8; }
QLabel#val_good { font-size:22px; font-weight:300; color:#3ecf8e; }
QLabel#val_warn { font-size:22px; font-weight:300; color:#f5a623; }
QLabel#val_bad  { font-size:22px; font-weight:300; color:#e05c5c; }
QLabel#key { font-size:10px; color:#4a5270; letter-spacing:0.8px; }
QLabel#tag_nvidia { background:#1a2e18; border:1px solid #2a6a28;
  border-radius:4px; color:#4ecf4e; font-size:10px; font-weight:700; padding:2px 8px; }
QLabel#tag_amd { background:#2e1a18; border:1px solid #6a2a28;
  border-radius:4px; color:#cf4e4e; font-size:10px; font-weight:700; padding:2px 8px; }
QLabel#tag_cloud { background:#1a1e2e; border:1px solid #2a3468;
  border-radius:4px; color:#5b8af5; font-size:10px; font-weight:700; padding:2px 8px; }
QLabel#tag_local { background:#1e1a2e; border:1px solid #4a2a68;
  border-radius:4px; color:#a78bfa; font-size:10px; font-weight:700; padding:2px 8px; }
QTextEdit#code { background:#1c2030; border:1px solid #1e2540; border-radius:6px;
  font-family:"Cascadia Code","Consolas","Courier New",monospace;
  font-size:12px; color:#9ab0e8; padding:10px; }
"""
