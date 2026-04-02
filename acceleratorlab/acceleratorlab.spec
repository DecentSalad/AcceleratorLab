# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for AcceleratorLab Console Pro
# Works on both Windows and Linux.
# Run:  pyinstaller acceleratorlab.spec

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['scalelab/gui/app.py'],
    pathex=[str(Path('.').resolve())],
    binaries=[],
    datas=[
        ('scalelab', 'scalelab'),
    ],
    hiddenimports=[
        'scalelab.core.models',
        'scalelab.core.io',
        'scalelab.core.traffic',
        'scalelab.core.planner',
        'scalelab.core.orchestrator',
        'scalelab.core.projects',
        'scalelab.backends.base',
        'scalelab.backends.registry',
        'scalelab.backends.vllm',
        'scalelab.backends.sglang',
        'scalelab.backends.tgi',
        'scalelab.backends.openai_compat',
        'scalelab.backends.tensorrt_llm',
        'scalelab.executors.base',
        'scalelab.executors.local',
        'scalelab.executors.ssh',
        'scalelab.executors.slurm',
        'scalelab.ui.state',
        'scalelab.core.results',
        'scalelab.ui.sample_data',
        'scalelab.gui.theme',
        'scalelab.gui.worker',
        'scalelab.gui.health_worker',
        'scalelab.gui.server_setup',
        'scalelab.gui.charts',
        'scalelab.gui.model_picker',
        'scalelab.gui.target_picker',
        'scalelab.gui.main_window',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_qt',
        'yaml',
        'requests',
        'pandas',
        'concurrent.futures',
        'statistics',
        'subprocess',
        'shlex',
        'pathlib',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'wx', 'streamlit', 'tornado'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AcceleratorLab' if sys.platform == 'win32' else 'AcceleratorLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,            # no terminal window — GUI only
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Windows only:
    icon=None,                # add .ico path here if you have one
    version=None,
)
