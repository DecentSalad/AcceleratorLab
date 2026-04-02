#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  AcceleratorLab Console Pro — Linux build script
#  Run on Ubuntu 22.04+ / Debian 12+ with Python 3.11+ installed.
#  Produces:  dist/AcceleratorLab  (standalone ELF binary)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "=== AcceleratorLab Desktop Build (Linux) ==="
echo

# Ensure Qt platform libs are available
if ! dpkg -l libxcb-cursor0 &>/dev/null; then
    echo "Installing required Qt system libraries..."
    sudo apt-get update -qq
    sudo apt-get install -y libxcb-cursor0 libxcb-xinerama0 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxkbcommon-x11-0 \
        libfontconfig1 libdbus-1-3
fi

# 1. Create venv
python3 -m venv .build_venv
source .build_venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller>=6.3.0

# 3. Run PyInstaller
pyinstaller acceleratorlab.spec --clean --noconfirm

# 4. Make executable
chmod +x dist/AcceleratorLab

# 5. Report
if [ -f dist/AcceleratorLab ]; then
    SIZE=$(du -sh dist/AcceleratorLab | cut -f1)
    echo ""
    echo "================================================================"
    echo " SUCCESS: dist/AcceleratorLab is ready to distribute."
    echo " File size: $SIZE"
    echo " To run:    ./dist/AcceleratorLab"
    echo "================================================================"
else
    echo "ERROR: Build failed. Check output above."
    exit 1
fi

deactivate
