@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  AcceleratorLab Console Pro — Windows build script
REM  Run this on a Windows 10/11 machine with Python 3.11+ installed.
REM  Produces:  dist\AcceleratorLab.exe
REM ─────────────────────────────────────────────────────────────────────────

echo === AcceleratorLab Desktop Build (Windows) ===
echo.

REM 1. Create venv
python -m venv .build_venv
call .build_venv\Scripts\activate.bat

REM 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller>=6.3.0

REM 3. Run PyInstaller
pyinstaller acceleratorlab.spec --clean --noconfirm

REM 4. Report result
if exist dist\AcceleratorLab.exe (
    echo.
    echo ================================================================
    echo  SUCCESS: dist\AcceleratorLab.exe is ready to distribute.
    echo  File size:
    for %%I in (dist\AcceleratorLab.exe) do echo    %%~zI bytes
    echo ================================================================
) else (
    echo ERROR: Build failed. Check output above.
    exit /b 1
)

deactivate
