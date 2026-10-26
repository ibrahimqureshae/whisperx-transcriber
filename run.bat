@echo off
setlocal enabledelayedexpansion
title WhisperX Transcriber

rem  WhisperX Transcriber - Windows launcher
rem  First run : creates .venv and installs all dependencies (5-20 min)
rem  Later runs: opens the app instantly

set "PY=%~dp0.venv\Scripts\python.exe"
set "PIP=%~dp0.venv\Scripts\pip.exe"
set "APP=%~dp0app.py"

echo.
echo   +------------------------------------------+
echo   ^|       WhisperX  Transcriber             ^|
echo   ^|       AI Transcription by Muqaddimah    ^|
echo   +------------------------------------------+
echo.

rem -- Check app.py is present -------------------------------------------------
if not exist "%APP%" (
    echo   ERROR: app.py not found.
    echo   Make sure run.bat is in the same folder as app.py.
    echo.
    pause
    exit /b 1
)

rem -- Check Python is installed -----------------------------------------------
echo   Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Python not found.
    echo   Install Python 3.10+ from: https://www.python.org/downloads/
    echo   IMPORTANT: tick "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>&1') do set PY_VER=%%v
echo   Found Python !PY_VER!
echo.

rem -- Skip setup if venv already exists ---------------------------------------
if exist "%PY%" (
    echo   Environment ready.
    echo.
    goto :launch
)

rem -- First-time setup --------------------------------------------------------
echo   First-time setup - this takes a few minutes.
echo   Internet connection required to download packages.
echo.

echo   [1/4] Creating virtual environment...
python -m venv "%~dp0.venv"
if errorlevel 1 (
    echo   ERROR: Could not create virtual environment.
    pause
    exit /b 1
)
echo         Done.

echo   [2/4] Upgrading pip...
"%PY%" -m pip install --upgrade pip --quiet
echo         Done.

echo   [3/4] Installing PyTorch...
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo         NVIDIA GPU detected - installing CUDA 12.1 build
    echo         Downloading ~2.5 GB, please be patient...
    "%PIP%" install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
) else (
    echo         No NVIDIA GPU - installing CPU build...
    "%PIP%" install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
)
if errorlevel 1 (
    echo   ERROR: PyTorch install failed. Check your internet connection.
    pause
    exit /b 1
)
echo         Done.

echo   [4/4] Installing WhisperX and UI dependencies...
echo         Downloading ~500 MB, almost there...
"%PIP%" install whisperx customtkinter Pillow --quiet
if errorlevel 1 (
    echo   ERROR: Package install failed.
    pause
    exit /b 1
)
echo         Done.

echo.
echo   Setup complete! The app will open now.
echo   Next time just double-click run.bat - it will open instantly.
echo.

rem -- Launch ------------------------------------------------------------------
:launch
echo   Launching...
echo.
"%PY%" "%APP%"

if errorlevel 1 (
    echo.
    echo   The app exited with an error.
    echo   Open an issue at: https://github.com/muqaddimah/whisperx-transcriber/issues
    echo.
    pause
)
