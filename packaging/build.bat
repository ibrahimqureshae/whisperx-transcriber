@echo off
setlocal enabledelayedexpansion
title WhisperX Build

echo.
echo   +----------------------------------------------------+
echo   ^|    WhisperX Transcriber - Thin Launcher Build    ^|
echo   +----------------------------------------------------+
echo.

set "ROOT=%~dp0.."
set "VENV=%ROOT%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"
set "PYINST=%VENV%\Scripts\pyinstaller.exe"
set "DIST=%ROOT%\dist\WhisperXTranscriber"

rem -- Check venv ---------------------------------------------------------------
if not exist "%PY%" (
    echo   ERROR: .venv not found at %VENV%
    echo   Run run.bat once to create the environment, then re-run this script.
    echo.
    exit /b 1
    exit /b 1
)

rem -- Install build tools ------------------------------------------------------
echo   [1/4] Checking build tools...
"%PY%" -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo         Installing PyInstaller...
    "%PIP%" install pyinstaller --quiet
)
"%PY%" -c "import PIL" >nul 2>&1
if errorlevel 1 (
    echo         Installing Pillow...
    "%PIP%" install Pillow --quiet
)
echo         Done.

rem -- Build thin launcher ------------------------------------------------------
echo   [2/4] Building thin launcher (no torch / no whisperx)...
echo         This should be fast - only GUI deps are bundled.
echo.
"%PYINST%" "%ROOT%\packaging\launcher.spec" --noconfirm --distpath "%ROOT%\dist"
if errorlevel 1 (
    echo.
    echo   ERROR: PyInstaller build failed.
    exit /b 1
    exit /b 1
)
echo.
echo         Done.

rem -- Stage source files into dist ---------------------------------------------
echo   [3/4] Copying source files into dist...
copy /Y "%ROOT%\app.py"           "%DIST%\app.py"           >nul
copy /Y "%ROOT%\setup_wizard.py"  "%DIST%\setup_wizard.py"  >nul
copy /Y "%ROOT%\transcribe.py"    "%DIST%\transcribe.py"    >nul
copy /Y "%ROOT%\version.txt"      "%DIST%\version.txt"      >nul

rem Copy the pipeline + runner modules
if not exist "%DIST%\core" mkdir "%DIST%\core"
copy /Y "%ROOT%\core\__init__.py"  "%DIST%\core\__init__.py"  >nul
copy /Y "%ROOT%\core\pipeline.py"  "%DIST%\core\pipeline.py"  >nul
copy /Y "%ROOT%\core\runner.py"    "%DIST%\core\runner.py"    >nul

rem Create empty runtime/ and Models/ placeholder folders
if not exist "%DIST%\runtime"  mkdir "%DIST%\runtime"
if not exist "%DIST%\Models"   mkdir "%DIST%\Models"

rem Copy icon into assets/ subdir
if exist "%ROOT%\assets\icon.ico" (
    if not exist "%DIST%\assets" mkdir "%DIST%\assets"
    copy /Y "%ROOT%\assets\icon.ico" "%DIST%\assets\icon.ico" >nul
)
echo         Done.

rem -- Create portable zip ------------------------------------------------------
echo   [4/4] Creating portable zip...
if not exist "%ROOT%\dist\release" mkdir "%ROOT%\dist\release"
set "ZIP=%ROOT%\dist\release\WhisperXTranscriber.zip"
if exist "%ZIP%" del /f "%ZIP%"
"%PY%" -c "import zipfile,pathlib; src=pathlib.Path(r'%DIST%'); out=pathlib.Path(r'%ZIP%'); zf=zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6); [zf.write(f,f.relative_to(src)) for f in src.rglob('*') if f.is_file()]; zf.close(); print(f'  Packed {sum(1 for _ in src.rglob(chr(42)) if _.is_file())} files -> {out.stat().st_size//1048576} MB')"
if errorlevel 1 (
    echo   ERROR: Zip creation failed.
    exit /b 1
    exit /b 1
)

echo.
echo   ============================================================
echo   Release zip ready: dist\release\WhisperXTranscriber.zip
echo   ============================================================
echo.
