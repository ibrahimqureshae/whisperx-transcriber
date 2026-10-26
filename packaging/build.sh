#!/usr/bin/env bash
#
# WhisperX Transcriber — Thin Launcher build (macOS / Linux)
#
# Mirrors packaging/build.bat: builds the small GUI-only launcher with
# PyInstaller, stages the source files (app.py, setup_wizard.py, transcribe.py,
# core/) alongside it, and packages a portable zip. The ML backend is NOT
# bundled — it installs into runtime/ on first launch via the setup wizard.
#
# Usage (from anywhere):
#   bash packaging/build.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
DIST="$ROOT/dist/WhisperXTranscriber"

echo
echo "  +----------------------------------------------------+"
echo "  |   WhisperX Transcriber - Thin Launcher Build (mac) |"
echo "  +----------------------------------------------------+"
echo

# -- Build tools --------------------------------------------------------------
echo "  [1/4] Checking build tools..."
"$PY" -c "import PyInstaller" 2>/dev/null || { echo "        Installing PyInstaller..."; "$PY" -m pip install pyinstaller --quiet; }
"$PY" -c "import PIL"         2>/dev/null || { echo "        Installing Pillow...";      "$PY" -m pip install Pillow --quiet; }
"$PY" -c "import customtkinter" 2>/dev/null || { echo "        Installing customtkinter..."; "$PY" -m pip install customtkinter --quiet; }
echo "        Done."

# -- Build thin launcher ------------------------------------------------------
echo "  [2/4] Building thin launcher (no torch / no whisperx)..."
"$PY" -m PyInstaller "$ROOT/packaging/launcher-mac.spec" --noconfirm --distpath "$ROOT/dist"
echo "        Done."

# -- Stage source files into dist ---------------------------------------------
echo "  [3/4] Copying source files into dist..."
cp -f "$ROOT/app.py"          "$DIST/app.py"
cp -f "$ROOT/setup_wizard.py" "$DIST/setup_wizard.py"
cp -f "$ROOT/transcribe.py"   "$DIST/transcribe.py"
cp -f "$ROOT/version.txt"     "$DIST/version.txt"

mkdir -p "$DIST/core"
cp -f "$ROOT/core/__init__.py" "$DIST/core/__init__.py"
cp -f "$ROOT/core/pipeline.py" "$DIST/core/pipeline.py"
cp -f "$ROOT/core/runner.py"   "$DIST/core/runner.py"

mkdir -p "$DIST/runtime" "$DIST/Models"

if [ -f "$ROOT/assets/icon.ico" ]; then
  mkdir -p "$DIST/assets"
  cp -f "$ROOT/assets/icon.ico" "$DIST/assets/icon.ico"
fi
echo "        Done."

# -- Create portable zip ------------------------------------------------------
echo "  [4/4] Creating portable zip..."
mkdir -p "$ROOT/dist/release"
ZIP="$ROOT/dist/release/WhisperXTranscriber-macos.zip"
rm -f "$ZIP"
if command -v ditto >/dev/null 2>&1; then
  ( cd "$ROOT/dist" && ditto -c -k --sequesterRsrc --keepParent "WhisperXTranscriber" "$ZIP" )
else
  ( cd "$ROOT/dist" && zip -r -q "$ZIP" "WhisperXTranscriber" )
fi
SIZE_MB=$(( $(stat -f%z "$ZIP" 2>/dev/null || stat -c%s "$ZIP") / 1048576 ))
echo "        Packed -> $ZIP (${SIZE_MB} MB)"

echo
echo "  ============================================================"
echo "  Release zip ready: dist/release/WhisperXTranscriber-macos.zip"
echo "  ============================================================"
echo
