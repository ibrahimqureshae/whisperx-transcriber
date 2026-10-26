#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  WhisperX Transcriber — macOS / Linux launcher
#
#  First run  : creates .venv and installs all dependencies (5–20 min)
#  Later runs : opens the app in ~2 seconds
#
#  Requirements: Python 3.10+   brew install python  |  sudo apt install python3
#               ffmpeg           brew install ffmpeg  |  sudo apt install ffmpeg
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/.venv"
APP="$DIR/app.py"

echo ""
echo "  +------------------------------------------+"
echo "  |         WhisperX  Transcriber            |"
echo "  |         AI Transcription  by Muqaddimah  |"
echo "  +------------------------------------------+"
echo ""

# ── Verify app.py exists ──────────────────────────────────────────────────────
if [ ! -f "$APP" ]; then
    echo "  ERROR: app.py not found in $DIR"
    echo "  Make sure run.sh is in the same folder as app.py."
    echo ""
    exit 1
fi

# ── Find Python 3.10+ ─────────────────────────────────────────────────────────
echo "  Checking Python..."
PYTHON=""
for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver="$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")"
        major="${ver%%.*}"
        minor="${ver#*.}"
        if [ "${major:-0}" -ge 3 ] && [ "${minor:-0}" -ge 10 ] 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "  Python 3.10+ not found."
    echo ""
    echo "  macOS:  brew install python"
    echo "  Ubuntu: sudo apt install python3.11"
    echo "  Other:  https://www.python.org/downloads/"
    echo ""
    exit 1
fi
echo "  Found $($PYTHON --version)"

# ── Skip setup if venv already ready ─────────────────────────────────────────
if [ -f "$VENV/bin/python" ]; then
    echo "  Environment ready."
    echo ""
    "$VENV/bin/python" "$APP"
    exit $?
fi

# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "  First-time setup — please wait, this takes a few minutes."
echo "  Your internet connection is needed to download packages."
echo ""
# ═════════════════════════════════════════════════════════════════════════════

# ── Create virtual environment ────────────────────────────────────────────────
echo "  [1/4] Creating virtual environment..."
"$PYTHON" -m venv "$VENV"
echo "         Done."

# ── Upgrade pip ───────────────────────────────────────────────────────────────
echo "  [2/4] Upgrading pip..."
"$VENV/bin/pip" install --upgrade pip --quiet
echo "         Done."

# ── Detect hardware and install PyTorch ───────────────────────────────────────
echo "  [3/4] Installing PyTorch..."
UNAME="$(uname -s)"
ARCH="$(uname -m)"
TORCH_FLAGS=""

if [ "$UNAME" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    echo "         Apple Silicon (M-series) — MPS acceleration available"
    # Standard PyTorch build has MPS support on Apple Silicon
elif [ "$UNAME" = "Linux" ] && command -v nvidia-smi &>/dev/null; then
    echo "         NVIDIA GPU detected — installing CUDA 12.1 build"
    echo "         (This download is ~2.5 GB, please be patient)"
    TORCH_FLAGS="--index-url https://download.pytorch.org/whl/cu121"
else
    echo "         CPU-only build (no NVIDIA GPU detected)"
fi

# shellcheck disable=SC2086
"$VENV/bin/pip" install torch torchaudio $TORCH_FLAGS --quiet
echo "         Done."

# ── Install remaining packages ────────────────────────────────────────────────
echo "  [4/4] Installing WhisperX + UI dependencies..."
echo "         (Another ~500 MB — almost there)"
"$VENV/bin/pip" install whisperx customtkinter reportlab --quiet
echo "         Done."

# ── ffmpeg check ─────────────────────────────────────────────────────────────
echo ""
if ! command -v ffmpeg &>/dev/null; then
    echo "  NOTE: ffmpeg not found. WhisperX needs it to read video files."
    echo "        Audio files (.wav/.mp3) will work without it."
    if [ "$UNAME" = "Darwin" ]; then
        echo "        Install: brew install ffmpeg"
    else
        echo "        Install: sudo apt install ffmpeg"
    fi
fi

echo ""
echo "  Setup complete!  The app will open now and on every future run."
echo ""

# ── Launch ───────────────────────────────────────────────────────────────────
echo "  Launching WhisperX Transcriber..."
echo ""
"$VENV/bin/python" "$APP"
