# -*- mode: python ; coding: utf-8 -*-
#
# WhisperX Transcriber — Thin Launcher spec
#
# Produces a small onedir build that contains ONLY the GUI framework.
# torch, whisperx, ctranslate2, and all ML packages are intentionally
# excluded. They are pip-installed into runtime/ on first app launch.
#
# Target installed size:  ~30 MB
# Target compressed size: ~18–22 MB (with UPX)
#
# Usage (from project root):
#   pyinstaller packaging/launcher.spec --noconfirm

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

# SPECPATH is the dir containing this spec file (packaging/)
ROOT = Path(SPECPATH).parent

datas = []
datas += collect_data_files("customtkinter")   # themes, images, JSON
try:
    datas += collect_data_files("PIL")          # libjpeg, libpng, etc.
except Exception:
    pass

icon_ico = ROOT / "assets" / "icon.ico"

# ── Explicit heavy excludes ────────────────────────────────────────────────────
# Everything listed here is downloaded on first run instead of bundled.
# Keeping this list explicit (rather than just relying on PyInstaller's scan)
# prevents transitive imports from accidentally pulling large packages in.
excludes = [
    # ML / AI backend — installed at runtime
    "torch", "torch.cuda", "torch.backends.cudnn",
    "torchaudio", "torchvision",
    "whisperx", "faster_whisper",
    "ctranslate2",
    "transformers", "transformers.models",
    "tokenizers",
    "onnxruntime",
    "huggingface_hub",
    # Audio / video processing — not needed in setup wizard
    "av", "soundfile", "sounddevice", "librosa",
    # Scientific computing — not needed in setup wizard
    "numpy", "scipy", "pandas",
    "sklearn", "skimage",
    # PyAnnote / diarization — optional, not in base install
    "pyannote", "pyannote.audio", "pyannote.core", "pyannote.pipeline",
    "speechbrain",
    # Training frameworks
    "lightning", "lightning_utilities", "pytorch_lightning",
    # Plotting
    "matplotlib", "mpl_toolkits",
    # Jupyter / notebook stack
    "IPython", "ipykernel", "notebook", "nbformat", "nbconvert", "traitlets",
    # Cloud clients
    "boto3", "botocore", "google.cloud", "azure",
    # Dev tooling
    "pytest", "hypothesis", "setuptools", "pkg_resources",
    "docutils", "sphinx",
    # Triton (CUDA JIT compiler)
    "triton",
]

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        # GUI deps used by launcher.py and setup_wizard.py
        "customtkinter",
        "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont", "PIL.ImageTk",
        "tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox",
        # setup_wizard is imported dynamically inside main()
        "setup_wizard",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WhisperXTranscriber",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(icon_ico) if icon_ico.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="WhisperXTranscriber",
)
