# -*- mode: python ; coding: utf-8 -*-
#
# WhisperX Transcriber — Thin Launcher spec (macOS)
#
# Mirrors launcher.spec but targets macOS: produces a onedir portable folder
# (the same "runtime installed alongside" model as the Windows build) rather
# than a .app bundle, so the first-run wizard can write runtime/ next to the
# executable exactly like on Windows.
#
# Usage (from project root, on macOS):
#   pyinstaller packaging/launcher-mac.spec --noconfirm

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).parent

datas = []
datas += collect_data_files("customtkinter")
try:
    datas += collect_data_files("PIL")
except Exception:
    pass

# Same heavy excludes as the Windows spec — the ML backend installs at runtime.
excludes = [
    "torch", "torch.cuda", "torch.backends.cudnn",
    "torchaudio", "torchvision",
    "whisperx", "faster_whisper",
    "ctranslate2",
    "transformers", "transformers.models",
    "tokenizers",
    "onnxruntime",
    "huggingface_hub",
    "av", "soundfile", "sounddevice", "librosa",
    "numpy", "scipy", "pandas",
    "sklearn", "skimage",
    "pyannote", "pyannote.audio", "pyannote.core", "pyannote.pipeline",
    "speechbrain",
    "lightning", "lightning_utilities", "pytorch_lightning",
    "matplotlib", "mpl_toolkits",
    "IPython", "ipykernel", "notebook", "nbformat", "nbconvert", "traitlets",
    "boto3", "botocore", "google.cloud", "azure",
    "pytest", "hypothesis", "setuptools", "pkg_resources",
    "docutils", "sphinx",
    "triton",
]

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "customtkinter",
        "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont", "PIL.ImageTk",
        "tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox",
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
    upx=False,            # UPX is unreliable / unavailable on macOS runners
    console=False,
    icon=None,            # .icns generated/added later once signing is set up
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="WhisperXTranscriber",
)
