# Packaging Guide

For maintainers and contributors who want to build the Windows release.

---

## How the portable zip works

```
WhisperXTranscriber.zip  (~19 MB download)
│
│  extract anywhere, e.g. C:\Apps\WhisperXTranscriber\
│
├── WhisperXTranscriber.exe    ← thin launcher (PyInstaller, GUI only)
├── _internal/                 ← Python + customtkinter + Pillow only
├── app.py                     ← main app (source)
├── setup_wizard.py            ← first-run wizard (source)
├── transcribe.py              ← headless CLI (source)
├── version.txt
├── assets/
├── runtime/                   ← empty on first run, filled by setup wizard
└── Models/                    ← empty on first run, filled on first transcription
```

The zip ships **no AI models, no PyTorch, no whisperx**.  
Everything downloads on first launch via the setup wizard.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10–3.13 | Must be in PATH — used to create the dev `.venv` |
| UPX *(optional)* | any | [upx.github.io](https://upx.github.io/) — reduces launcher size by ~30% |

---

## Build steps

### 1 — Set up the dev environment (once)

```batch
run.bat
```

Creates `.venv` with all dependencies. Required before building.

### 2 — Build

```batch
packaging\build.bat
```

What it does:
1. Checks `.venv` exists (exits with error if not)
2. Installs `pyinstaller` and `Pillow` into `.venv` if missing
3. Runs PyInstaller with `packaging/launcher.spec` — produces a thin launcher with **no torch, no whisperx, no ML packages**
4. Copies `app.py`, `setup_wizard.py`, `transcribe.py`, and `version.txt` into `dist/WhisperXTranscriber/`
5. Copies `assets/icon.ico` into `dist/WhisperXTranscriber/assets/`
6. Creates empty `runtime/` and `Models/` placeholder directories
7. Zips `dist/WhisperXTranscriber/` → `dist/release/WhisperXTranscriber.zip`

---

## Output

```
dist/
├── WhisperXTranscriber/              ← PyInstaller onedir (intermediate)
│   ├── WhisperXTranscriber.exe
│   ├── _internal/
│   ├── app.py
│   ├── setup_wizard.py
│   ├── transcribe.py
│   ├── version.txt
│   ├── assets/
│   ├── runtime/
│   └── Models/
│
└── release/
    └── WhisperXTranscriber.zip       ← ship this (~19 MB)
```

---

## What is and isn't bundled

### Inside the zip (~19 MB)

| Included | Size |
|---|---|
| Python runtime (slim) | ~12 MB |
| customtkinter | ~3 MB |
| Pillow | ~3 MB |
| app.py, setup_wizard.py, assets | ~1 MB |

### Downloaded on first launch (not bundled)

| Package | Notes |
|---|---|
| torch + torchaudio | ~1 GB CPU / ~2–3 GB GPU — whisperx resolves the correct version |
| whisperx + faster-whisper + ctranslate2 | Core inference stack |
| customtkinter, Pillow | Also installed into the runtime venv |
| Models (large-v2) | ~3 GB — downloaded on first transcription |

---

## Release checklist

Before tagging a new release:

- [ ] Update `version.txt` (root of repo)
- [ ] Delete `dist/` and run `packaging\build.bat` clean
- [ ] Test `dist/WhisperXTranscriber/WhisperXTranscriber.exe` directly before zipping
- [ ] Test the zip on a clean machine (no prior runtime/ installed)
- [ ] Test with NVIDIA GPU (verify GPU is detected in wizard)
- [ ] Upload `dist/release/WhisperXTranscriber.zip` to the GitHub release
- [ ] Tag the release: `git tag v1.x.x && git push --tags`

---

## Version bump

One file to update per release:

```
version.txt    ← plain version string, e.g. 1.0.1
```
