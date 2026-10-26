<div align="center">

# WhisperX Transcriber

**Offline AI transcription for Windows.**
Word-level timestamps. 99 languages. No cloud. No subscription. Free forever.

[![License](https://img.shields.io/github/license/ibrahimqureshae/whisperx-transcriber)](LICENSE)
[![Release](https://img.shields.io/github/v/release/ibrahimqureshae/whisperx-transcriber)](https://github.com/ibrahimqureshae/whisperx-transcriber/releases)
[![Platform](https://img.shields.io/badge/platform-Windows-blue)](#)
[![Offline](https://img.shields.io/badge/works-offline-brightgreen)](#)
[![No subscription](https://img.shields.io/badge/subscription-none-brightgreen)](#)

[**⬇️ Download for Windows**](https://github.com/ibrahimqureshae/whisperx-transcriber/releases/latest)
&nbsp;&nbsp;·&nbsp;&nbsp;
[Documentation](#for-developers)
&nbsp;&nbsp;·&nbsp;&nbsp;
[Report a bug](https://github.com/ibrahimqureshae/whisperx-transcriber/issues)
&nbsp;&nbsp;·&nbsp;&nbsp;
[❤️ Sponsor](https://github.com/sponsors/ibrahimqureshae)

![WhisperX Transcriber screenshot](assets/screenshots/Screenshot_new_ui.png)

</div>

---

## Why WhisperX Transcriber?

Most transcription tools are expensive cloud services that send your audio to someone else's server — or complex Python scripts that require a terminal and technical setup. WhisperX Transcriber is neither.

It's a clean Windows desktop app powered by OpenAI's Whisper model. Download once, transcribe forever — completely offline.

---

## Key Features

| | |
|---|---|
| 🔒 **100% private** | Your audio never leaves your machine. No cloud, no API key, no account. |
| ⚡ **Word-level timestamps** | Every word is precisely timed — not just sentence segments. Essential for subtitles and content search. |
| 🌐 **99 languages** | Arabic, Urdu, Persian, Chinese, Hindi, and 94 more. Right-to-left scripts supported natively. |
| 🖥️ **No terminal required** | Extract the zip, run the exe, click through a one-time wizard. Done. |
| 🎯 **Fast and accurate** | Powered by faster-whisper + WhisperX — the fastest open-source speech stack available. |
| 📦 **Multiple export formats** | SRT, VTT, TXT, TSV, JSON, Word JSON — pick what your workflow needs. |
| 🔁 **Works offline forever** | AI engine downloads once (~1–3 GB). Every transcription after that needs no internet. |

---

## Who uses this

🎙️ **Journalists** — transcribe interviews with accurate timestamps, no upload required

📹 **Content creators** — generate subtitles for YouTube, TikTok, and Reels in minutes

🔬 **Researchers & academics** — batch-process lectures, fieldwork recordings, and seminars

📝 **Translators** — get a timestamped base transcript before translating

🎓 **Students** — transcribe recorded classes and seminars offline

🔒 **Privacy-conscious users** — nothing is ever uploaded anywhere

---

## How it works

```
1. Download & extract     →   No installer. No admin rights needed.
2. Run setup wizard       →   One-time download of AI engine (~1–3 GB)
3. Transcribe anything    →   100% offline from here on. Forever.
```

**Setup time:** 5–30 minutes (depending on connection speed, one time only)
**Transcription speed:** Up to 15× faster than realtime on a modern NVIDIA GPU

---

## Getting started

### Step 1 — Download

Go to [**Releases**](https://github.com/ibrahimqureshae/whisperx-transcriber/releases) and download `WhisperXTranscriber.zip` (~19 MB).

### Step 2 — Extract and run

Extract the zip anywhere (e.g. `C:\Apps\WhisperXTranscriber\`) and double-click `WhisperXTranscriber.exe`. No installer, no admin rights required.

### Step 3 — First launch (one-time, internet required)

The **Setup Wizard** opens automatically:

1. **Welcome** — shows what will be downloaded and your GPU status
2. **Begin Setup** — downloads the AI engine (~1–3 GB, takes 5–30 min)
3. **Done** — click Launch. Every future launch opens instantly with no internet needed

> **Python 3.10+ is required** for the setup wizard. Install from [python.org](https://www.python.org/downloads/) and tick **"Add Python to PATH"**. After setup, Python runs internally — you never need to touch it again.

### Step 4 — Transcribe

1. Click **Files** → select your audio or video file
2. Go to **Model** → choose model size and device
3. Go to **Output** → choose your export formats
4. Click **Run**

---

## What the app looks like

One clean screen — file, language, model, output formats, and options all in one place. Nothing is buried.

<p align="center">
  <img src="assets/screenshots/Screenshot_new_ui.png" width="640" alt="WhisperX Transcriber main window"/>
</p>

Advanced settings (precision, performance tuning, silence detection, subtitle formatting, and model storage) live behind the ⚙ tab, so the main view stays focused on getting your transcript.

---

## Performance

| | CPU | GPU (NVIDIA) |
|---|---|---|
| Setup download | ~1 GB | ~2–3 GB |
| Transcription speed | ~0.3–0.5× realtime | ~8–15× realtime |
| Hardware required | Any Windows PC | NVIDIA GPU + driver 525+ |

The app auto-detects your GPU. You can override in the **Quality** panel.

---

## Models

Models download on first use and are cached locally — never re-downloaded.

| Model | Size | Best for |
|---|---|---|
| `tiny` | ~75 MB | Quick preview, low accuracy |
| `base` | ~145 MB | Fast drafts |
| `small` | ~465 MB | Good balance of speed and accuracy |
| `medium` | ~1.5 GB | High accuracy |
| `large-v2` *(default)* | ~3 GB | Best quality — recommended |
| `large-v3` | ~3 GB | Latest Whisper large model |

Start with `large-v2` unless you need faster results or have limited disk space.

---

## Output formats

| Format | Description |
|---|---|
| `srt` | Standard subtitles — works in VLC, YouTube, Premiere, DaVinci Resolve |
| `vtt` | WebVTT subtitles — for web video players |
| `txt` | Plain text transcript — no timestamps |
| `tsv` | Tab-separated with start/end timestamps per segment |
| `json` | Full segment-level JSON with all metadata |
| `word_json` | Per-word `{word, start, end, score}` — for developers and downstream tools |

Multiple formats can be selected at once. All saved to the same output folder.

---

## Supported languages

The GUI exposes **20 languages** with a dropdown. The underlying model supports **99 languages** via the CLI.

| Language | Code | Language | Code |
|---|---|---|---|
| English | `en` | Russian | `ru` |
| Arabic | `ar` | Portuguese | `pt` |
| French | `fr` | Italian | `it` |
| German | `de` | Dutch | `nl` |
| Spanish | `es` | Polish | `pl` |
| Chinese | `zh` | Turkish | `tr` |
| Japanese | `ja` | Persian | `fa` |
| Korean | `ko` | Urdu | `ur` |
| Hindi | `hi` | Georgian | `ka` |
| Pashto | `ps` | | |

**Auto-detect** mode lets Whisper identify the language automatically.

---

## Troubleshooting

**"Python not found"** — Install [Python 3.10+](https://www.python.org/downloads/) and tick "Add Python to PATH". Re-run the app.

**Setup fails mid-download** — Click **Retry**. The wizard resumes and skips already-installed packages.

**"An Application Control policy has blocked this file" (WinError 4551)** — Open **Windows Security → App & browser control → Smart App Control settings** and switch it to **Off**, then restart.

**"The system cannot find the file specified" (WinError 2)** — Update to the latest version from the [Releases](https://github.com/ibrahimqureshae/whisperx-transcriber/releases) page. The setup wizard now installs FFmpeg automatically.

**CUDA out of memory** — Lower Batch Size in the Quality panel. Try 8, then 4.

**Alignment fails** — Disable word alignment in the Output panel. All other formats still export without it.

**Transcription is slow on CPU** — Use a smaller model (`small` or `base`), or reduce Batch Size.

**Anything else** — Open a [GitHub issue](https://github.com/ibrahimqureshae/whisperx-transcriber/issues) and paste the log output (click **Activity** in the sidebar).

---

## Roadmap

- [ ] Drag-and-drop file input
- [ ] Batch folder transcription
- [ ] Speaker diarization UI
- [ ] In-app transcript editor
- [ ] Translation mode (transcribe + translate to English)
- [ ] macOS `.app` bundle
- [ ] Linux AppImage

---

## For Developers

> Clone, run one script, and you're in.

### Requirements

- Python 3.10, 3.11, 3.12, or 3.13
- Git
- Windows (Linux/macOS supported via `run.sh`, GUI features may vary)
- NVIDIA GPU recommended but not required

### Quick start

```bash
git clone https://github.com/ibrahimqureshae/whisperx-transcriber.git
cd whisperx-transcriber
run.bat          # Windows
bash run.sh      # macOS / Linux
```

`run.bat` / `run.sh` creates a `.venv`, detects your GPU, installs everything,
and launches the app. Every subsequent run opens instantly.

### Project structure

```
whisperx-transcriber/
├── app.py                  Main GUI application
├── launcher.py             Entry point for the packaged .exe
├── setup_wizard.py         First-run wizard
├── transcribe.py           Headless CLI for scripting and batch use
├── core/
│   └── pipeline.py         All AI/WhisperX logic — zero GUI dependencies
├── run.bat                 Windows developer launcher
├── run.sh                  macOS / Linux developer launcher
├── requirements-cpu.txt    PyTorch CPU wheels
├── requirements-gpu.txt    PyTorch CUDA wheels
├── requirements-core.txt   whisperx, customtkinter, Pillow
├── requirements-build.txt  Build tools (pyinstaller)
├── packaging/
│   ├── build.bat           Builds portable zip
│   └── launcher.spec       PyInstaller spec
└── assets/
    ├── icon.ico            App icon
    └── screenshots/        README screenshots
```

### CLI usage

```bash
python transcribe.py audio.mp3
python transcribe.py audio.mp3 --language ar --model large-v2
python transcribe.py audio.mp3 --device cpu --output srt
python transcribe.py audio.mp3 --model-dir D:\models
```

### Tech stack

| Layer | Library |
|---|---|
| GUI | [customtkinter](https://github.com/TomSchimansky/CustomTkinter) |
| Transcription | [WhisperX](https://github.com/m-bain/whisperX) |
| ASR backend | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| Inference runtime | [CTranslate2](https://github.com/OpenNMT/CTranslate2) |
| Audio / video decoding | [FFmpeg](https://ffmpeg.org/) (via imageio-ffmpeg) |
| Packaging | [PyInstaller](https://pyinstaller.org/) |

See [CONTRIBUTING.md](CONTRIBUTING.md) to get started contributing.

---

## Contributing

Contributions are welcome. Bug reports, feature requests, code, language additions,
and testing on different hardware all help.

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

---

## Support the project

WhisperX Transcriber is free and will stay free. If it saves you time or money,
consider sponsoring development:

<p align="center">
  <a href="https://github.com/sponsors/ibrahimqureshae">
    <img src="https://img.shields.io/badge/Sponsor%20on%20GitHub-%E2%99%A5-ea4aaa?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor on GitHub"/>
  </a>
  &nbsp;&nbsp;
  <a href="https://www.paypal.me/mibrahimqr">
    <img src="https://img.shields.io/badge/Donate%20via%20PayPal-0070ba?style=for-the-badge&logo=paypal&logoColor=white" alt="Donate via PayPal"/>
  </a>
</p>

Sponsorships fund continued development, new features, and platform support (macOS, Linux).

---

## License

MIT — free to use, modify, and distribute. See [LICENSE](LICENSE).

---

<div align="center">
Built on open-source foundations: <a href="https://github.com/openai/whisper">OpenAI Whisper</a> · <a href="https://github.com/m-bain/whisperX">WhisperX</a> · <a href="https://github.com/SYSTRAN/faster-whisper">faster-whisper</a> · <a href="https://github.com/TomSchimansky/CustomTkinter">customtkinter</a>
</div>
