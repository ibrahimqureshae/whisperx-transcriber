# Changelog

All notable changes to WhisperX Transcriber are documented here.

## [v2.0.0] — 2026-10-26

Includes the v1.5 hardening below, which was never released on its own.

### Added
- A ♥ Support menu at the foot of the sidebar: star the project on GitHub, sponsor it, buy me a coffee or donate with PayPal. Each opens in your browser; the app itself sends nothing.
- A redesigned first-time setup window in the app's theme: a step tracker, a download card with size and speed, a clear done page, and an error page that shows the reason and the details log.
- `python-bidi` and `arabic-reshaper`, for right-to-left display. The setup wizard installs them, and the launcher adds them once to existing installs.

### Fixed
- A failed first-time setup shows its error page with the reason; it stayed on "Installing…" before.
- First-time setup installs the GPU engine with NVIDIA driver 610 and newer, which print "CUDA UMD Version"; it installed the CPU engine before.
- A file that cannot be read fails with the reason ("Could not read this file: …") instead of the ffmpeg version line.
- The session waveform's start and end times stay readable on top of the wave, and the end time no longer shows twice when the playhead reaches it.
- The Transcript timeline fits short files instead of always spanning 2½ minutes; its ruler ticks follow the length.
- Every Queue and Transcribe row shows its own file's waveform, also after a restart (kept in `queue.json`).
- Queue row spacing grows with the display scale.
- The scaled window opens at its full 1366x768 size from the first frame.
- Speaker names, lanes, the ruler and click areas on the Transcript timeline scale with the display; names no longer clip to `SPEAKER_...`.
- Right-to-left transcripts (Urdu, Arabic, Persian, Pashto) read in order in the transcript, on the timeline, in Doubtful words and in Find; English words and numbers inside them keep their order. Stored and exported text is unchanged.
- File rows and the session title measure names in the scale they are drawn at; the waveform timecodes use the scaled font.
- Speed figures (Queue tooltip, Transcript Re-run panel) leave out model-load time.
- Starting a run from another screen shows the Transcribe screen.
- Failed rows show the full error on hover.
- Crisp drawing at every scale: anti-aliased rounded boxes, whole-pixel lines and playhead, full-resolution window icon; Queue and Models headers line up with their rows.

### Changed
- An uncertain language guess names the language (for example "Tibetan 17 %") and turns the file row's Done pill orange with the warning as its tooltip.
- The session pill names the live stage (Loading model, Detecting language, Aligning words …).
- The default batch also starts smaller when free GPU memory is low.

## [v1.5] — hardening (released in v2.0.0)

### Changed
- Speaker identification no longer needs a Hugging Face token, account or terms: it runs on sherpa-onnx and downloads two open models (about 34 MB) on first use. The `--hf-token` flag and the token setting are gone.

### Fixed
- The real app could not transcribe anything (`core/queue.py` shadowed the stdlib); the runner now starts as `python -m core.runner`.
- Word timing for non-English languages crashed the worker; the speech model is freed before alignment, and an alignment failure is now a visible warning.
- Translate with turbo, distil and `.en` models silently returned untranslated text; Translate is now refused and greyed out for them.
- An 8 GB GPU ran out of memory at the default batch; the batch starts at 8 on small GPUs and backs off on out-of-memory.
- Cancel and closing the app no longer leave the worker running on the GPU; only one copy of the app can run at a time.
- Corrupt settings or queue files are kept as `.corrupt-<time>` copies instead of being overwritten; save errors show in the status bar; settings are saved atomically.
- Invalid numbers in Settings are refused with the reason in a tooltip; the token field always stays masked.
- Screens no longer paint over each other; Settings, Models, Queue and Transcript open quickly, also with hundreds of finished jobs.

### Changed
- One fixed window size (1366×768); no page scrolling.
- Transcribe: per-file stage dots replace the Stages panel; a grouped setup panel (Recognition dropdowns with their names, Output check chips, Options switches); one speaker-count dropdown ("Auto", "2 speakers" …).
- Settings use the same dropdowns, check chips and switch rows.
- Transcripts are kept across restarts; finished files can be reopened.
- Opening the output folder once per batch is on by default; the window title shows the version.

## [v1.4.0] — 2026-10-22

### Added
- **Five screens** — Transcribe, Queue, Transcript, Models and Settings, with an Activity log in the sidebar. New dark "session" look.
- **Batch queue** — queue several files and run them back to back. Drag and drop files onto the window (needs tkinterdnd2; Browse works without it).
- **Real progress** — transcription progress, per-stage timings and a waveform of the audio.
- **Language detection confidence** — the detected language is shown with its probability.
- **Speaker identification** — pyannote speaker diarization. It needs a Hugging Face token and acceptance of the model terms. Speaker names are applied to every export, and speaker embeddings can be saved as `<name>_speakers.json`.
- **Transcript screen** — open a finished transcript in the app: speaker lanes timeline, segment list with doubtful words underlined, Find, speaker rename, re-export, and "Add to hotwords and re-run".
- **Transcription options** — hotwords, suppress numerals, full decoding options, VAD method (Silero or Pyannote), GPU device index, CPU threads and offline-only loading.
- **Alignment options** — alignment model override, interpolation method and character timings.
- **New models** — `distil-large-v3.5`, `distil-medium.en` and `distil-small.en`.
- **Audacity labels export** — writes an `.aud` label file.
- **Models screen** — download and delete models, and see disk use.
- **CLI flags** — `--hotwords`, `--suppress-numerals`, `--vad-method`, `--device-index`, `--threads`, `--offline`, `--temperature`, `--beam-size`, `--no-speech-threshold`, `--logprob-threshold`, `--compression-ratio-threshold`, `--condition-on-previous-text`, `--align-model`, `--interpolate-method`, `--char-alignments`, `--segment-resolution`, `--diarize`, `--min-speakers`, `--max-speakers`, `--hf-token` and `--speaker-embeddings`.

### Changed
- Advanced settings moved from the ⚙ tab to the Settings screen.
- Settings are saved in `settings.json`, including every new option. Older settings files load with defaults for the new keys.
- The Hugging Face token is stored only in `settings.json`. It is never written to `queue.json` or to logs.
- `segment_resolution` is accepted, but the whisperx 3.8.6 writers ignore it.

## [v1.3.0] — 2026-10-22

### Added
- **Open folder when done** — new option that opens the output folder when a run finishes (GitHub issue #1).
- **Remembers your settings** — your last choices come back the next time you open the app.
- **≡ button** — toggles the Activity log at any time, not only during a run.
- **Run feedback** — elapsed time, an **Open folder** button when a run finishes, and an error dialog when a run fails.
- **Alignment warning** — the Transcribe screen warns when word timestamps are not available for the selected language.
- **Translate task** — choose "Translate to English" in Settings to translate speech while transcribing.
- **Initial prompt** — bias the transcription with names, jargon, or spelling hints.
- **New models** — `large-v3-turbo` and `distil-large-v3`.
- **Keyboard shortcuts** — Ctrl+O to open a file, Ctrl+Enter to transcribe, Esc to stop.
- **Resizable window**.
- **Translate runs skip word alignment** (alignment models are per source language and cannot time translated text).
- **CLI flags** — `--task`, `--output-dir` and `--initial-prompt`. `--output` can be repeated.

### Fixed
- Word timestamps now run whenever the option is on. Previously they ran only when Word JSON or Karaoke output was selected, so SRT and VTT files had no word timing.
- Silence detection and chunk settings now take effect. They were collected by the UI but never passed to the pipeline.
- GPU install: the setup wizard picks the CUDA 12.6 or 12.8 PyTorch build from the driver version and pins torch 2.8. The GPU is checked after setup. Drivers older than 560 get the CPU build with a clear message instead of a silent CPU install. `run.bat`, `run.sh` and `requirements-gpu.txt` were updated to match.
- `run.bat` linked to the wrong GitHub issues page.
- The "Remove silence" toggle was removed: whisperx's voice-activity detection is always on, so the switch never did anything.
- README and CONTRIBUTING described the pre-1.2 tabbed interface; they now match the single-screen app and the real driver requirement.
- The test suite pointed at a missing fixture and crashed without whisperx installed; it now uses `English_test.wav` and skips cleanly.

## [v1.2.0] — 2026-06-09

### Added
- **Instant Cancel** — transcription now runs in a separate process that is
  terminated immediately when you click Cancel, instead of waiting for the
  current stage to finish.

### Changed
- **Redesigned interface** — consolidated into a single, fixed-size Transcribe
  screen: file, language, model, device, output formats, save location, and
  options are all in one place. Advanced settings moved behind a dedicated tab.
- Cleaner dropdowns, improved text contrast and readability throughout.
- Activity log now appears only while a job is running and auto-hides when done.

## [v1.1.4] — 2026-06-09

### Changed
- Removed progress bar and cancel button mentions from docs (not yet implemented)

## [v1.1.3] — 2026-06-07

### Added
- Bundled FFmpeg via imageio-ffmpeg — no separate download needed
- Real-time download progress during setup wizard

### Fixed
- FFmpeg resolution now works reliably across all configurations

## [v1.1.0] — 2026-06-07

### Added
- Full GUI desktop app for Windows (customtkinter)
- One-time Setup Wizard with automatic GPU/CPU detection
- Support for 20 languages in the GUI dropdown, 99 via CLI
- Word-level timestamp alignment via WhisperX
- Export to SRT, VTT, TXT, TSV, JSON, and Word JSON formats
- CPU and GPU (CUDA) transcription support
- Fully offline operation after one-time setup
- VAD (Voice Activity Detection) silence stripping
- Karaoke-style word-highlighted subtitle export
- Model support: tiny, base, small, medium, large-v2, large-v3
- Headless CLI (transcribe.py) for scripting and batch use
- Portable distribution — no installer, no admin rights required
- Thin launcher pattern — ~19 MB exe, AI engine installed separately
- Auto-detection of NVIDIA GPU with CPU fallback
