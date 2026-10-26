# Changelog

All notable changes to WhisperX Transcriber are documented here.

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
