# Security

## Reporting a vulnerability

If you discover a security issue, please open a
[GitHub issue](https://github.com/ibrahimqureshae/whisperx-transcriber/issues)
or contact the maintainer directly via GitHub.

## How this app handles your data

WhisperX Transcriber is designed to be **completely private by default**:

- **No network requests during normal operation.** The app makes zero outbound
  connections after the one-time setup is complete.
- **No telemetry.** No usage data, crash reports, or analytics are ever collected or sent.
- **No hidden processes.** The full source code is public and auditable — you can
  verify exactly what the app does.
- **Local models only.** AI models are downloaded once to your local disk and
  run entirely on your own hardware.

The only time the app accesses the internet is during the **first-run Setup Wizard**,
which downloads the AI engine (PyTorch, WhisperX) from their official sources (PyPI, HuggingFace).
After setup is complete, the app never needs the internet again.
