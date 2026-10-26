# Contributing to WhisperX Transcriber

Contributions are genuinely welcome — whether you're fixing a bug,
improving the UI, adding a language, or just reporting an issue.

## Ways to contribute

- **Bug reports** — open an [issue](https://github.com/ibrahimqureshae/whisperx-transcriber/issues) with log output and steps to reproduce
- **Feature requests** — open an issue describing the use case and why it matters
- **Code contributions** — fork the repo, make your changes, open a pull request
- **Language support** — add entries to the `_lang_map` dict in `app.py` (one line per language)
- **Testing** — test on different hardware, Windows versions, or audio types and share findings
- **Documentation** — fix typos, improve clarity, add examples

## Getting started

```bash
git clone https://github.com/ibrahimqureshae/whisperx-transcriber.git
cd whisperx-transcriber
run.bat   # creates .venv, installs everything, launches the app
```

## Project structure

The codebase is intentionally small and readable:

- `app.py` — the entire GUI (~1000 lines)
- `setup_wizard.py` — the first-run installer
- `transcribe.py` — the headless CLI
- `core/pipeline.py` — all AI/WhisperX logic, zero GUI dependencies

No framework magic, no hidden abstractions. A new contributor can understand
the whole project in an afternoon.

## Coding style

- Match the style of the surrounding code
- Keep functions focused — one responsibility per function
- Prefer clarity over cleverness
- No new dependencies without a clear reason

## How to test changes

```bash
# Run the app from source
run.bat

# Run the test suite
python -m pytest tests/

# Test the CLI directly
python transcribe.py tests/fixtures/English_test.wav --model tiny
```

## Pull request guidelines

- Keep PRs focused — one feature or fix per PR
- Test on both CPU and GPU if possible
- Don't change requirements files without a clear reason
- Update CHANGELOG.md with a brief description of your change

## PR checklist

- [ ] Tested locally (CPU mode at minimum)
- [ ] No Python files changed unnecessarily
- [ ] CHANGELOG.md updated
- [ ] No new dependencies added without discussion

## Reporting bugs

Please include:
1. Steps to reproduce
2. Expected vs actual behaviour
3. Log output (click **Activity** in the sidebar and paste it)
4. Your system: Windows version, GPU model (if any), model used, file format
