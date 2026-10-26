"""
WhisperX transcription runner — runs the pipeline in a separate process so the
GUI can terminate it instantly (true Cancel) by killing this process.

Protocol:
  • Reads one line of JSON config from stdin.
  • Emits newline-delimited JSON events to stdout, flushed immediately:
        {"t": "log",      "m": "..."}        log line for the Activity panel
        {"t": "status",   "m": "..."}        status-bar text
        {"t": "pmode",    "mode": "determinate" | "indeterminate"}
        {"t": "progress", "v": 0.0..1.0}     progress-bar fraction
        {"t": "done",     "path": "..."|null} finished successfully
        {"t": "error",    "m": "..."}         failed (also exits non-zero)

This module has NO GUI dependencies. It exits 0 on success, 1 on error.
"""
import os
import sys
import json
import time
import traceback

# ── Isolate a clean JSON channel ────────────────────────────────────────────
# The GUI reads our stdout line-by-line expecting one JSON event per line.
# Heavy libraries (huggingface tqdm bars, torch/ctranslate2, pyannote/Lightning
# logging, warnings) write loads of unstructured text — and tqdm uses '\r' with
# no newline, which would otherwise be concatenated onto our JSON lines and
# corrupt them.  So: clone the real stdout into a private fd used ONLY for JSON,
# then point fd 1 (and Python's sys.stdout) at stderr, which the parent
# discards.  Every byte of library noise is dropped; only our events survive.
try:
    _JSON = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)               # fd 1 -> fd 2 (parent discards stderr)
    sys.stdout = sys.stderr     # Python-level prints -> stderr too
except Exception:
    _JSON = sys.stdout          # best-effort fallback

# Make the `core` package importable regardless of how we were launched.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import pipeline


def _emit(obj):
    """Write one JSON event line to the private JSON channel and flush."""
    try:
        _JSON.write(json.dumps(obj, ensure_ascii=True) + "\n")
        _JSON.flush()
    except Exception:
        pass


def log(msg):          _emit({"t": "log", "m": str(msg)})
def status(msg):       _emit({"t": "status", "m": str(msg)})
def pmode(mode):       _emit({"t": "pmode", "mode": mode})
def progress(v):       _emit({"t": "progress", "v": float(v)})


def _default_model_dir():
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), "Models")
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library",
                            "Application Support", "WhisperX", "Models")
    return os.path.join(_ROOT, "Models")


class _DLProgress:
    """Throttled model-download progress, mirroring the old GUI behaviour."""
    def __init__(self):
        self.last = 0.0
        self.finalizing = False
        self.determinate = False

    def __call__(self, done, total):
        frac = (done / total) if total else 0
        # After every byte arrives, HuggingFace finalizes files on disk with no
        # further size growth — switch to an indeterminate bar so it doesn't
        # look frozen at 100%.
        if frac >= 1.0 and total > 20 * 1048576:
            if not self.finalizing:
                self.finalizing = True
                pmode("indeterminate")
                status("Finalizing model files...")
            return
        now = time.time()
        if now - self.last < 0.2:
            return
        self.last = now
        if not self.determinate:
            self.determinate = True
            pmode("determinate")
        mb_d, mb_t = done / 1048576, total / 1048576
        status(f"Downloading model... {mb_d:.0f} / {mb_t:.0f} MB  ({frac * 100:.0f}%)")
        progress(min(1.0, frac))


def run(cfg):
    if not cfg.get("model_dir"):
        cfg["model_dir"] = _default_model_dir()
    os.environ["HF_HOME"] = cfg["model_dir"]

    # ── Device ─────────────────────────────────────────────────────────────────
    _, t_dev, a_dev = pipeline.detect_device(cfg["device"])
    compute, adjusted = pipeline.resolve_compute(t_dev, cfg["compute"])
    if adjusted:
        log("  float16 not supported on CPU — using int8")
    log(f"Device: {t_dev.upper()}   Compute: {compute}")
    if a_dev != t_dev:
        log(f"Alignment: {a_dev.upper()}")

    lang = None if cfg["language"] == "auto" else cfg["language"]

    # ── Model ──────────────────────────────────────────────────────────────────
    model_cache = os.path.join(
        cfg["model_dir"], "hub",
        f"models--Systran--faster-whisper-{cfg['model']}")
    if not os.path.isdir(model_cache):
        log(f"Downloading model '{cfg['model']}' for the first time...")
        log("  The model will be saved and reused on all future runs.")
        status("Downloading model...")
        pmode("indeterminate")
        try:
            pipeline.download_model(
                cfg["model"], cfg["model_dir"],
                progress_cb=_DLProgress())
            log("  Model downloaded.")
        except Exception as e:
            log(f"  Could not track progress ({e}) — downloading...")
        pmode("indeterminate")
        status("Loading model...")
    else:
        log(f"Loading model '{cfg['model']}' from cache...")
        status("Loading model...")

    model = pipeline.load_model(
        cfg["model"], t_dev, compute, lang, cfg["beam"])

    # ── Audio ──────────────────────────────────────────────────────────────────
    log("Loading audio...")
    status("Loading audio...")
    audio = pipeline.load_audio(cfg["audio"])

    # ── Transcribe ─────────────────────────────────────────────────────────────
    log("Transcribing...")
    status("Transcribing...")
    result, det = pipeline.transcribe(model, audio, cfg["batch"], lang)
    log(f"  Language: {det}   Segments: {len(result['segments'])}")

    # ── Align ──────────────────────────────────────────────────────────────────
    needs_align = cfg["align"] and (
        "word_json" in cfg["formats"] or cfg["highlight"])
    if needs_align:
        log("Aligning word timestamps...")
        status("Aligning...")
        try:
            result = pipeline.align(result, det, a_dev, audio)
        except Exception as e:
            log(f"  Alignment failed: {e} — continuing without word timestamps")
            result.setdefault("language", det)
    else:
        result.setdefault("language", det)

    # ── Export ─────────────────────────────────────────────────────────────────
    log("Saving files...")
    status("Saving...")
    last_path = None
    for fmt, path, err in pipeline.export(
            result, cfg["formats"], cfg["out_dir"], cfg["audio"],
            {"max_width": cfg["max_width"],
             "max_count": cfg["max_count"],
             "highlight": cfg["highlight"]}):
        if err:
            log(f"  {fmt} error: {err}")
        else:
            log(f"  {fmt:<10}->  {path}")
            last_path = path

    log("Done!")
    _emit({"t": "done", "path": last_path})


def main():
    raw = sys.stdin.readline()
    if not raw.strip():
        _emit({"t": "error", "m": "No configuration received."})
        sys.exit(1)
    cfg = json.loads(raw)
    try:
        run(cfg)
    except Exception as e:
        _emit({"t": "error", "m": f"{e}\n{traceback.format_exc()}"})
        sys.exit(1)


if __name__ == "__main__":
    main()
