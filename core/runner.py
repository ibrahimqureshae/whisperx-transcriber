"""
WhisperX transcription runner — runs the pipeline in a separate process so the
GUI can terminate it instantly (true Cancel) by killing this process.

Protocol:
  • Reads one line of JSON config from stdin.
  • Emits newline-delimited JSON events to stdout, flushed immediately:
        {"t": "log",      "m": "..."}        log line for the Activity panel
        {"t": "status",   "m": "..."}        status-bar text
        {"t": "pmode",    "mode": "determinate" | "indeterminate"}
        {"t": "progress", "v": 0.0..1.0}     overall fraction of the job (all stages)
        {"t": "language", "code": "..", "prob": 0.0..1.0}  language in use
        {"t": "done",     "path": "..."|null} finished successfully
        {"t": "error",    "m": "..."}         failed (also exits non-zero)
        {"t": "warn",     "m": "..."}         non-fatal problem worth showing
    v1.4 additions:
        {"t": "stage", "name": "load|detect|vad|transcribe|align|diarize|export",
         "state": "start|done", "detail": "..."}
    v1.5 additions:
        {"t": "wave", "bins": [200 floats 0..1], "seconds": s}
            sent before the model loads: peak amplitude per equal slice of the
            decoded audio (the waveform) and its length
        {"t": "language", "code": "ur", "prob": 0.87}
        {"t": "transcript", "segments": [...]}   final result (words, speakers)

This module has NO GUI dependencies. It exits 0 on success, 1 on error.
"""
import os
import stat
import sys
import json
import threading
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

# Make the `core` package importable regardless of how we were launched, and never
# leave core/ itself on sys.path: run as a script, its module names would shadow the
# standard library for torch / huggingface (core/queue.py did, breaking every run).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]
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
def warn(msg):         _emit({"t": "warn", "m": str(msg)})
def status(msg):       _emit({"t": "status", "m": str(msg)})
def pmode(mode):       _emit({"t": "pmode", "mode": mode})
def progress(v):       _emit({"t": "progress", "v": float(v)})


def stage(name, state, detail="", **extra):
    _emit({"t": "stage", "name": name, "state": state, "detail": detail, **extra})


def _default_model_dir():
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), "Models")
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library",
                            "Application Support", "WhisperX", "Models")
    return os.path.join(_ROOT, "Models")


# ── Overall progress ───────────────────────────────────────────────────────────
# One bar for the whole job. Load (or the model download) fills the first 4 %, export the
# last 2 %; the stages in between share the rest by their rough cost in seconds of work per
# second of audio on this device (H19 real check on an RTX 4060 / 8-core CPU: turbo transcribes
# at 0.011, large models about 3x that; align 0.009; diarize 0.12). Stages cfg turns off get
# no span; diarization always runs on the CPU (sherpa-onnx).
_LOAD, _EXPORT = 0.04, 0.02
_COST = {               # stage: (cuda, cpu)
    "transcribe": (0.03, 0.5),
    "align":      (0.01, 0.05),
    "diarize":    (0.12, 0.12),
}


class _Overall:
    """overall(stage, frac) maps a stage's own 0..1 onto the job's span and emits it as
    progress: never backwards, and only in steps of at least 0.2 % unless forced."""
    def __init__(self, cfg, t_dev):
        col = 0 if t_dev == "cuda" else 1
        cost = {k: c[col] for k, c in _COST.items() if k == "transcribe" or cfg.get(k)}
        lo, mid = _LOAD, 1 - _LOAD - _EXPORT
        self.spans = {"load": (0.0, lo)}
        for k, c in cost.items():
            self.spans[k] = (lo, lo + mid * c / sum(cost.values()))
            lo = self.spans[k][1]
        self.spans["export"] = (lo, 1.0)
        self.last = -1.0

    def __call__(self, name, frac, force=False):
        lo, hi = self.spans[name]
        v = lo + (hi - lo) * max(0.0, min(1.0, frac))
        if v > self.last and (force or v - self.last >= 0.002):
            self.last = v
            progress(v)


class _DLProgress:
    """Throttled model-download progress, mirroring the old GUI behaviour;
    report(fraction) places it on the overall bar."""
    def __init__(self, report):
        self.report = report
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
        self.report(min(1.0, frac))


class UserError(Exception):
    """A problem the user can fix; its message is shown as-is."""


def run(cfg):
    task = cfg.get("task", "transcribe")
    if task == "translate" and not pipeline.supports_translate(cfg["model"]):
        raise UserError("Translate needs a multilingual model that supports translation "
                        f"(e.g. large-v3, medium). '{cfg['model']}' can only transcribe.")
    if not cfg.get("model_dir"):
        cfg["model_dir"] = _default_model_dir()
    os.environ["HF_HOME"] = cfg["model_dir"]
    os.environ["NLTK_DATA"] = os.path.join(cfg["model_dir"], "nltk_data")
    os.environ["TORCH_HOME"] = os.path.join(cfg["model_dir"], "torch")

    # ── Device ─────────────────────────────────────────────────────────────────
    _, t_dev, a_dev = pipeline.detect_device(cfg["device"])
    compute, adjusted = pipeline.resolve_compute(t_dev, cfg["compute"])
    if adjusted:
        log("  float16 not supported on CPU — using int8")
    log(f"Device: {t_dev.upper()}   Compute: {compute}")
    if a_dev != t_dev:
        log(f"Alignment: {a_dev.upper()}")

    lang = None if cfg["language"] == "auto" else cfg["language"]
    overall = _Overall(cfg, t_dev)

    # ── Audio (first, so the waveform shows while the model loads) ──────────────
    log("Loading audio...")
    status("Loading audio...")
    if not os.path.isfile(cfg["audio"]):
        raise UserError("Could not read this file: it no longer exists.")
    try:
        audio = pipeline.load_audio(cfg["audio"])
    except Exception as e:
        msg = str(e)
        if not msg.startswith("Failed to load audio:"):
            raise
        lines = [ln.strip() for ln in msg.splitlines() if ln.strip()]
        reason = lines[-1]
        if reason.startswith(cfg["audio"] + ": "):
            reason = reason[len(cfg["audio"]) + 2:]
        raise UserError(f"Could not read this file: {reason}") from None
    try:
        secs = len(audio) / 16000
    except TypeError:
        secs = 0
    _emit({"t": "wave", "bins": pipeline.waveform_bins(audio), "seconds": round(secs, 2)})

    # ── Model ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    stage("load", "start", cfg["model"])
    overall("load", 0, force=True)
    model_cache = pipeline.model_cache_dir(cfg["model"], cfg["model_dir"])
    offline = bool(cfg.get("offline_only"))
    if not os.path.isdir(model_cache) and offline:
        raise UserError(f"Model '{cfg['model']}' is not downloaded. "
                        "Download it in Models or turn off Offline only.")
    elif not os.path.isdir(model_cache):
        log(f"Downloading model '{cfg['model']}' for the first time...")
        log("  The model will be saved and reused on all future runs.")
        status("Downloading model...")
        pmode("indeterminate")
        try:
            pipeline.download_model(
                cfg["model"], cfg["model_dir"],
                progress_cb=_DLProgress(lambda f: overall("load", f)))
            log("  Model downloaded.")
        except Exception as e:
            log(f"  Could not track progress ({e}) — downloading...")
        pmode("indeterminate")
        status("Loading model...")
    else:
        log(f"Loading model '{cfg['model']}' from cache...")
        status("Loading model...")

    model = pipeline.load_model(
        cfg["model"], t_dev, compute, lang, cfg["beam"],
        vad_onset=cfg.get("vad_onset", 0.5),
        vad_offset=cfg.get("vad_offset", 0.363),
        chunk_size=cfg.get("chunk", 30),
        task=task,
        initial_prompt=cfg.get("initial_prompt", ""),
        hotwords=cfg.get("hotwords"),
        suppress_numerals=cfg.get("suppress_numerals", False),
        vad_method=cfg.get("vad_method", "pyannote"),
        device_index=cfg.get("device_index", 0),
        threads=cfg.get("threads", 4),
        local_files_only=offline,
        decode=cfg.get("decode"))
    stage("load", "done", f"{cfg['model']} · {time.time() - t0:.1f} s")
    overall("load", 1, force=True)

    stage("vad", "start")
    stage("vad", "done", f"{secs:.0f} s audio")

    stage("detect", "start")
    prob = 1.0
    if lang is None and cfg["model"].endswith(".en"):
        lang = "en"                     # English-only models can't run detection
    elif lang is None:
        try:
            lang, prob = pipeline.detect_language(model, audio)
            log(f"  Detected language: {lang} ({prob:.2f})")
            if prob < 0.5:
                warn(f"Language guess is uncertain: {lang} {prob:.0%}. "
                     "Pick the language for a better result.")
        except Exception as e:
            prob = None
            log(f"  Language detection failed ({(str(e).splitlines() or [''])[0]}) "
                "— detecting while transcribing")
    if lang is not None:
        _emit({"t": "language", "code": lang, "prob": prob})
        stage("detect", "done", f"{lang} · {prob:.2f}")
    else:
        stage("detect", "done", "during transcription")

    # ── Transcribe ─────────────────────────────────────────────────────────────
    log("Transcribing...")
    status("Transcribing...")
    stage("transcribe", "start")
    det_mode = []

    def on_progress(p):
        if not det_mode:
            det_mode.append(1)
            pmode("determinate")
        overall("transcribe", p / 100)
        status(f"Transcribing… {p:.0f}%")

    batch = pipeline.initial_batch(cfg["batch"], t_dev, cfg.get("device_index", 0))
    if batch != cfg["batch"]:
        log(f"  Small GPU — starting at batch {batch}")
    result, det = pipeline.transcribe_with_backoff(
        model, audio, batch, lang, notify=warn,
        chunk_size=cfg.get("chunk", 30), task=task, progress_cb=on_progress)
    log(f"  Language: {det}   Segments: {len(result['segments'])}")
    segs = result["segments"]
    speech = sum(s.get("end", 0) - s.get("start", 0) for s in segs)
    stage("transcribe", "done", f"{len(segs)} chunks · {speech:.0f} s speech")
    overall("transcribe", 1, force=True)
    del model                    # drop the last reference first, then collect
    pipeline.unload_model()

    # ── Align ──────────────────────────────────────────────────────────────────
    needs_align = cfg["align"]
    align_model = cfg.get("align_model") or None
    if needs_align and task == "translate":
        warn("Word timestamps are not available for translated output; "
             "exporting segment timing only.")
        result.setdefault("language", det)
    elif needs_align and not align_model and not pipeline.align_supported(det):
        warn(f"Word timestamps are not available for {det}; "
             "exporting segment timing only.")
        result.setdefault("language", det)
    elif needs_align and not pipeline.ensure_punkt(cfg["model_dir"], offline):
        warn("Word timestamps need the NLTK sentence tokenizer, which "
             + ("is not downloaded and Offline only is on" if offline else "could not be downloaded")
             + "; exporting segment timing only.")
        result.setdefault("language", det)
    elif needs_align:
        log("Aligning word timestamps...")
        status("Aligning...")
        if align_model:
            log(f"  Alignment model: {align_model}")
        stage("align", "start", align_model or f"default model for {det}")
        failed = ""
        try:
            result = pipeline.align(
                result, det, a_dev, audio, align_model=align_model,
                interpolate_method=cfg.get("interpolate_method", "nearest"),
                return_char_alignments=bool(cfg.get("char_alignments", False)),
                progress=lambda p: overall("align", p / 100))
        except Exception as e:
            failed = (str(e).splitlines() or [""])[0] or type(e).__name__
            warn(f"Word timestamps failed: {failed}; exporting segment timing only.")
            result.setdefault("language", det)
        stage("align", "done", "failed" if failed
              else align_model or f"default model for {det}")
        overall("align", 1, force=True)
    else:
        result.setdefault("language", det)

    # ── Diarize ────────────────────────────────────────────────────────────────
    embeddings = None
    if cfg.get("diarize"):
        lo, hi = cfg.get("min_speakers"), cfg.get("max_speakers")
        want_emb = bool(cfg.get("speaker_embeddings"))
        log("Identifying speakers...")
        status("Identifying speakers...")
        stage("diarize", "start", f"{lo}–{hi} speakers" if lo and hi else "auto")
        try:
            if not pipeline.speaker_models_ready(cfg["model_dir"]):
                log("Downloading the speaker models (about 34 MB, once)...")
                pipeline.ensure_speaker_models(
                    cfg["model_dir"], bool(cfg.get("offline_only")),
                    progress=lambda f: (status(f"Downloading speaker models... {round(f * 100)}%"),
                                        overall("diarize", 0.1 * f)))
                status("Identifying speakers...")
            out = pipeline.diarize(audio, cfg["model_dir"], min_speakers=lo, max_speakers=hi,
                                   return_embeddings=want_emb, threads=cfg.get("threads") or 0,
                                   progress=lambda f: overall("diarize", 0.1 + 0.9 * f))
            segments_df, embeddings = out if want_emb else (out, None)
            result = pipeline.assign_speakers(segments_df, result, embeddings)
            n = len({s["speaker"] for s in result["segments"] if s.get("speaker")})
            stage("diarize", "done", f"{n} speakers")
        except Exception as e:
            embeddings = None
            msg = (str(e).splitlines() or [""])[0]
            warn(f"Speaker identification failed: {msg}")
            stage("diarize", "done", "failed")
        overall("diarize", 1, force=True)

    # ── Export ─────────────────────────────────────────────────────────────────
    log("Saving files...")
    status("Saving...")
    stage("export", "start")
    last_path = None
    files = []
    for fmt, path, err in pipeline.export(
            result, cfg["formats"], cfg["out_dir"], cfg["audio"],
            {"max_width": cfg["max_width"],
             "max_count": cfg["max_count"],
             "highlight": cfg["highlight"],
             "segment_resolution": cfg.get("segment_resolution", "sentence"),
             "speaker_names": cfg.get("speaker_names") or {},
             "speaker_names_in_cues": cfg.get("speaker_names_in_cues", True),
             "speaker_embeddings": embeddings}):
        if err:
            log(f"  {fmt} error: {err}")
        else:
            log(f"  {fmt:<10}->  {path}")
            last_path = path
            files.append(path)
    stage("export", "done", ", ".join(os.path.basename(f) for f in files),
          files=files)
    overall("export", 1, force=True)

    # Full result for the in-app viewer (raw speaker labels; the UI renames them)
    _emit({"t": "transcript", "segments": result["segments"]})
    log("Done!")
    _emit({"t": "done", "path": last_path})


def _start_stdin_watchdog():
    """The parent keeps our stdin open for the whole run; EOF means it died, so exit
    instead of lingering as an orphan that holds the GPU. (A stdin that is not an OS
    pipe, e.g. an in-memory or file stdin, has nothing to watch.)"""
    try:
        if not stat.S_ISFIFO(os.fstat(sys.stdin.fileno()).st_mode):
            return
    except (AttributeError, ValueError, OSError):
        return

    def watch_posix():
        try:
            while os.read(0, 65536):
                pass
        except OSError:
            pass
        os._exit(3)

    def watch_windows():
        # A blocking read on stdin stalls DLL loads (numpy/torch imports hang while it
        # is pending), so poll the pipe instead: it errors once the parent is gone.
        import ctypes
        import msvcrt
        from ctypes import wintypes
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(0))
        avail = wintypes.DWORD(0)
        peek = ctypes.windll.kernel32.PeekNamedPipe
        while peek(handle, None, 0, None, ctypes.byref(avail), None):
            if avail.value:
                os.read(0, avail.value)
            time.sleep(0.5)
        os._exit(3)

    watch = watch_windows if sys.platform == "win32" else watch_posix
    threading.Thread(target=watch, daemon=True).start()


def main():
    raw = sys.stdin.readline()
    if not raw.strip():
        _emit({"t": "error", "m": "No configuration received."})
        sys.exit(1)
    _start_stdin_watchdog()
    cfg = json.loads(raw)
    try:
        run(cfg)
    except Exception as e:
        lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
        if not isinstance(e, UserError):
            log(traceback.format_exc())
        _emit({"t": "error", "m": lines[0] if lines else type(e).__name__})
        sys.exit(1)


if __name__ == "__main__":
    main()
