"""
WhisperX pipeline — pure computation, no GUI dependencies.
Shared by app.py (_worker) and transcribe.py (main).
"""
import os
import json


# ── Shared contracts (pure data, no torch/whisperx imports) ──────────────────

# Languages whisperx 3.8.x ships an alignment model for (whisperx/alignment.py).
ALIGN_LANGS = frozenset({
    "en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt", "ar", "cs", "ru", "pl",
    "hu", "fi", "fa", "el", "tr", "da", "he", "vi", "ko", "ur", "te", "hi", "ca", "ml",
    "no", "nn", "sk", "sl", "hr", "ro", "eu", "gl", "ka", "lv", "tl", "sv", "id",
})


def align_supported(lang) -> bool:
    """True if whisperx can word-align this language code ('auto'/None -> True)."""
    if lang is None or lang == "auto":
        return True
    return lang in ALIGN_LANGS


# name -> (HuggingFace repo id, approximate download size in MB)
MODEL_INFO = {
    "tiny":            ("Systran/faster-whisper-tiny", 75),
    "tiny.en":         ("Systran/faster-whisper-tiny.en", 75),
    "base":            ("Systran/faster-whisper-base", 145),
    "base.en":         ("Systran/faster-whisper-base.en", 145),
    "small":           ("Systran/faster-whisper-small", 465),
    "small.en":        ("Systran/faster-whisper-small.en", 465),
    "medium":          ("Systran/faster-whisper-medium", 1500),
    "medium.en":       ("Systran/faster-whisper-medium.en", 1500),
    "large-v1":        ("Systran/faster-whisper-large-v1", 3000),
    "large-v2":        ("Systran/faster-whisper-large-v2", 3000),
    "large-v3":        ("Systran/faster-whisper-large-v3", 3000),
    "large-v3-turbo":  ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", 1600),
    "distil-large-v3": ("Systran/faster-distil-whisper-large-v3", 1500),
    "distil-large-v3.5": ("distil-whisper/distil-large-v3.5-ct2", 1500),
    "distil-medium.en": ("Systran/faster-distil-whisper-medium.en", 800),
    "distil-small.en": ("Systran/faster-distil-whisper-small.en", 340),
}


# CTranslate2 compute types offered in the UI (resolve_compute still adjusts for CPU).
COMPUTE_TYPES = ("float16", "int8_float16", "int8", "bfloat16", "int8_bfloat16", "float32")

# Export format ids; "aud" = Audacity label file (whisperx WriteAudacity).
EXPORT_FORMATS = ("srt", "vtt", "txt", "tsv", "json", "word_json", "aud")


def model_repo(model: str) -> str:
    """HF repo id for a model name; unknown names map to Systran/faster-whisper-<model>."""
    info = MODEL_INFO.get(model)
    return info[0] if info else f"Systran/faster-whisper-{model}"


def model_cache_dir(model: str, model_dir: str) -> str:
    """{model_dir}/hub/models--{repo with '/' replaced by '--'}"""
    return os.path.join(model_dir, "hub",
                        "models--" + model_repo(model).replace("/", "--"))


def format_srt_timestamp(seconds: float) -> str:
    """Convert a float number of seconds to SRT timestamp format: HH:MM:SS,mmm"""
    assert seconds >= 0, f"Timestamp must be non-negative, got {seconds}"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = round((seconds % 1) * 1000)
    # Handle rounding up to a full second
    if ms == 1000:
        ms = 0
        secs += 1
        if secs == 60:
            secs = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def detect_device(requested: str) -> tuple:
    """
    Returns (device, t_dev, a_dev).
      device — resolved device: "cuda", "mps", or "cpu"
      t_dev  — transcription device (MPS uses CPU; whisperx doesn't support MPS for ASR)
      a_dev  — alignment device (MPS is fine for alignment)
    """
    import torch
    if requested == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested
    t_dev = "cpu" if device == "mps" else device
    a_dev = device
    return device, t_dev, a_dev


def resolve_compute(t_dev: str, compute: str) -> tuple:
    """
    Returns (compute_type, was_adjusted).
    float16 and int8_float16 are not supported on CPU — falls back to int8.
    """
    if t_dev == "cpu" and compute in ("float16", "int8_float16"):
        return "int8", True
    return compute, False


def _dir_size(path: str) -> int:
    """Total size in bytes of every file under path (0 if it doesn't exist)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def download_model(model: str, model_dir: str,
                   progress_cb=None, cancel_cb=None) -> None:
    """
    Pre-download a faster-whisper model with real progress reporting.

    progress_cb(done_bytes, total_bytes) is called ~3x/sec while downloading,
    for display only (exceptions from it are ignored).  cancel_cb() is polled
    between files; return True from it to abort the download.

    Stores into {model_dir}/hub so a later load_model() finds it cached and
    does not re-download.  Raises on download failure — the caller may fall
    back to load_model(), which downloads the model itself (no progress).

    Progress is measured by polling the size of the download folder on disk
    rather than via tqdm: huggingface_hub only routes a custom tqdm to its
    outer "Fetching N files" counter, not the per-byte file bars, so a tqdm
    hook reports file-count progress (effectively 0%) instead of bytes.
    """
    import threading
    import time as _time
    from huggingface_hub import snapshot_download

    repo       = model_repo(model)
    model_root = model_cache_dir(model, model_dir)
    cache_dir  = os.path.join(model_dir, "hub")

    # Total download size, fetched up front so the fraction has a fixed
    # denominator (the progress bar fills smoothly from 0 to 100%).
    total_size = 0
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        total_size = sum((s.size or 0) for s in (info.siblings or []))
    except Exception:
        total_size = 0

    stop = threading.Event()

    def _poll():
        while not stop.is_set():
            sz = _dir_size(model_root)
            try:
                progress_cb(min(sz, total_size), total_size)
            except Exception:
                pass
            stop.wait(0.3)

    poller = None
    if progress_cb is not None and total_size:
        poller = threading.Thread(target=_poll, daemon=True)
        poller.start()

    # A tqdm subclass purely to honor cancellation (checked between files).
    cancel_tqdm = None
    if cancel_cb is not None:
        try:
            from huggingface_hub.utils import tqdm as _BaseTqdm
        except Exception:
            from tqdm.auto import tqdm as _BaseTqdm

        class _CancelTqdm(_BaseTqdm):
            def update(self, n=1):
                if cancel_cb():
                    raise RuntimeError("cancelled")
                return super().update(n)

        cancel_tqdm = _CancelTqdm

    try:
        kwargs = {"cache_dir": cache_dir, "max_workers": 1}
        if cancel_tqdm is not None:
            try:
                snapshot_download(repo, tqdm_class=cancel_tqdm, **kwargs)
            except TypeError:
                snapshot_download(repo, **kwargs)
        else:
            snapshot_download(repo, **kwargs)
    finally:
        stop.set()
        # Land cleanly on 100% (unless we were cancelled)
        if progress_cb is not None and total_size and not (cancel_cb and cancel_cb()):
            try:
                progress_cb(total_size, total_size)
            except Exception:
                pass


def load_model(model: str, t_dev: str, compute: str, language, beam: int, *,
               vad_onset: float = 0.5, vad_offset: float = 0.363, chunk_size: int = 30,
               task: str = "transcribe", initial_prompt=None,
               hotwords=None, suppress_numerals: bool = False, vad_method: str = "pyannote",
               device_index: int = 0, threads: int = 4, local_files_only: bool = False,
               decode: dict | None = None):
    """Load and return a WhisperX ASR model."""
    import whisperx
    asr_options = {"beam_size": beam, "initial_prompt": initial_prompt or None,
                   "hotwords": hotwords or None, "suppress_numerals": bool(suppress_numerals)}
    asr_options.update(decode or {})
    return whisperx.load_model(
        model, t_dev,
        compute_type=compute,
        language=language,
        asr_options=asr_options,
        vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset,
                     "chunk_size": chunk_size},
        task=task,
        vad_method=vad_method,
        device_index=device_index,
        threads=threads,
        local_files_only=local_files_only,
    )


def _ensure_ffmpeg_on_path() -> None:
    """
    whisperx.load_audio() invokes a bare ``ffmpeg``.  imageio-ffmpeg ships a
    version-named binary (e.g. ``ffmpeg-win-x86_64-v7.1.exe``), which a bare
    ``ffmpeg`` call cannot find — that is the cause of ``[WinError 2]``.

    Expose the bundled binary under the plain name ``ffmpeg(.exe)`` and put its
    folder on PATH so the call resolves.  Idempotent and best-effort.
    """
    import shutil
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return
    if not exe or not os.path.isfile(exe):
        return

    bin_dir = os.path.dirname(exe)
    name    = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    target  = os.path.join(bin_dir, name)
    if not os.path.isfile(target):
        try:
            os.link(exe, target)        # instant, no extra disk space (same volume)
        except Exception:
            try:
                shutil.copy2(exe, target)   # fallback if hardlink unsupported
            except Exception:
                return
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def load_audio(path: str):
    """Load audio from file and return audio array."""
    import whisperx
    _ensure_ffmpeg_on_path()
    return whisperx.load_audio(path)


def waveform_bins(audio, n: int = 200) -> list:
    """Peak |amplitude| of n equal slices, normalised so the max is 1.0."""
    if hasattr(audio, "reshape"):                       # numpy fast path
        import numpy as np
        a = np.abs(audio)
        per = max(1, -(-a.size // n))                   # ceil(size / n)
        a = np.pad(a, (0, per * n - a.size))
        peaks = a.reshape(n, per).max(axis=1)
        top = float(peaks.max()) if peaks.size else 0.0
        return [float(x) / top if top else 0.0 for x in peaks]
    a = [abs(float(x)) for x in audio]
    per = max(1, -(-len(a) // n))
    peaks = [max(a[i * per:(i + 1) * per], default=0.0) for i in range(n)]
    top = max(peaks)
    return [x / top if top else 0.0 for x in peaks]


def transcribe(model, audio, batch_size: int, language, *,
               chunk_size: int = 30, task: str = "transcribe",
               progress_cb=None) -> tuple:
    """
    Transcribe audio. Returns (result, detected_language).
    Pass language=None to auto-detect. progress_cb(percent 0-100) is called
    per chunk by whisperx.
    """
    kwargs = {"batch_size": batch_size, "chunk_size": chunk_size, "task": task}
    if language:
        kwargs["language"] = language
    if progress_cb is not None:
        kwargs["progress_callback"] = progress_cb
    result = model.transcribe(audio, **kwargs)
    detected = result.get("language", language or "en")
    return result, detected


def supports_translate(model: str) -> bool:
    """False for models that can only transcribe: turbo, English-only (.en) and distil."""
    name = model.lower()
    return not ("turbo" in name or name.endswith(".en") or name.startswith("distil"))


def initial_batch(batch: int, t_dev: str, device_index: int = 0, *, memory=None) -> int:
    """Start the default CUDA batch lower on GPUs with small total or free memory."""
    if batch != 16 or t_dev != "cuda":
        return batch
    try:
        free = None
        if memory is None:
            import torch
            try:
                free, total = torch.cuda.mem_get_info(device_index)
            except Exception:
                total = torch.cuda.get_device_properties(device_index).total_memory
        else:
            free, total = memory
    except Exception:
        return batch
    start = 8 if total < 10 * 1024 ** 3 else batch
    if free is not None:
        if free < 1.5 * 1024 ** 3:
            start = min(start, 2)
        elif free < 3 * 1024 ** 3:
            start = min(start, 4)
    return start


def transcribe_with_backoff(model, audio, batch_size: int, language, *, notify=None, **kw):
    """transcribe(), halving the batch (down to 1) and retrying when the GPU runs out of memory."""
    recovering = False
    while True:
        try:
            return transcribe(model, audio, batch_size, language, **kw)
        except Exception as e:
            text = str(e).lower()      # pyannote's VAD rewraps CUDA OOM as "...probably too large"
            oom = "out of memory" in text or "probably too large" in text
            if not (oom or recovering) or (oom and batch_size <= 1):
                raise
        if not oom:                    # CTranslate2 fails once more right after an OOM, then recovers
            recovering = False
            continue
        recovering = True
        batch_size = max(1, batch_size // 2)
        unload_model()
        if notify:
            notify(f"GPU memory is full — retrying with batch {batch_size}")


def unload_model(model=None) -> None:
    """Free the ASR model's memory (CTranslate2 and the HF aligner crash together).
    Callers drop their own reference first; this collects and empties the CUDA cache."""
    try:
        del model
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass


def ensure_punkt(model_dir: str, offline: bool) -> bool:
    """True if NLTK's punkt_tab sentence tokenizer is under <model_dir>/nltk_data."""
    nltk_dir = os.path.join(model_dir, "nltk_data")
    if os.path.isdir(os.path.join(nltk_dir, "tokenizers", "punkt_tab")):
        return True
    if offline:
        return False
    try:
        import nltk
        os.makedirs(nltk_dir, exist_ok=True)
        nltk.download("punkt_tab", download_dir=nltk_dir, quiet=True)
    except Exception:
        return False
    return os.path.isdir(os.path.join(nltk_dir, "tokenizers", "punkt_tab"))


def align(result: dict, detected_lang: str, a_dev: str, audio, *,
          align_model=None, interpolate_method: str = "nearest",
          return_char_alignments: bool = False, progress=None) -> dict:
    """
    Word-level alignment. Returns updated result with word timestamps.
    progress(percent 0-100) is called per segment by whisperx.
    Tries a_dev first; falls back to CPU if that fails.
    Preserves the "language" key that whisperx.align() drops.
    Raises if alignment fails on both devices.
    """
    import whisperx

    def _do_align(device):
        load_kw = {"model_name": align_model} if align_model else {}
        model_a, meta = whisperx.load_align_model(
            language_code=detected_lang, device=device, **load_kw)
        return whisperx.align(result["segments"], model_a, meta, audio, device,
                              interpolate_method=interpolate_method,
                              return_char_alignments=return_char_alignments,
                              **({"progress_callback": progress} if progress else {}))

    try:
        aligned = _do_align(a_dev)
    except Exception:
        if a_dev == "cpu":
            raise
        aligned = _do_align("cpu")

    aligned.setdefault("language", detected_lang)
    return aligned


def export(result: dict, formats: list, out_dir: str,
           audio_path: str, opts: dict) -> list:
    """
    Export result to the requested formats.
    Returns list of (fmt, path, error) tuples — error is None on success.
    opts keys: max_width, max_count, highlight, segment_resolution,
    speaker_names, speaker_names_in_cues, speaker_embeddings.
    """
    import copy
    from whisperx.utils import get_writer

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    results = []

    # Renamed deep copy: the caller's result is never mutated
    names = {k: v.strip() for k, v in (opts.get("speaker_names") or {}).items()
             if isinstance(v, str) and v.strip()}
    result = copy.deepcopy(result)
    if names:
        for seg in result["segments"]:
            for item in [seg, *seg.get("words", [])]:
                if item.get("speaker") in names:
                    item["speaker"] = names[item["speaker"]]
    no_cue_names = not opts.get("speaker_names_in_cues", True)

    if "word_json" in formats:
        words = [w for seg in result["segments"] for w in seg.get("words", [])]
        path = os.path.join(out_dir, base + "_words.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(words, fh, indent=2, ensure_ascii=False)
            results.append(("word_json", path, None))
        except Exception as e:
            results.append(("word_json", None, str(e)))

    writer_opts = {
        "max_line_width":  opts.get("max_width"),
        "max_line_count":  opts.get("max_count"),
        "highlight_words": opts.get("highlight", False),
        # whisperx 3.8.6 parses segment_resolution but its writers ignore it: pass-through
        "segment_resolution": opts.get("segment_resolution", "sentence"),
    }
    for fmt in formats:
        if fmt == "word_json":
            continue
        path = os.path.join(out_dir, base + "." + fmt)
        out = result
        if no_cue_names and fmt in ("srt", "vtt"):
            out = copy.deepcopy(result)
            for seg in out["segments"]:
                seg.pop("speaker", None)
        try:
            get_writer(fmt, out_dir)(out, audio_path, writer_opts)
            results.append((fmt, path, None))
        except Exception as e:
            results.append((fmt, None, str(e)))

    embeddings = opts.get("speaker_embeddings")
    if embeddings:
        path = os.path.join(out_dir, base + "_speakers.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(embeddings, fh, indent=2, ensure_ascii=False)
            results.append(("speakers", path, None))
        except Exception as e:
            results.append(("speakers", None, str(e)))

    return results


# ── v1.4 contracts (implemented by later tasks) ─────────────────────────────

def detect_language(model, audio) -> tuple:
    """(code, probability) for the first 30 s of audio."""
    from whisperx.audio import log_mel_spectrogram, N_SAMPLES
    n_mels = model.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(
        audio[:N_SAMPLES],
        n_mels=n_mels if n_mels is not None else 80,
        padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
    encoder_output = model.model.encode(segment)
    token, prob = model.model.model.detect_language(encoder_output)[0][0]
    return token[2:-2], float(prob)


# Speaker identification runs on sherpa-onnx with open models from its GitHub releases, so it needs
# no account, token or terms page (owner rule: no sign-up blockers). Both download on first use.
_SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SPEAKER_MODELS = {      # key -> (path under <model_dir>/speakers, download url, approx bytes)
    "segmentation": ("sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
                     f"{_SHERPA}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
                     6_958_444),
    "embedding": ("wespeaker_en_voxceleb_resnet34_LM.onnx",
                  f"{_SHERPA}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
                  26_530_550),
}


def speaker_model_path(model_dir: str, key: str) -> str:
    return os.path.join(model_dir, "speakers", *SPEAKER_MODELS[key][0].split("/"))


def speaker_models_ready(model_dir: str) -> bool:
    return all(os.path.isfile(speaker_model_path(model_dir, k)) for k in SPEAKER_MODELS)


def ensure_speaker_models(model_dir: str, offline: bool, progress=None) -> None:
    """Download the speaker models that are missing. progress(fraction) is called while
    downloading. Raises RuntimeError when offline and a model is missing."""
    import tarfile
    import tempfile
    import urllib.request
    missing = [k for k in SPEAKER_MODELS if not os.path.isfile(speaker_model_path(model_dir, k))]
    if not missing:
        return
    if offline:
        raise RuntimeError("The speaker models are not downloaded yet. Turn off Offline only once "
                           "to download them (about 34 MB).")
    root = os.path.join(model_dir, "speakers")
    os.makedirs(root, exist_ok=True)
    total = sum(SPEAKER_MODELS[k][2] for k in missing) or 1
    done = 0
    for key in missing:
        rel, url, size = SPEAKER_MODELS[key]
        fd, tmp = tempfile.mkstemp(dir=root, suffix=".part")
        os.close(fd)
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(min(1.0, done / total))
            if url.endswith(".tar.bz2"):
                with tarfile.open(tmp) as t:
                    t.extractall(root, filter="data")     # data filter: no absolute paths or links
            else:
                os.replace(tmp, os.path.join(root, *rel.split("/")))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def diarize(audio, model_dir: str, *, min_speakers=None, max_speakers=None,
            return_embeddings: bool = False, threads: int = 0, progress=None):
    """Speaker diarization on 16 kHz mono float audio -> DataFrame(start, end, speaker), or
    (DataFrame, {speaker: embedding}) when return_embeddings. Speakers are labelled
    SPEAKER_00, SPEAKER_01, ... like pyannote's, so renames and exports are unchanged.
    progress(fraction 0..1) is called per processed chunk."""
    import numpy as np
    import pandas as pd
    import sherpa_onnx as so
    threads = threads or max(1, min(8, (os.cpu_count() or 2) // 2))
    seg_cfg = so.OfflineSpeakerSegmentationModelConfig(
        pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=speaker_model_path(model_dir, "segmentation")), num_threads=threads)
    emb_cfg = so.SpeakerEmbeddingExtractorConfig(
        model=speaker_model_path(model_dir, "embedding"), num_threads=threads)
    fixed = min_speakers if min_speakers and min_speakers == max_speakers else -1

    def run(n):
        cfg = so.OfflineSpeakerDiarizationConfig(
            segmentation=seg_cfg, embedding=emb_cfg,
            clustering=so.FastClusteringConfig(num_clusters=n, threshold=0.5),
            min_duration_on=0.3, min_duration_off=0.5)
        d = so.OfflineSpeakerDiarization(cfg)
        if progress is None:
            return d.process(audio).sort_by_start_time()
        return d.process(audio, callback=lambda done, total: (
            progress(done / total) if total else None, 0)[1]).sort_by_start_time()

    segs = run(fixed)
    found = len({s.speaker for s in segs})
    if fixed == -1 and segs and (min_speakers and found < min_speakers or max_speakers and found > max_speakers):
        segs = run(max(min_speakers or 1, min(found, max_speakers or found)))   # clamp into the range
    df = pd.DataFrame([{"start": s.start, "end": s.end, "speaker": f"SPEAKER_{s.speaker:02d}"} for s in segs],
                      columns=["start", "end", "speaker"])
    if not return_embeddings:
        return df
    extractor = so.SpeakerEmbeddingExtractor(emb_cfg)
    embeddings = {}
    for spk, rows in df.groupby("speaker"):
        clip = np.concatenate([audio[int(r.start * 16000):int(r.end * 16000)] for r in rows.itertuples()])[:30 * 16000]
        stream = extractor.create_stream()
        stream.accept_waveform(16000, clip)
        stream.input_finished()
        embeddings[spk] = list(map(float, extractor.compute(stream)))
    return df, embeddings


def assign_speakers(diarize_segments, result: dict, speaker_embeddings=None) -> dict:
    """whisperx.assign_word_speakers wrapper."""
    import whisperx
    return whisperx.assign_word_speakers(diarize_segments, result, speaker_embeddings)


def _model_kind(repo: str) -> str:
    """Classify a cached repo id: whisper | align | vad | diarize | other."""
    low = repo.lower()
    if any(r == repo for r, _mb in MODEL_INFO.values()) or "whisper" in low:
        return "whisper"
    if "wav2vec2" in low:
        return "align"
    if "vad" in low or "silero" in low or low.startswith("pyannote/segmentation"):
        return "vad"
    if low.startswith(("pyannote/", "speechbrain/")):
        return "diarize"
    return "other"


def cached_models(model_dir: str) -> list:
    """[{name, kind ("whisper"|"align"|"diarize"|"vad"|"other"), size_bytes, path}]
    found under {model_dir}/hub, sorted by (kind, name)."""
    hub = os.path.join(model_dir, "hub")
    if not os.path.isdir(hub):
        return []
    by_repo = {repo: name for name, (repo, _mb) in MODEL_INFO.items()}
    found = []
    for entry in os.listdir(hub):
        path = os.path.join(hub, entry)
        if not entry.startswith("models--") or not os.path.isdir(path):
            continue
        org, sep, rest = entry[len("models--"):].partition("--")
        if not sep:
            continue
        repo = org + "/" + rest
        found.append({
            "name": by_repo.get(repo, repo),
            "kind": _model_kind(repo),
            "size_bytes": _dir_size(path),
            "path": os.path.abspath(path),
        })
    found.sort(key=lambda e: (e["kind"], e["name"]))
    return found


def delete_cached_model(model_dir: str, name: str) -> None:
    """Remove one cached model folder; KeyError for unknown names."""
    import shutil
    for entry in cached_models(model_dir):
        if entry["name"] == name:
            shutil.rmtree(entry["path"])
            return
    raise KeyError(name)
