"""
Persisted GUI settings — stdlib only, no GUI imports.

Contract (see .orchestrate/plan.md):
    settings_path(base_dir) -> str
    load(path) -> dict       DEFAULTS merged with the JSON file; tolerant of
                             missing/corrupt files, unknown keys, wrong types.
    save(path, data) -> None Writes only DEFAULTS keys; never raises.
"""
import copy
import json
import os
import sys
import tempfile
import time

DEFAULTS = {
    "file":           "",
    "out_dir":        "",
    "language":       "Auto-detect",
    "model":          "large-v2",
    "device":         "auto",
    "compute":        "float16",
    "batch":          "16",
    "beam":           "5",
    "chunk":          "30",
    "formats":        ["srt"],
    "align":          True,
    "open_folder":    True,
    "highlight":      False,
    "vad_onset":      "0.500",
    "vad_offset":     "0.363",
    "max_width":      "",
    "max_count":      "",
    "model_dir":      "",          # empty = use the app default
    "theme":          "dark",
    "task":           "transcribe",
    "initial_prompt": "",
    # ── v1.4: transcription options ──
    "hotwords":         "",
    "suppress_numerals": False,
    "vad_method":       "pyannote",
    "device_index":     "0",
    "threads":          "4",
    "offline_only":     False,
    "decode": {                      # advanced decode options; "" = library default
        "temperatures": "", "best_of": "", "patience": "", "length_penalty": "",
        "repetition_penalty": "", "no_repeat_ngram_size": "",
        "compression_ratio_threshold": "", "log_prob_threshold": "",
        "no_speech_threshold": "", "condition_on_previous_text": "",
        "prompt_reset_on_temperature": "", "hallucination_silence_threshold": "",
        "max_new_tokens": "",
    },
    # ── v1.4: alignment ──
    "align_model":      "",
    "interpolate_method": "nearest",
    "char_alignments":  False,
    # ── v1.4: diarization ──
    "diarize":          False,
    "min_speakers":     "",
    "max_speakers":     "",
    "speaker_embeddings": False,
    "segment_resolution": "sentence",
    "speaker_names_in_cues": True,
    # ── v1.4: queue and output ──
    "queue_autostart":  True,
    "queue_on_failure": "continue",
    "open_folder_per_batch": True,
    "last_dir":         "",          # folder the Settings pickers last used
    "ui_scale":         "auto",      # "auto" or a number such as "1.5" (settings file only)
}


# ── Field rules ───────────────────────────────────────────────────────────────
# key -> (kind, lo, hi, blank_ok); kind: int float floats(comma list) bool(true/false). Decode keys are
# written "decode.<name>" (the bare name works too). Keys not listed accept any text.
FIELD_RULES = {
    "beam": ("int", 1, 10, False), "batch": ("int", 1, 64, False), "chunk": ("int", 5, 30, False),
    "threads": ("int", 1, 64, False), "device_index": ("int", 0, 15, False),
    "vad_onset": ("float", 0.05, 0.95, False), "vad_offset": ("float", 0.05, 0.95, False),
    "min_speakers": ("int", 1, 20, True), "max_speakers": ("int", 1, 20, True),
    "max_width": ("int", 10, 200, True), "max_count": ("int", 1, 5, True),
    "decode.temperatures": ("floats", 0, 1, True), "decode.best_of": ("int", 1, 10, True),
    "decode.patience": ("float", 0.1, 5, True), "decode.length_penalty": ("float", 0.1, 5, True),
    "decode.repetition_penalty": ("float", 0.5, 3, True),
    "decode.no_repeat_ngram_size": ("int", 0, 10, True),
    "decode.compression_ratio_threshold": ("float", 1, 5, True),
    "decode.log_prob_threshold": ("float", -5, 0, True),
    "decode.no_speech_threshold": ("float", 0, 1, True),
    "decode.prompt_reset_on_temperature": ("float", 0, 1, True),
    "decode.hallucination_silence_threshold": ("float", 0.1, 10, True),
    "decode.max_new_tokens": ("int", 1, 448, True),
    "decode.condition_on_previous_text": ("bool", None, None, True),
}


def _num(kind, text):
    """Parse one number; None when it is not a finite number of that kind."""
    try:
        v = int(text) if kind == "int" else float(text)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def _fmt(v):
    return str(v) if isinstance(v, int) else format(v, "g")


def validate(key, value, settings=None):
    """-> (ok, normalised text, reason). Blank is fine where the rule allows it. With `settings`,
    minimum/maximum speakers are also checked against each other."""
    rule = FIELD_RULES.get(key) or FIELD_RULES.get("decode." + key)
    text = str(value).strip()
    if rule is None:
        return True, text, ""
    kind, lo, hi, blank_ok = rule
    if not text:
        return (True, "", "") if blank_ok else (False, text, "Required.")
    if kind == "bool":
        ok = text.lower() in ("true", "false")
        return (True, text.lower(), "") if ok else (False, text, "Enter true or false.")
    rng = f"{_fmt(lo)} to {_fmt(hi)}"
    parts = [p.strip() for p in text.split(",")] if kind == "floats" else [text]
    vals = [_num("int" if kind == "int" else "float", p) for p in parts]
    if None in vals:
        what = "a whole number" if kind == "int" else "numbers separated by commas" if kind == "floats" else "a number"
        return False, text, f"Enter {what} from {rng}."
    if any(not lo <= v <= hi for v in vals):
        return False, text, f"Must be {rng}."
    if settings is not None and key in ("min_speakers", "max_speakers"):
        other = settings.get("max_speakers" if key == "min_speakers" else "min_speakers", "")
        o = _num("int", str(other).strip())
        if o is not None and ((key == "min_speakers" and vals[0] > o) or (key == "max_speakers" and vals[0] < o)):
            return False, text, "Minimum speakers cannot be above maximum."
    return True, ", ".join(_fmt(v) for v in vals), ""


def clamp(key, value):
    """Typed value of a numeric setting forced into its range (list for temperatures);
    None when blank or not a number, so the caller applies its default."""
    rule = FIELD_RULES.get(key) or FIELD_RULES.get("decode." + key)
    if rule is None or rule[0] in ("bool",):
        return None
    kind, lo, hi, _ = rule
    vals = [_num("int" if kind == "int" else "float", p.strip()) for p in str(value).split(",")]
    if None in vals or (kind != "floats" and len(vals) != 1):
        return None
    vals = [min(hi, max(lo, v)) for v in vals]
    return vals if kind == "floats" else vals[0]


# Set by every load(): text for the UI when the file was unreadable, else None.
last_load_notice = None


def quarantine(path: str):
    """Rename an unreadable data file to <name>.corrupt-<YYYYmmdd-HHMMSS>. Returns the
    notice text for the UI (also when the rename itself fails)."""
    name = os.path.basename(path)
    kept = f"{name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        os.replace(path, os.path.join(os.path.dirname(path), kept))
    except OSError as e:
        print(f"[settings] could not keep a copy of {path}: {e}", file=sys.stderr)
        return f"{name} was unreadable and could not be set aside; started from defaults"
    return f"{name} was unreadable; kept a copy as {kept} and started from defaults"


def acquire_lock(base_dir: str):
    """Exclusive non-blocking lock on <base_dir>/.whisperx.lock. Returns the open handle
    (keep it for the process lifetime) or None if another instance holds it. The OS
    drops the lock when the process dies, so a stale file never blocks."""
    os.makedirs(base_dir, exist_ok=True)
    f = open(os.path.join(base_dir, ".whisperx.lock"), "a+b")
    try:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def release_lock(f) -> None:
    """Unlock and close a handle from acquire_lock()."""
    try:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    f.close()


def settings_path(base_dir: str) -> str:
    """os.path.join(base_dir, 'settings.json')"""
    return os.path.join(base_dir, "settings.json")


def load(path: str) -> dict:
    """DEFAULTS merged with the JSON at path. Missing/corrupt file -> DEFAULTS copy.
    Unknown keys are dropped; values with the wrong type fall back to the default.
    An unparsable file is renamed aside first (see quarantine, last_load_notice)."""
    global last_load_notice
    last_load_notice = None
    result = copy.deepcopy(DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except OSError:
        return result
    except ValueError:
        last_load_notice = quarantine(path)
        return result
    if not isinstance(raw, dict):
        last_load_notice = quarantine(path)
        return result
    for key, default in DEFAULTS.items():
        if key == "decode":
            if isinstance(raw.get(key), dict):
                for sub, sub_default in default.items():
                    sub_value = raw[key].get(sub)
                    if type(sub_value) is type(sub_default):
                        result[key][sub] = sub_value
        elif key in raw and type(raw[key]) is type(default):
            result[key] = copy.deepcopy(raw[key])
    return result


def save(path: str, data: dict):
    """Write only DEFAULTS keys, JSON, utf-8, indent 2. Never raises; returns the error
    text on failure, else None."""
    tmp = None
    try:
        out = {k: data.get(k, v) for k, v in DEFAULTS.items()}
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[settings] could not save {path}: {e}", file=sys.stderr)
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return f"Could not save settings: {getattr(e, 'strerror', None) or e}"
