"""
AppState — the one object every screen shares (settings, job queue, devices,
the runner subprocess and a tiny event bus). No widgets in here.

Screens subscribe with state.on(topic, cb) and publish with state.emit(topic, **data).
Topics used by the shell and screens:
    "settings"  key=<name>            a setting changed (after save)
    "job"       job=<Job>             a job was added/changed/removed
    "runner"    job=<Job>, ev=<dict>  one event from core/runner.py for the running job
    "finished"  job=<Job>             the running job ended (done/failed/cancelled)
    "screen"    name=<screen>         ask the shell to switch screens
    "settings-tab" name=<tab>         open Settings on that tab ("Accuracy")
    "commit"    (no data)             a run is starting: screens commit their text entries
    "status"    text=<str>            a short message for the shell's status bar
    "notice"    text=<str>            a short message for the status bar (load/save problems)
All callbacks run on the Tk thread (runner events are marshalled with `schedule`).
"""
import json
import os
import subprocess
import sys
import threading
import time

from core import settings
from core.pipeline import supports_translate
from core.jobqueue import JobQueue, queue_path, save_transcript

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Display name -> whisper language code (v1.3 list).
LANGUAGES = {
    "Auto-detect": "auto", "English": "en", "Arabic": "ar",
    "Chinese": "zh", "Dutch": "nl", "French": "fr",
    "Georgian": "ka", "German": "de", "Hindi": "hi",
    "Italian": "it", "Japanese": "ja", "Korean": "ko",
    "Pashto": "ps", "Persian": "fa", "Polish": "pl",
    "Portuguese": "pt", "Russian": "ru", "Spanish": "es",
    "Turkish": "tr", "Urdu": "ur",
}

MEDIA_TYPES = ("*.mp4 *.mkv *.avi *.mov *.mp3 *.wav *.m4a *.flac *.ogg *.webm "
               "*.aac *.wma *.opus *.m4v *.mpg *.mpeg *.ts *.3gp *.aiff *.wmv")
MEDIA_EXTS = tuple(t[1:] for t in MEDIA_TYPES.split())


_STAGE_TITLES = {
    "load": "Load model", "detect": "Detect language", "vad": "Find speech",
    "transcribe": "Transcribe", "align": "Align words",
    "diarize": "Identify speakers", "export": "Write files",
}


# ── Paths and devices (moved from app.py) ─────────────────────────────────────

def default_model_dir():
    if getattr(sys, "frozen", False):
        # PyInstaller onedir: store Models next to the .exe, not inside _internal/
        return os.path.join(os.path.dirname(sys.executable), "Models")
    if IS_MAC:
        return os.path.join(os.path.expanduser("~"), "Library",
                            "Application Support", "WhisperX", "Models")
    return os.path.join(_ROOT, "Models")


def base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return _ROOT


def available_devices():
    """Returns device list. Safe to call from any thread; imports torch lazily."""
    try:
        import torch
        devs = ["auto", "cpu"]
        if torch.cuda.is_available():
            devs.insert(1, "cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devs.insert(1, "mps")
        return devs
    except Exception:
        return ["auto", "cpu"]


# ── Settings -> runner cfg ────────────────────────────────────────────────────

_DECODE_INT   = {"best_of", "no_repeat_ngram_size", "max_new_tokens"}
_DECODE_FLOAT = {"patience", "length_penalty", "repetition_penalty",
                 "compression_ratio_threshold", "log_prob_threshold",
                 "no_speech_threshold", "prompt_reset_on_temperature",
                 "hallucination_silence_threshold"}


def _num(key, raw, default):
    """Numeric setting forced into its allowed range; the default when it is not a number."""
    v = settings.clamp(key, raw)
    return default if v is None else v


def decode_options(decode: dict) -> dict:
    """Settings' decode strings -> typed asr options. Blank or unparsable values
    are left out so the library default applies."""
    out = {}
    for key, raw in (decode or {}).items():
        s = str(raw).strip()
        if not s:
            continue
        if key == "temperatures":
            vals = settings.clamp(key, s)
            if vals:
                out[key] = vals
        elif key == "condition_on_previous_text":
            if s.lower() in ("true", "false"):
                out[key] = s.lower() == "true"
        elif key in _DECODE_INT or key in _DECODE_FLOAT:
            v = settings.clamp(key, s)
            if v is not None:
                out[key] = v
    return out


def build_cfg(s: dict, audio: str) -> dict:
    """Runner cfg (core/runner.py JSON) for one file from a settings dict."""
    md = s.get("model_dir", "").strip() or default_model_dir()
    out_dir = s.get("out_dir", "").strip() or os.path.dirname(os.path.abspath(audio))
    lo = settings.clamp("min_speakers", s["min_speakers"])
    hi = settings.clamp("max_speakers", s["max_speakers"])
    if lo and hi and lo > hi:
        lo, hi = hi, lo
    return {
        "audio":          audio,
        "model_dir":      md,
        "model":          s["model"],
        "device":         s["device"],
        "compute":        s["compute"],
        "batch":          _num("batch", s["batch"], 16),
        "beam":           _num("beam", s["beam"], 5),
        "chunk":          _num("chunk", s["chunk"], 30),
        "out_dir":        out_dir,
        "formats":        list(s["formats"]),
        "align":          bool(s["align"]),
        "language":       LANGUAGES.get(s["language"], s["language"] or "auto"),
        "task":           s["task"],
        "initial_prompt": s["initial_prompt"].strip(),
        "vad_onset":      _num("vad_onset", s["vad_onset"], 0.5),
        "vad_offset":     _num("vad_offset", s["vad_offset"], 0.363),
        "highlight":      bool(s["highlight"]),
        "max_width":      settings.clamp("max_width", s["max_width"]),
        "max_count":      settings.clamp("max_count", s["max_count"]),
        # v1.4
        "hotwords":          s["hotwords"].strip(),
        "suppress_numerals": bool(s["suppress_numerals"]),
        "vad_method":        s["vad_method"],
        "device_index":      _num("device_index", s["device_index"], 0),
        "threads":           _num("threads", s["threads"], 4),
        "offline_only":      bool(s["offline_only"]),
        "decode":            decode_options(s["decode"]),
        "align_model":       s["align_model"].strip() or None,
        "interpolate_method": s["interpolate_method"],
        "char_alignments":   bool(s["char_alignments"]),
        "diarize":           bool(s["diarize"]),
        "min_speakers":      lo,
        "max_speakers":      hi,
        "speaker_embeddings": bool(s["speaker_embeddings"]),
        "speaker_names":     {},
        "speaker_names_in_cues": bool(s["speaker_names_in_cues"]),
        "segment_resolution": s["segment_resolution"],
    }


# ── Runner subprocess ─────────────────────────────────────────────────────────

def runner_command():
    """[python, -m, core.runner] (run with cwd=_ROOT); WHISPERX_RUNNER overrides it with a
    script (tests/fake_runner.py). As a module, core/ is never on sys.path, so core's module
    names can't shadow the standard library (core/queue.py once broke every real run)."""
    script = os.environ.get("WHISPERX_RUNNER")
    return [sys.executable, script] if script else [sys.executable, "-m", "core.runner"]


# ── Windows Job Object: runner processes die with the app ─────────────────────

_job_handle = None


def _kill_on_close_job():
    """One process-lifetime Job Object with KILL_ON_JOB_CLOSE (Windows). The handle is
    never closed, so when the app dies the OS kills every runner process in it,
    including the venv launcher's grandchild. Returns the handle, or None."""
    global _job_handle
    if _job_handle is not None or not IS_WIN:
        return _job_handle
    try:
        import ctypes
        from ctypes import wintypes

        class _Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in (
                "ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

        class _Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _Basic),
                        ("IoInfo", _IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        job = k32.CreateJobObjectW(None, None)
        info = _Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000   # KILL_ON_JOB_CLOSE
        if job and k32.SetInformationJobObject(
                job, 9, ctypes.byref(info), ctypes.sizeof(info)):   # ExtendedLimitInformation
            _job_handle = job
    except Exception as e:
        print(f"[state] job object unavailable: {e}", file=sys.stderr)
    return _job_handle


def _assign_to_job(proc, job=None):
    """Put proc in the kill-on-close job (best-effort; Windows only)."""
    job = job or _kill_on_close_job()
    if job is None:
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        # Best-effort: the launcher may already have spawned the base interpreter,
        # which then stays outside the job. Cancel therefore kills the whole tree
        # (RunnerProcess.kill) and an app crash is covered by the runner's stdin
        # watchdog; the job only guarantees the launcher itself dies with the app.
        if not k32.AssignProcessToJobObject(job, int(proc._handle)):
            print("[state] runner was not added to the job object", file=sys.stderr)
    except Exception as e:
        print(f"[state] could not assign runner to job: {e}", file=sys.stderr)


class RunnerProcess:
    """One core/runner.py process. on_event(ev) and on_exit(rc) run on the reader
    thread; AppState marshals them to the Tk thread."""

    def __init__(self, cfg, on_event, on_exit):
        self.cfg, self.on_event, self.on_exit = cfg, on_event, on_exit
        self.proc = None

    def start(self):
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        kwargs = dict(cwd=_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                      stderr=subprocess.DEVNULL, env=env, text=True,
                      encoding="utf-8", errors="replace", bufsize=1)
        if IS_WIN:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.proc = subprocess.Popen(runner_command(), **kwargs)
        if IS_WIN:
            _assign_to_job(self.proc)
        try:
            # stdin stays open: the runner exits when it sees EOF (we died)
            self.proc.stdin.write(json.dumps(self.cfg) + "\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                ev = {"t": "log", "m": line}   # stray non-JSON output (e.g. traceback)
            self.on_event(ev)
        self.on_exit(self.proc.wait())

    def kill(self, block=False):
        """Stop the runner and everything it spawned; True if it was still running. In a
        Windows venv the process we start is only a launcher for the real interpreter, so
        kill the whole tree (taskkill runs on a thread unless block, so Cancel can't freeze the UI)."""
        p = self.proc
        if p is None or p.poll() is not None:
            return False
        try:
            p.stdin.close()          # EOF is the runner watchdog's signal
        except Exception:
            pass

        def tree():
            if IS_WIN:
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                                   capture_output=True, timeout=5,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                except Exception:
                    pass
            try:
                p.kill()
            except Exception:
                pass
        if IS_WIN and not block:
            threading.Thread(target=tree, daemon=True).start()
        else:
            tree()
        return True


def open_folder(path):
    """Show a folder in the system file browser (best-effort)."""
    try:
        if IS_WIN:
            os.startfile(path)
        elif IS_MAC:
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"[state] could not open {path}: {e}", file=sys.stderr)


# ── Progress estimate (Smooth Motion between runner events) ─────────────────────

class ProgressEstimate:
    """The running job's shown progress: the last reported value plus the recent speed times the
    time since, at most the previous jump (and 0.15) ahead, below 0.995 until done, never backwards.
    Reports closer together than MERGE seconds (a burst of callbacks) count as one step."""
    WINDOW, MERGE, LEAD, TOP = 4, 0.3, 0.15, 0.995

    def __init__(self):
        self.pts, self.jump, self.shown = [], 0.0, 0.0

    def add(self, v, t):
        pts = self.pts
        v = max(v, pts[-1][1]) if pts else v
        if pts and t - pts[-1][0] < self.MERGE:
            prev = pts[-2][1] if len(pts) > 1 else 0.0
            pts[-1] = (t, v)
        else:
            prev = pts[-1][1] if pts else 0.0
            self.pts = (pts + [(t, v)])[-self.WINDOW:]
        self.jump = v - prev
        return self.value(t)

    def value(self, t):
        if not self.pts:
            return self.shown
        (t0, v0), (t1, v1) = self.pts[0], self.pts[-1]
        rate = (v1 - v0) / (t1 - t0) if t1 > t0 else 0.0
        cap = min(v1 + min(self.jump, self.LEAD), self.TOP)
        self.shown = max(self.shown, min(v1 + rate * max(0.0, t - t1), cap))
        return self.shown


# ── AppState ──────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self, schedule, base=None):
        """schedule(fn) runs fn on the Tk thread (the shell passes
        a thread-safe queue's put; see Shell._drain)."""
        self.schedule = schedule
        self.base = base or base_dir()
        os.makedirs(self.base, exist_ok=True)
        self.settings_path = settings.settings_path(self.base)
        self.queue_path = queue_path(self.base)
        self.settings = settings.load(self.settings_path)
        notices = [settings.last_load_notice]
        self.queue = JobQueue.load(self.queue_path)
        notices.append(JobQueue.last_load_notice)
        self.queue.on_save_error = self._save_failed
        self.queue.base_dir = self.base
        if JobQueue.last_load_notice is None:
            self.queue.sweep_transcripts()
        self._notice_times = {}      # message -> monotonic time last emitted
        for text in filter(None, notices):   # delivered once the shell is subscribed
            self.schedule(lambda t=text: self.emit("notice", text=t))
        self.devices = ["auto", "cpu"]
        self.running = None          # Job currently running, or None
        self.transcripts = {}        # job id -> segments from the "transcript" event
        self.stages = {}             # job id -> {stage: {state, detail, t0, t1, skip}} from "stage" events
        self._proc = None
        self._cancelled = False
        self._ended = False          # the running job's "done" / export-done event arrived
        self._subs = {}
        self._stage = None           # current runner stage of the running job
        self._batch_dirs = []        # output folders to open when the batch ends
        self._est = ProgressEstimate()
        self._est_due = False        # a 30 fps estimate tick is waiting in schedule
        self._opened = False         # the running job's output folder was handled

    # event bus
    def on(self, topic, cb):
        self._subs.setdefault(topic, []).append(cb)

    def emit(self, topic, **data):
        for cb in list(self._subs.get(topic, [])):
            try:
                cb(**data)
            except Exception as e:
                print(f"[state] {topic} handler failed: {e}", file=sys.stderr)

    # settings
    def set(self, key, value):
        self.settings[key] = value
        reset = (key == "model" and self.settings.get("task") == "translate"
                 and not supports_translate(value))
        if reset:                    # holds even if the Settings screen was never built
            self.settings["task"] = "transcribe"
        self.save_settings()
        self.emit("settings", key=key)
        if reset:
            self.emit("settings", key="task")

    def save_settings(self):
        err = settings.save(self.settings_path, self.settings)
        if err:
            self._save_failed(err)

    def _save_failed(self, text):
        """Tell the user a save failed (same message at most once a minute)."""
        now = time.monotonic()
        last = self._notice_times.get(text)
        if last is None or now - last >= 60:
            self._notice_times[text] = now
            self.emit("notice", text=text)

    def model_dir(self):
        return self.settings["model_dir"].strip() or default_model_dir()

    # queue
    def add_files(self, paths):
        """Queue media files. Skips non-media extensions and files already waiting or running;
        emits ONE "job" event (job=None) for the batch and a "status" if anything was skipped.
        Returns the jobs added."""
        key = lambda p: os.path.normcase(os.path.normpath(p))
        seen = {key(j.audio) for j in self.queue.jobs if j.status in ("waiting", "running")}
        jobs, bad, dup = [], 0, 0
        for p in paths:
            if not str(p).lower().endswith(MEDIA_EXTS):
                bad += 1
            elif key(p) in seen:
                dup += 1
            else:
                seen.add(key(p))
                jobs.append(self.queue.add(p, build_cfg(self.settings, p)))
        if jobs:
            self.queue.save(self.queue_path)
            self.emit("job", job=None)
        if bad or dup:
            why = [t for t in (f"{bad} not audio/video" if bad else "",
                               f"{dup} already queued" if dup else "") if t]
            self.emit("status", text=f"Skipped {bad + dup}: " + " · ".join(why))
        return jobs

    def run_job(self, job):
        """Start one job in a runner process. Returns False if one is already running."""
        if self.running is not None:
            return False
        self.emit("commit")
        return self._launch(job)

    def _launch(self, job):
        self.running, self._cancelled, self._ended = job, False, False
        self._est, self._opened = ProgressEstimate(), False
        # Options apply at run time, so changes made after adding the file count.
        names = (job.cfg or {}).get("speaker_names")
        job.cfg = build_cfg(self.settings, job.audio)
        if isinstance(names, dict):
            job.cfg["speaker_names"] = dict(names)       # renames survive a re-run
        job.status, job.progress, job.error = "running", 0.0, None
        job.started, job.finished = time.time(), None
        job.outputs = []
        self.stages[job.id] = {}
        self.emit("job", job=job)
        self._proc = RunnerProcess(
            job.cfg,
            on_event=lambda ev: self.schedule(lambda: self._on_event(job, ev)),
            on_exit=lambda rc: self.schedule(lambda: self._on_exit(job, rc)))
        try:
            self._proc.start()
        except Exception as e:
            self._on_event(job, {"t": "error", "m": f"could not start worker: {e}"})
            # deferred, so a spawn failure can't recurse through the queue loop
            self.schedule(lambda: self._on_exit(job, 1))
            return False
        threading.Thread(target=self._est_loop, args=(job,), daemon=True).start()
        return True

    def _est_loop(self, job):
        """~30 fps while the job runs: one estimate tick at a time goes to the Tk thread."""
        while self.running is job:
            time.sleep(1 / 30)
            if not self._est_due:
                self._est_due = True
                self.schedule(lambda: self._est_tick(job))

    def _est_tick(self, job):
        """Feed the progress estimate to every screen as a progress event (marked "est")."""
        self._est_due = False
        if self.running is not job or self._ended:
            return
        v = self._est.value(time.monotonic())
        if v - job.progress >= 2e-4:
            job.progress = v
            self.emit("runner", job=job, ev={"t": "progress", "v": v, "est": True})

    def start_queue(self):
        """Start the next waiting job if none is running. Returns that job, or None."""
        if self.running is not None:
            return None
        job = self.queue.next_waiting()
        if job is None:
            return None
        self.emit("commit")
        return job if self._launch(job) else None

    def cancel(self):
        """Kill the running job's process (it ends as 'cancelled')."""
        if self._proc is not None and self._proc.kill():
            self._cancelled = True       # only when there was still a process to stop

    def _track(self, job, ev):
        """Per-job stage record (kept after the job ends, so every row can show its final dots)."""
        rec, t, now = self.stages.setdefault(job.id, {}), ev.get("t"), time.time()
        new = lambda: {"state": "pending", "detail": "", "t0": None, "t1": None, "skip": False}
        if t == "stage":
            s = rec.setdefault(ev.get("name"), new())
            if ev.get("state") == "start":
                for o in rec.values():
                    if o["state"] == "now":
                        o.update(state="done", t1=now)
                s.update(state="now", t0=now, t1=None)
            else:                                           # the runner reports a stage that ran and failed as done/"failed"
                s.update(state="failed" if ev.get("detail") == "failed" else "done",
                         t0=s["t0"] or now, t1=now)
            s["detail"] = ev.get("detail") or s["detail"]
        elif t == "warn":                                   # the runner says a stage will not run
            m = ev.get("m") or ""
            for key, start in (("align", "Word timestamps are not available"), ("align", "Word timestamps need")):
                if m.startswith(start):
                    rec.setdefault(key, new())["skip"] = True

    def _on_event(self, job, ev):
        t = ev.get("t")
        self._track(job, ev)
        if t == "done" or (t == "stage" and ev.get("name") == "export" and ev.get("state") == "done"):
            self._ended = True
        if t == "stage" and ev.get("state") == "start":
            self._stage = ev.get("name")
        if t == "progress":                                  # overall fraction of the job
            v = min(1.0, max(0.0, float(ev.get("v", 0))))
            job.progress = max(job.progress, self._est.add(v, time.monotonic()))
        elif t == "language":
            job.language, job.language_prob = ev.get("code"), ev.get("prob")
        elif t == "wave":
            if ev.get("seconds") is not None:
                job.duration = float(ev["seconds"])
            job.bins = list(ev.get("bins") or [])
        elif t == "stage" and ev.get("name") == "export" and ev.get("state") == "done":
            job.outputs = list(ev.get("files") or job.outputs)
        elif t == "done":                                   # 100 % and the folder together
            job.progress = 1.0
            if ev.get("path") and not job.outputs:
                job.outputs = [ev["path"]]
            self._open_output(job)
        elif t == "transcript":
            self.transcripts[job.id] = ev.get("segments") or []
            save_transcript(self.base, job.id, self.transcripts[job.id])
        elif t == "error":
            job.error = ev.get("m", "Unknown error")
        self.emit("runner", job=job, ev=ev)

    def _open_output(self, job):
        """Open the job's output folder now, or note it for the end of the batch (once per job)."""
        if self._opened or not self.settings.get("open_folder"):
            return
        self._opened = True
        folder = job.cfg.get("out_dir") or os.path.dirname(os.path.abspath(job.audio))
        if not self.settings.get("open_folder_per_batch"):
            open_folder(folder)
        elif folder not in self._batch_dirs:
            self._batch_dirs.append(folder)

    def _on_exit(self, job, rc):
        if self._cancelled and not self._ended:
            job.status = "cancelled"
        elif job.error or (rc != 0 and not self._cancelled):
            job.status = "failed"
            job.error = job.error or (
                f"The worker crashed during {_STAGE_TITLES.get(self._stage, 'start-up')} "
                f"(code {rc}).")
        else:
            job.status, job.progress = "done", 1.0
        job.finished = time.time()
        for st in self.stages.get(job.id, {}).values():     # a stage still running was cut short
            if st["state"] == "now":
                st.update(state="done" if job.status == "done" else "stopped", t1=job.finished)
        self.running, self._proc, self._stage = None, None, None
        if job.status == "done":
            self._open_output(job)
        self.queue.save(self.queue_path)
        self.emit("job", job=job)
        self.emit("finished", job=job)
        # Back-to-back running; a cancelled job just skips to the next file.
        if self.settings.get("queue_autostart") and not (
                job.status == "failed" and self.settings.get("queue_on_failure") == "stop"):
            self.start_queue()
        if self.running is None:                     # the batch is over
            for folder in self._batch_dirs:
                open_folder(folder)
            self._batch_dirs = []

    def shutdown(self):
        self.save_settings()
        self.queue.save(self.queue_path)
        if self._proc is not None:
            self._proc.kill(block=True)
