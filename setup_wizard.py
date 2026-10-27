"""
WhisperX Transcriber — first-run setup wizard.

Downloads and installs the AI backend (torch + whisperx) into a local
runtime/ venv on first launch. Uses ONLY stdlib + customtkinter + tkinter
so it can run from the thin PyInstaller launcher without any ML packages.
"""
import math
import os
import sys
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import tkinter as tk
import urllib.parse
import urllib.request
import webbrowser
import customtkinter as ctk

_PYTHON_VERSION   = "3.13.7"
_PYTHON_URL       = f"https://www.python.org/ftp/python/{_PYTHON_VERSION}/python-{_PYTHON_VERSION}-amd64.exe"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Colours (single dark values)
GROUND   = "#0c0c0e"
PANEL    = "#151517"
RAISED   = "#1e1e21"
WELL     = "#101012"
RULE     = "#26262a"
INK      = "#f3efe8"
INK_2    = "#b9b3aa"
INK_3    = "#8f8981"
HEAT     = "#ff6a1a"
HEAT_HOV = "#ff7d38"
EMBER    = "#ffb067"
OK       = "#38e07b"
WARN     = "#ffb020"
BAD      = "#ff4d4f"
ON_HEAT  = "#1a0d05"
INERT    = "#3a3a40"

# Typeface per weight -> (family, tk weight); swapped for the bundled fonts by _load_fonts()
_FACES = {"normal": ("Segoe UI", "normal"), "medium": ("Segoe UI", "normal"),
          "semibold": ("Segoe UI", "bold"), "bold": ("Segoe UI", "bold")}
_MONO = "Consolas"


def _load_fonts(base_dir: str) -> None:
    """Register the bundled Barlow / Martian Mono files; keep the Segoe UI / Consolas fallback on failure."""
    global _MONO
    d = os.path.join(base_dir, "assets", "fonts")

    def _load(name):
        try:
            return bool(ctk.FontManager.load_font(os.path.join(d, name)))
        except Exception:
            return False

    ui = [_load(n) for n in ("Barlow-Regular.ttf", "Barlow-Medium.ttf",
                             "Barlow-SemiBold.ttf", "Barlow-Bold.ttf")]
    if all(ui):
        _FACES.update(normal=("Barlow", "normal"), medium=("Barlow Medium", "normal"),
                      semibold=("Barlow SemiBold", "normal"), bold=("Barlow", "bold"))
    if _load("MartianMono.ttf"):
        _MONO = "Martian Mono"


def _F(sz, wt="normal"):
    fam, weight = _FACES.get(wt, _FACES["normal"])
    return ctk.CTkFont(family=fam, size=sz, weight=weight)


def _FM(sz=10):
    return ctk.CTkFont(family=_MONO, size=sz)


def _tkfont(sz, wt="normal", s=1.0):
    """Font tuple for raw tk canvases (negative size = pixels, like CTkFont)."""
    fam, weight = _FACES.get(wt, _FACES["normal"])
    return (fam, -max(1, round(sz * s)), weight)


# ── Detection helpers ─────────────────────────────────────────────────────────

def _has_nvidia() -> bool:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=6)
        return r.returncode == 0
    except Exception:
        return False


def torch_index_for_cuda(major: int, minor: int) -> str | None:
    """
    Map a driver-reported CUDA version to a PyTorch wheel index that carries
    torch 2.8 wheels (cu126 / cu128).  Older drivers return None (CPU build).
    """
    version = major * 10 + minor  # e.g. 12.8 -> 128
    if version >= 128:
        return "https://download.pytorch.org/whl/cu128"
    if version >= 126:
        return "https://download.pytorch.org/whl/cu126"
    return None


def _cuda_torch_index() -> str | None:
    """
    Return the PyTorch wheel index URL for the detected CUDA driver, or None
    if no NVIDIA GPU is present or the driver is too old.
    """
    try:
        import re
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=6)
        if r.returncode != 0:
            return None
        # "CUDA Version: 12.8" on older drivers, "CUDA UMD Version: 13.3" from driver 610 on
        m = re.search(r"CUDA(?:\s+(?:UMD|Driver))?\s+Version:\s*(\d+)\.(\d+)", r.stdout)
        if not m:
            return None
        return torch_index_for_cuda(int(m.group(1)), int(m.group(2)))
    except Exception:
        return None


_COMPAT_MINORS = (10, 11, 12, 13)   # Python 3.x versions with full wheel support


def _py_version_ok(version_output: str) -> bool:
    """Return True if version string is a compatible Python (3.10–3.13)."""
    import re
    m = re.search(r"Python 3\.(\d+)", version_output)
    return bool(m and int(m.group(1)) in _COMPAT_MINORS)


def _find_python() -> str | None:
    """Return path to a compatible system Python (3.10–3.13), or None.

    Python 3.14+ is excluded — ctranslate2/faster-whisper/whisperx do not
    yet publish wheels for it.  If the system only has 3.14+, the wizard
    will auto-download Python 3.13 instead.
    """
    # Check specific version commands first (3.13 preferred), then generic
    candidates = ["python3.13", "python3.12", "python3.11", "python3.10",
                  "python3", "python"]
    for cmd in candidates:
        path = shutil.which(cmd)
        if not path:
            continue
        try:
            r = subprocess.run([path, "--version"], capture_output=True,
                               text=True, timeout=5)
            if _py_version_ok(r.stdout + r.stderr):
                return path
        except Exception:
            continue
    return None


def _find_python_fresh() -> str | None:
    """Like _find_python() but also searches well-known install locations.

    shutil.which() reads os.environ['PATH'] which is a snapshot from process
    start — it won't see Python installed after this process launched.  This
    function falls back to globbing the standard Windows install directories so
    'Check again' works without requiring the user to restart the app.
    """
    result = _find_python()
    if result:
        return result

    import glob
    local_app   = os.environ.get("LOCALAPPDATA", "")
    prog_files  = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    patterns = [
        os.path.join(local_app,  "Programs", "Python", "Python3*", "python.exe"),
        os.path.join(prog_files, "Python3*", "python.exe"),
        r"C:\Python3*\python.exe",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))

    def _sort_key(p: str):
        import re
        m = re.search(r"Python3(\d+)", p, re.IGNORECASE)
        minor = int(m.group(1)) if m else 0
        # Only accept compatible minors; prefer 3.13 → 3.12 → 3.11 → 3.10
        return (0 if minor in _COMPAT_MINORS else 1, -minor)

    for path in sorted(candidates, key=_sort_key):
        try:
            r = subprocess.run(
                [path, "--version"], capture_output=True, text=True,
                timeout=5, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if _py_version_ok(r.stdout + r.stderr):
                return path
        except Exception:
            continue
    return None


_PIP_UNITS = {
    "B": 1, "kB": 1000, "KB": 1000, "MB": 1000 ** 2, "GB": 1000 ** 3,
    "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3,
}
_PIP_UNIT_RE = r"(KiB|MiB|GiB|kB|KB|MB|GB|B)(?![A-Za-z])"
_PIP_PROGRESS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*" + _PIP_UNIT_RE
    + r"(?:\s+(\d+(?:\.\d+)?)\s*" + _PIP_UNIT_RE + r"/s)?"
)


def parse_pip_progress(line: str) -> tuple[int, int, float | None] | None:
    """(done_bytes, total_bytes, bytes_per_second or None) from one pip progress line, else None."""
    m = _PIP_PROGRESS_RE.search(line)
    if not m:
        return None
    mult = _PIP_UNITS[m.group(3)]
    done = int(round(float(m.group(1)) * mult))
    total = int(round(float(m.group(2)) * mult))
    if total <= 0:
        return None
    speed = None
    if m.group(4):
        speed = float(m.group(4)) * _PIP_UNITS[m.group(5)]
    return done, total, speed


def human_bytes(n: float) -> str:
    """950 -> '950 B', 1_200_000 -> '1.2 MB', 2_500_000_000 -> '2.5 GB'."""
    if n < 1000:
        return f"{int(n)} B"
    for unit, size in (("kB", 1000), ("MB", 1000 ** 2), ("GB", 1000 ** 3)):
        if round(n / size, 1) < 1000 or unit == "GB":
            return f"{n / size:.1f} {unit}"


def human_speed(bps: float) -> str:
    """8_100_000 -> '8.1 MB/s'."""
    return human_bytes(bps) + "/s"


def _scale_of(widget) -> float:
    """customtkinter's DPI scale for the widget's window (raw tk canvases are not scaled for us)."""
    try:
        return float(ctk.ScalingTracker.get_window_scaling(widget.winfo_toplevel()))
    except Exception:
        return 1.0


def _mix(a: str, b: str, t: float) -> str:
    """Blend hex colour a toward b by fraction t."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(ca, cb))


def _logo_mark(parent, sc: float) -> tk.Canvas:
    """Three rounded bars (heat) then three short lines (violet, pink, ink)."""
    s = lambda v: v * sc
    cv = tk.Canvas(parent, width=round(s(36)), height=round(s(22)), bg=GROUND,
                   highlightthickness=0, bd=0)
    for i, h in enumerate((10, 20, 14)):
        x, mid = 2 + i * 7, 11
        cv.create_line(s(x), s(mid - h / 2 + 2), s(x), s(mid + h / 2 - 2), fill=HEAT,
                       width=round(s(4)), capstyle="round")
    for i, (ln, col) in enumerate(((12, "#8b6cff"), (9, "#ff3d8a"), (11, INK))):
        y = 4.5 + i * 6
        cv.create_line(s(24 + 1.5), s(y), s(24 + ln - 1.5), s(y), fill=col,
                       width=round(s(3)), capstyle="round")
    return cv


class _Tooltip:
    """Small hover tooltip for tk / customtkinter widgets."""

    def __init__(self, widgets, text: str):
        self._text, self._tip = text, None
        for w in widgets:
            w.bind("<Enter>", self._show, add="+")
            w.bind("<Leave>", self._hide, add="+")

    def _show(self, e):
        if self._tip is not None:
            return
        tip = tk.Toplevel(e.widget)
        tip.wm_overrideredirect(True)
        tip.configure(bg=RULE)
        tk.Label(tip, text=self._text, bg=RAISED, fg=INK_2, font=_tkfont(12),
                 wraplength=360, justify="left", padx=10, pady=6).pack(padx=1, pady=1)
        tip.wm_geometry(f"+{e.x_root + 12}+{e.y_root + 16}")
        self._tip = tip

    def _hide(self, _e=None):
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


_PIP_DETAIL_RE = re.compile(r"^(Collecting|Downloading|Using cached)\s+(\S+)(?:\s+\(([^)]+)\))?")


def parse_pip_detail(line: str) -> str | None:
    """'torch 2.8.0+cu128 · 2.5 GB' from a pip 'Collecting' / 'Downloading' line, else None."""
    m = _PIP_DETAIL_RE.match(line.strip())
    if not m:
        return None
    verb, target, size = m.groups()
    if target.endswith(".metadata"):
        return None
    if verb == "Collecting":
        return target
    fname = urllib.parse.unquote(target.rsplit("/", 1)[-1])
    parts = re.sub(r"\.(?:whl|tar\.gz|zip)$", "", fname).split("-")
    text = f"{parts[0]} {parts[1]}" if len(parts) >= 2 and parts[0] and parts[1] else fname
    return f"{text} · {size}" if size else text


class _GlideBar(ctk.CTkProgressBar):
    """Progress bar that glides to each new determinate value instead of jumping."""
    _GLIDE = 0.25   # seconds

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._glide_job = None
        super().set(0)      # customtkinter starts a bar half full

    def _glide_cancel(self):
        if self._glide_job is not None:
            try:
                self.after_cancel(self._glide_job)
            except Exception:
                pass
            self._glide_job = None

    def set(self, value, from_variable_callback=False):
        self._glide_cancel()
        target = max(0.0, min(1.0, value))
        start, t0 = self.get(), time.monotonic()
        if self._mode != "determinate" or start == target:
            super().set(target, from_variable_callback)
            return

        def _step():
            self._glide_job = None
            try:
                p = min(1.0, (time.monotonic() - t0) / self._GLIDE)
                super(_GlideBar, self).set(start + (target - start) * (1 - (1 - p) ** 3))
                if p < 1.0:
                    self._glide_job = self.after(16, _step)
            except tk.TclError:
                pass
        _step()

    def configure(self, require_redraw=False, **kwargs):
        if kwargs.get("mode") == "determinate":
            self.stop()
        super().configure(require_redraw=require_redraw, **kwargs)

    def destroy(self):
        self._glide_cancel()
        super().destroy()


def _pill(parent, text: str, command) -> ctk.CTkButton:
    """The one orange primary button."""
    return ctk.CTkButton(parent, text=text, width=160, height=44, corner_radius=22,
                         font=_F(15, "bold"), fg_color=HEAT, hover_color=HEAT_HOV,
                         text_color=ON_HEAT, command=command)


def _link(parent, text: str, command) -> ctk.CTkLabel:
    """Quiet text link: ink-2, ink on hover, no background."""
    lbl = ctk.CTkLabel(parent, text=text, font=_F(13), text_color=INK_2, width=0)
    lbl.bind("<Enter>", lambda _e: lbl.configure(text_color=INK), add="+")
    lbl.bind("<Leave>", lambda _e: lbl.configure(text_color=INK_2), add="+")
    lbl.bind("<Button-1>", lambda _e: command(), add="+")
    try:
        lbl._label.configure(cursor="hand2")
    except Exception:
        pass
    return lbl


STEPS = ("Python", "Speech engine", "Finishing")
_STATES = ("pending", "active", "done", "failed")
_FADE = 0.2          # seconds for a state change to cross-fade
_PULSE = 1.2         # seconds per pulse of the active node

# Target colour of every part of a node, per state
_LOOK = {
    "pending": dict(ring=INERT, fill=GROUND, num=INK_3, dot=GROUND, check=GROUND, bang=GROUND, label=INK_2, conn=RULE),
    "active":  dict(ring=HEAT, fill=GROUND, num=GROUND, dot=HEAT, check=GROUND, bang=GROUND, label=INK, conn=RULE),
    "done":    dict(ring=HEAT, fill=HEAT, num=HEAT, dot=HEAT, check=ON_HEAT, bang=HEAT, label=INK, conn=HEAT),
    "failed":  dict(ring=BAD, fill=GROUND, num=GROUND, dot=GROUND, check=GROUND, bang=BAD, label=INK, conn=RULE),
}


class StepTracker(tk.Canvas):
    """Numbered step nodes joined by connectors; state changes cross-fade, the active node pulses."""

    def __init__(self, parent, steps=STEPS, width=520):
        self._sc = _scale_of(parent)
        self._span = width
        self._steps = tuple(steps)
        super().__init__(parent, width=round(width * self._sc), height=round(74 * self._sc),
                         bg=GROUND, highlightthickness=0, bd=0)
        n = len(self._steps)
        self.states = ["pending"] * n
        self._details = [""] * n
        self._from = [dict(_LOOK["pending"]) for _ in range(n)]
        self._to = [dict(_LOOK["pending"]) for _ in range(n)]
        self._t0 = [float("-inf")] * n
        self._job = None
        self._draw()

    def set_state(self, index: int, state: str) -> None:
        if state not in _STATES:
            raise ValueError(f"invalid step state: {state!r}")
        now = time.monotonic()
        self._from[index] = self._current(index, now)
        self._to[index] = dict(_LOOK[state])
        self._t0[index] = now
        self.states[index] = state
        self._schedule()

    def set_detail(self, index: int, text: str) -> None:
        self._details[index] = text
        self._draw()

    def destroy(self):
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except Exception:
                pass
            self._job = None
        super().destroy()

    def _current(self, i: int, now: float) -> dict:
        p = min(1.0, max(0.0, (now - self._t0[i]) / _FADE))
        p = p * p * (3 - 2 * p)
        return {k: _mix(self._from[i][k], self._to[i][k], p) for k in self._to[i]}

    def _schedule(self):
        if self._job is None:
            self._job = self.after(16, self._tick)

    def _tick(self):
        self._job = None
        try:
            if not self.winfo_exists():
                return
            self._draw()
        except tk.TclError:
            return
        now = time.monotonic()
        if "active" in self.states or any(now - t < _FADE for t in self._t0):
            self._schedule()

    def _draw(self):
        self.delete("all")
        sc, now = self._sc, time.monotonic()
        s = lambda v: v * sc
        n = len(self._steps)
        gap = self._span / n
        cy, rad = 15, 12
        pulse = 0.5 - 0.5 * math.cos(2 * math.pi * (now % _PULSE) / _PULSE)
        for i, name in enumerate(self._steps):
            cx, c = gap * (i + 0.5), self._current(i, now)
            if i < n - 1:
                self.create_line(s(cx + rad + 9), s(cy), s(gap * (i + 1.5) - rad - 9), s(cy),
                                 fill=c["conn"], width=max(1, round(s(2))))
            self.create_oval(s(cx - rad), s(cy - rad), s(cx + rad), s(cy + rad),
                             outline=c["ring"], fill=c["fill"], width=max(1, round(s(2))))
            if c["num"] != c["fill"]:
                self.create_text(s(cx), s(cy), text=str(i + 1), fill=c["num"],
                                 font=_tkfont(12, "semibold", sc))
            if self.states[i] == "active" or c["dot"] != c["fill"]:
                r = 4.2 + 1.6 * pulse if self.states[i] == "active" else 4.2
                self.create_oval(s(cx - r), s(cy - r), s(cx + r), s(cy + r), outline="",
                                 fill=_mix(c["fill"], c["dot"], 0.55 + 0.45 * pulse))
            if c["check"] != c["fill"]:
                self.create_line(s(cx - 5), s(cy + 0.5), s(cx - 1.5), s(cy + 4), s(cx + 5), s(cy - 3.5),
                                 fill=c["check"], width=round(s(2.5)), capstyle="round",
                                 joinstyle="round")
            if c["bang"] != c["fill"]:
                self.create_text(s(cx), s(cy), text="!", fill=c["bang"], font=_tkfont(14, "bold", sc))
            self.create_text(s(cx), s(36), text=name, anchor="n", fill=c["label"],
                             font=_tkfont(13, "semibold", sc))
            self.create_text(s(cx), s(56), text=self._details[i], anchor="n", fill=INK_3,
                             font=(_MONO, -max(1, round(s(10)))))


def _python_minor(path: str) -> str | None:
    """'3.11' for the Python at path, or None if it can't be queried."""
    try:
        r = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        m = re.search(r"Python (3\.\d+)", r.stdout + r.stderr)
        return m.group(1) if m else None
    except Exception:
        return None


def _cuda_label(torch_index: str) -> str:
    """Wheel index URL ending in cu128 -> 'CUDA 12.8'."""
    digits = re.sub(r"\D", "", torch_index.rstrip("/").split("/")[-1])
    return f"CUDA {digits[:-1]}.{digits[-1]}"


# ── Wizard ────────────────────────────────────────────────────────────────────

class SetupWizard:
    def __init__(self, base_dir: str, on_complete):
        self.base_dir    = base_dir
        self.on_complete = on_complete
        self.runtime_dir = os.path.join(base_dir, "runtime")
        self.has_gpu     = _has_nvidia()
        self.torch_index = _cuda_torch_index()
        self._gpu_warning = ""
        self.python_path = _find_python()
        self._q: queue.Queue[tuple[str, str]] = queue.Queue()

        _load_fonts(base_dir)
        self.win = ctk.CTk(fg_color=GROUND)
        self.win.title("WhisperX Transcriber — Setup")
        self.win.geometry("720x540")
        self.win.resizable(False, False)
        icon = os.path.join(base_dir, "assets", "icon.ico")
        if os.path.isfile(icon):
            try:
                self.win.iconbitmap(icon)
            except Exception:
                pass
        self.win.grid_columnconfigure(0, weight=1)
        self.win.grid_rowconfigure(0, weight=1)

        self._build_shell()
        self._show_welcome()
        self.win.after(100, self._drain_log)

    # ── Shell (header + title + step tracker + swappable content) ─────────────

    def _build_shell(self):
        root = ctk.CTkFrame(self.win, fg_color=GROUND, corner_radius=0)
        root.grid(row=0, column=0, sticky="nsew")
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(2, weight=1)

        # Header
        hdr = ctk.CTkFrame(root, fg_color=GROUND, corner_radius=0, height=64)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.grid_rowconfigure(0, weight=1)
        _logo_mark(hdr, _scale_of(self.win)).grid(row=0, column=0, padx=(28, 10))
        ctk.CTkLabel(hdr, text="WhisperX", font=_F(16, "semibold"),
                     text_color=INK).grid(row=0, column=1, sticky="w")
        tk.Frame(root, bg=RULE, height=1).grid(row=1, column=0, sticky="ew")

        # Centre column: title, tracker, content
        col = ctk.CTkFrame(root, fg_color="transparent", width=520, height=400)
        col.grid(row=2, column=0, sticky="ns")
        col.grid_propagate(False)
        col.grid_columnconfigure(0, weight=1)
        col.grid_rowconfigure(2, weight=1)

        self._title = ctk.CTkLabel(col, text="Setting up WhisperX", font=_F(22, "semibold"),
                                   text_color=INK)
        self._title.grid(row=0, column=0, pady=(28, 0))
        self._tracker = StepTracker(col)
        self._tracker.grid(row=1, column=0, pady=(18, 0))
        self._step_states = list(self._tracker.states)

        # Content (swappable) and its action row
        self.body = ctk.CTkFrame(col, fg_color="transparent")
        self.body.grid(row=2, column=0, sticky="nsew", pady=(22, 0))
        self.body.grid_columnconfigure(0, weight=1)
        self.ftr = ctk.CTkFrame(col, fg_color="transparent", width=1, height=1)
        self.ftr.grid(row=3, column=0, sticky="ew", pady=(0, 16))
        self.ftr.grid_columnconfigure(0, weight=1)

    def _set_step(self, index: int, state: str) -> None:
        """Record a step's state and redraw the tracker (if there is one)."""
        if state not in _STATES:
            raise ValueError(f"invalid step state: {state!r}")
        states = getattr(self, "_step_states", None)
        if states is None:
            states = self._step_states = ["pending"] * len(STEPS)
        states[index] = state
        tracker = getattr(self, "_tracker", None)
        if tracker is not None:
            tracker.set_state(index, state)

    def _clear(self, frame):
        for w in frame.winfo_children():
            w.destroy()

    # ── Page 1: Welcome ───────────────────────────────────────────────────────

    def _show_welcome(self):
        self._clear(self.body)
        self._clear(self.ftr)

        gpu_build = bool(self.has_gpu and self.torch_index)
        py_ver = _python_minor(self.python_path) if self.python_path else None
        tip = None
        if gpu_build:
            gpu_text, gpu_dot = f"NVIDIA · {_cuda_label(self.torch_index)} build", OK
        elif self.has_gpu:
            gpu_text, gpu_dot = "Driver too old · CPU build", WARN
            tip = ("NVIDIA driver too old for GPU acceleration (needs CUDA 12.6+, "
                   "driver 560 or newer) — installing CPU build. "
                   "Update your driver and delete the runtime folder to retry.")
        else:
            gpu_text, gpu_dot = "None · CPU build", INERT
        if self.python_path:
            py_text, py_dot = (f"Found {py_ver}" if py_ver else "Found"), OK
        else:
            py_text, py_dot = "Will be installed", WARN
        rows = [("GPU", gpu_text, gpu_dot, tip),
                ("Python", py_text, py_dot, None),
                ("Download", "2–3 GB, once" if gpu_build else "about 1 GB, once", None, None)]

        for i in range(len(STEPS)):
            self._set_step(i, "pending")
        for i, text in enumerate((py_ver or ("" if self.python_path else "3.13"),
                                  _cuda_label(self.torch_index) if gpu_build else "CPU",
                                  "Speakers · ffmpeg")):
            self._tracker.set_detail(i, text)
        if not self.python_path:
            self._set_step(0, "active")

        self._set_title("Setting up WhisperX")
        card, inner = self._make_card()
        r = self._add_rows(inner, rows)

        if self.python_path:
            _pill(inner, "Install", self._start_install).grid(row=r, column=0, sticky="e", pady=(16, 0))
        else:
            # Auto-start after a brief pause so the user can read the screen
            self.win.after(2000, self._auto_install_python)

    def _make_card(self):
        """The panel card (1 px rule, radius 14, padding 20); returns (card, inner)."""
        card = ctk.CTkFrame(self.body, fg_color=PANEL, border_width=1, border_color=RULE,
                            corner_radius=14)
        card.grid(row=0, column=0, sticky="ew")
        card.grid_columnconfigure(0, weight=1)
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        inner.grid_columnconfigure(0, weight=1)
        return card, inner

    def _add_rows(self, inner, rows) -> int:
        """Key / value rows with an optional dot and tooltip; returns the next free grid row."""
        r = 0
        for n, (key, value, dot, tooltip) in enumerate(rows):
            if n:
                tk.Frame(inner, bg=RULE, height=1).grid(row=r, column=0, sticky="ew")
                r += 1
            row = ctk.CTkFrame(inner, fg_color="transparent")
            row.grid(row=r, column=0, sticky="ew")
            row.grid_columnconfigure(1, weight=1)
            r += 1
            ctk.CTkLabel(row, text=key, font=_F(13), text_color=INK_3
                         ).grid(row=0, column=0, sticky="w", pady=10)
            val = ctk.CTkFrame(row, fg_color="transparent")
            val.grid(row=0, column=1, sticky="e")
            hover = []
            if dot:
                d = ctk.CTkFrame(val, width=8, height=8, corner_radius=4, fg_color=dot)
                d.grid(row=0, column=0, padx=(0, 8))
                hover.append(d)
            lbl = ctk.CTkLabel(val, text=value, font=_F(13), text_color=INK)
            lbl.grid(row=0, column=1)
            hover.append(lbl)
            if tooltip:
                _Tooltip(hover, tooltip)
        return r

    def _set_title(self, text: str) -> None:
        if hasattr(self, "_title"):
            self._title.configure(text=text)

    def _make_log_box(self, parent, row: int, lines=()):
        """Read-only log panel, hidden until _toggle_details."""
        box = ctk.CTkTextbox(parent, height=100, font=_FM(10), fg_color=WELL, text_color=INK_2,
                             corner_radius=8, state="disabled")
        box.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        box.grid_remove()
        self._log_box = box
        self._details_open = False
        for line in lines:
            self._append_log(line)
        return box

    def _append_log(self, text: str) -> None:
        box = getattr(self, "_log_box", None)
        if box is None:
            return
        try:
            box.configure(state="normal")
            box.insert("end", text + "\n")
            box.see("end")
            box.configure(state="disabled")
        except tk.TclError:
            pass

    def _toggle_details(self) -> None:
        self._details_open = not getattr(self, "_details_open", False)
        box = getattr(self, "_log_box", None)
        if box is not None:
            for w in getattr(self, "_hide_on_details", []):   # the log takes their place
                w.grid_remove() if self._details_open else w.grid()
            if self._details_open:
                box.grid()
                box.see("end")
            else:
                box.grid_remove()
        link = getattr(self, "_details_link", None)
        if link is not None:
            link.configure(text="Hide details" if self._details_open else "Show details →")

    def _drop_install_widgets(self) -> None:
        """Stop the bar and forget the install page's widgets before another page replaces them."""
        try:
            self._prog.stop()
        except Exception:
            pass
        for name in ("_prog", "_status_var", "_activity_lbl", "_pct_lbl", "_speed_lbl",
                     "_log_box", "_details_link", "_hide_on_details"):
            self.__dict__.pop(name, None)

    def _progress_card(self, status: str, details: bool) -> None:
        """One card: status, package line, bar, size / speed row and (optionally) the details log."""
        self._drop_install_widgets()
        self._clear(self.body)
        self._set_title("Setting up WhisperX")
        card, inner = self._make_card()

        self._status_var = ctk.StringVar(value=status)
        ctk.CTkLabel(inner, textvariable=self._status_var, font=_F(15, "semibold"),
                     text_color=INK, width=0).grid(row=0, column=0, sticky="w")
        self._activity_lbl = ctk.CTkLabel(inner, text="", font=_FM(11), text_color=INK_3,
                                          width=0, anchor="w")
        self._activity_lbl.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self._prog = _GlideBar(inner, height=6, corner_radius=3, fg_color=WELL,
                               progress_color=HEAT, mode="determinate")
        self._prog.set(0)
        self._prog.grid(row=2, column=0, sticky="ew", pady=(12, 0))

        sizes = ctk.CTkFrame(inner, fg_color="transparent")
        sizes.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        sizes.grid_columnconfigure(0, weight=1)
        self._pct_lbl = ctk.CTkLabel(sizes, text="", font=_FM(11), text_color=INK_2,
                                     width=0, anchor="w")
        self._pct_lbl.grid(row=0, column=0, sticky="w")
        self._speed_lbl = ctk.CTkLabel(sizes, text="", font=_FM(11), text_color=EMBER,
                                       width=0, anchor="e")
        self._speed_lbl.grid(row=0, column=1, sticky="e")

        self._details_open = False
        self._hide_on_details = [self._prog, sizes]
        if details:
            self._details_link = _link(inner, "Show details →", self._toggle_details)
            self._details_link.grid(row=4, column=0, sticky="w", pady=(12, 0))
            self._make_log_box(inner, row=5)

    def _check_python_again(self):
        self.python_path = _find_python_fresh()
        self._show_welcome()

    # ── Auto Python install ───────────────────────────────────────────────────

    def _auto_install_python(self):
        self._show_python_installing()
        threading.Thread(target=self._run_python_install, daemon=True).start()

    def _show_python_installing(self):
        self._progress_card(f"Downloading Python {_PYTHON_VERSION}…", details=False)
        self._set_step(0, "active")

    def _set_py_status(self, msg: str):
        self._set_status(msg)

    def _set_py_progress(self, value: float):
        def _do():
            if hasattr(self, "_prog"):
                self._prog.set(max(0.0, min(1.0, value)))
        self.win.after(0, _do)

    def _run_python_install(self):
        import time
        try:
            self.win.after(0, lambda: self._set_step(0, "active"))
            tmp = os.path.join(tempfile.gettempdir(), f"python-{_PYTHON_VERSION}-installer.exe")

            # Download
            def _progress(count, block_size, total):
                if total > 0:
                    self._set_py_progress(min(count * block_size / total, 0.95))

            urllib.request.urlretrieve(_PYTHON_URL, tmp, _progress)
            self._set_py_progress(1.0)

            # Run installer — InstallAllUsers=0 keeps it per-user (no UAC needed)
            self._set_py_status("Installing Python…")
            result = subprocess.run(
                [tmp, "/passive", "PrependPath=1", "Include_test=0",
                 "SimpleInstall=1", "InstallAllUsers=0"],
                timeout=300,
            )
            # 1638 = already installed (also fine)
            if result.returncode not in (0, 1638):
                raise RuntimeError(f"Installer exited with code {result.returncode}.")

            # Retry a few times — installer may not have flushed to disk yet
            self._set_py_status("Checking Python…")
            for _ in range(6):
                self.python_path = _find_python_fresh()
                if self.python_path:
                    break
                time.sleep(2)

            if self.python_path:
                self._set_py_status("Python installed")
                self.win.after(0, lambda: self._set_step(0, "done"))
                self.win.after(800, self._show_welcome)
            else:
                raise RuntimeError("Python was installed — please restart the app to continue.")

        except Exception as exc:
            self._set_py_status(f"Failed: {exc}")
            self.win.after(0, lambda msg=str(exc): self._show_py_error(msg))   # exc is unbound once except ends

    def _show_py_error(self, msg: str):
        self._set_step(0, "failed")
        self._failure_page(msg, self._auto_install_python, "Install Python manually →",
                           lambda: webbrowser.open("https://www.python.org/downloads/"), log=False)

    # ── Page 2: Installing ────────────────────────────────────────────────────

    def _show_installing(self):
        self._log_lines = []
        self._progress_card("Preparing…", details=True)

    def _show_detail(self, text: str) -> None:
        """Show the package pip is fetching ('torch 2.8.0+cu128 · 2.5 GB') in the card."""
        detail = parse_pip_detail(text)
        if detail is None or not hasattr(self, "_activity_lbl"):
            return
        try:
            self._activity_lbl.configure(text=detail if len(detail) <= 56 else detail[:55] + "…")
        except tk.TclError:
            pass

    def _drain_log(self):
        while True:
            try:
                kind, text = self._q.get_nowait()
            except queue.Empty:
                break
            if kind == "r":
                self._show_detail(text)
            elif kind == "p":
                self.win.after(0, lambda v=text: self._set_install_progress(v))
            elif kind == "i":
                self.win.after(0, self._set_install_indeterminate)
            elif text:
                if "━" in text or "─" in text:
                    continue
                if not hasattr(self, "_log_lines"):
                    self._log_lines = []
                self._log_lines.append(text)
                self._append_log(text)
                self._show_detail(text)
        self.win.after(100, self._drain_log)

    def _log(self, msg: str):
        self._q.put(("n", msg))

    def _set_status(self, msg: str):
        def _do():
            if hasattr(self, "_status_var"):
                self._status_var.set(msg)
            if hasattr(self, "_activity_lbl"):
                self._activity_lbl.configure(text="")
        self.win.after(0, _do)

    def _set_progress(self, value: float, step: str = ""):
        def _do():
            if hasattr(self, "_prog"):
                self._prog.set(max(0.0, min(1.0, value)))
        self.win.after(0, _do)

    def _animate_progress(self, current: float, target: float):
        """Drift the bar toward target until real pip progress takes over."""
        if getattr(self, "_install_done", False) or \
                getattr(self, "_real_progress_seen", False):
            return
        next_val = min(current + 0.005, target)
        self._set_progress(next_val)
        if next_val < target:
            self.win.after(1000, lambda: self._animate_progress(next_val, target))
        else:
            # Ceiling reached — switch to indeterminate so it's clear we're still working
            if hasattr(self, "_prog"):
                self.win.after(0, lambda: self._prog.configure(mode="indeterminate"))
                self.win.after(0, self._prog.start)

    # ── Install logic ─────────────────────────────────────────────────────────

    def _start_install(self):
        self._show_installing()
        threading.Thread(target=self._run_install, daemon=True).start()

    def _pkg_installed(self, venv_pip: str, pkg: str) -> bool:
        r = subprocess.run(
            [venv_pip, "show", pkg],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return r.returncode == 0

    def _check_gpu(self, venv_py: str) -> None:
        """Verify torch can see the GPU in the runtime venv; record a warning if not."""
        self._set_status("Verifying GPU support...")
        try:
            r = subprocess.run(
                [venv_py, "-c",
                 "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"],
                capture_output=True, timeout=120,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            ok = r.returncode == 0
        except Exception:
            ok = False
        if ok:
            self._log("\n[GPU] CUDA is available to PyTorch.")
        else:
            self._gpu_warning = ("Warning: GPU not usable by PyTorch — "
                                 "transcription will run on CPU.")
            self._log(f"\n{self._gpu_warning}")

    def _run_pip(self, cmd: list[str]) -> int:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        buf = ""
        while True:
            chunk = proc.stdout.read(128)
            if not chunk:
                break
            for ch in chunk.decode("utf-8", errors="replace"):
                if ch == "\r":
                    if buf.strip():
                        self._q.put(("r", buf))
                        self._maybe_emit_progress(buf)
                    buf = ""
                elif ch == "\n":
                    if buf.strip():
                        self._q.put(("n", buf.rstrip()))
                        self._maybe_emit_progress(buf)
                        self._check_install_phase(buf)
                    buf = ""
                else:
                    buf += ch
        if buf.strip():
            self._q.put(("n", buf.rstrip()))
        proc.wait()
        return proc.returncode

    def _maybe_emit_progress(self, line: str):
        """Parse pip's 'X/Y MB speed' download lines into (done, total, speed)."""
        result = parse_pip_progress(line)
        if result is None:
            return
        self._real_progress_seen = True
        self._q.put(("p", result))

    def _set_install_progress(self, value):
        done, total, _speed = value
        frac = max(0.0, min(1.0, done / total)) if total > 0 else 0.0
        if hasattr(self, "_prog"):
            self._prog.configure(mode="determinate")
            self._prog.set(frac)
        if hasattr(self, "_pct_lbl"):
            self._pct_lbl.configure(
                text=f"{human_bytes(done)} / {human_bytes(total)}   {int(frac * 100 + 1e-6)} %")
        if hasattr(self, "_speed_lbl"):
            self._speed_lbl.configure(text=f"↓ {human_speed(_speed)}" if _speed is not None else "")

    def _check_install_phase(self, line: str):
        """Downloads are done once pip starts installing — switch to an
        indeterminate bar so it doesn't sit frozen at a download percentage."""
        if "Installing collected packages" in line or "Building wheel" in line:
            self._q.put(("i", ""))

    def _set_install_indeterminate(self):
        if hasattr(self, "_prog"):
            self._prog.configure(mode="indeterminate")
            self._prog.start()
        for name in ("_pct_lbl", "_speed_lbl"):
            if hasattr(self, name):
                getattr(self, name).configure(text="")
        if hasattr(self, "_status_var"):
            self._status_var.set("Installing packages…")

    def _run_install(self):
        self._gpu_warning = ""
        try:
            self._real_progress_seen = False
            self.win.after(0, lambda: self._set_step(0, "done"))
            self.win.after(0, lambda: self._set_step(1, "active"))
            py      = self.python_path
            runtime = self.runtime_dir
            venv_py  = os.path.join(runtime, "Scripts", "python.exe")
            venv_pip = os.path.join(runtime, "Scripts", "pip.exe")

            # 1. Create venv — recreate if it exists but was built with an
            #    incompatible Python version (e.g. 3.14 has no ctranslate2 wheel)
            def _venv_minor() -> int:
                try:
                    r = subprocess.run(
                        [venv_py, "-c", "import sys; print(sys.version_info.minor)"],
                        capture_output=True, text=True, timeout=5,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                    return int(r.stdout.strip())
                except Exception:
                    return 0

            needs_create = not os.path.isfile(venv_py)
            if not needs_create and _venv_minor() >= 14:
                self._log("Existing venv uses Python 3.14+ — recreating with compatible version.\n")
                shutil.rmtree(runtime, ignore_errors=True)
                needs_create = True

            if needs_create:
                self._set_status("Creating virtual environment...")
                self._log(f"Creating venv at: {runtime}\n")
                r = subprocess.run([py, "-m", "venv", runtime],
                                   capture_output=True, text=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                if r.returncode != 0:
                    raise RuntimeError(f"venv creation failed:\n{r.stderr}")
            else:
                self._log(f"Venv already exists, reusing: {runtime}\n")
            self._set_progress(0.05)

            # 2. Upgrade pip
            self._set_status("Upgrading pip...")
            self._log("\n[pip] Upgrading...")
            self._run_pip([venv_py, "-m", "pip", "install",
                           "--upgrade", "pip", "--quiet", "--timeout", "120"])
            self._set_progress(0.10)

            # 3. Install all packages in one pip call.
            #    GPU path: pytorch.org is the primary index so torch/torchaudio
            #    resolve to the CUDA build; PyPI is the fallback for everything
            #    else (whisperx, customtkinter, ...).  One torch download, no swap.
            #    CPU path: plain PyPI install.
            #
            #    Torch pin: torch~=2.8.0 matches whisperx and stops pip picking a
            #    newer CPU-only build from PyPI when the CUDA index lacks it.
            TORCH_MIN = "2.8.0"

            if self._pkg_installed(venv_pip, "whisperx"):
                self._log("\n[whisperx] Already installed — skipping.")
                self.win.after(0, lambda: self._set_step(1, "done"))
                self.win.after(0, lambda: self._set_step(2, "active"))
                # Existing installs (v1.3 and older) lack drag-and-drop support.
                if not self._pkg_installed(venv_pip, "tkinterdnd2"):
                    self._log("\n[tkinterdnd2] Installing drag-and-drop support...")
                    if self._run_pip([venv_pip, "install", "tkinterdnd2>=0.4.2",
                                      "--timeout", "120"]) != 0:
                        self._log("[tkinterdnd2] Warning: install failed, "
                                  "drag-and-drop disabled (use Browse instead).")
                if not self._pkg_installed(venv_pip, "sherpa-onnx"):
                    self._log("\n[sherpa-onnx] Installing speaker identification...")
                    if self._run_pip([venv_pip, "install", "sherpa-onnx>=1.13",
                                      "--timeout", "120"]) != 0:
                        self._log("[sherpa-onnx] Warning: install failed, "
                                  "speaker identification disabled.")
                if not (self._pkg_installed(venv_pip, "python-bidi")
                        and self._pkg_installed(venv_pip, "arabic-reshaper")):
                    self._log("\n[bidi] Installing right-to-left text support...")
                    if self._run_pip([venv_pip, "install", "python-bidi>=0.6",
                                      "arabic-reshaper>=3.0",
                                      "--timeout", "120"]) != 0:
                        self._log("[bidi] Warning: install failed, right-to-left "
                                  "text (Urdu, Arabic) may show in the wrong order.")
                self._set_progress(0.85)
                if _cuda_torch_index():
                    self._check_gpu(venv_py)
            else:
                torch_index = _cuda_torch_index()
                self._install_done = False
                self.win.after(1000, lambda: self._animate_progress(0.10, 0.83))

                if torch_index:
                    self._set_status(
                        "Downloading the speech engine for your GPU (about 2–3 GB)…")
                    self._log(
                        f"\n[Installing] torch~={TORCH_MIN}+CUDA, whisperx, ffmpeg, deps...")
                    cmd = [
                        venv_pip, "install", "--timeout", "120",
                        f"torch~={TORCH_MIN}",
                        f"torchaudio~={TORCH_MIN}",
                        "whisperx",
                        "imageio-ffmpeg",
                        "customtkinter>=5.2.2",
                        "tkinterdnd2>=0.4.2",
                        "sherpa-onnx>=1.13",
                        "python-bidi>=0.6",
                        "arabic-reshaper>=3.0",
                        "Pillow",
                        "--index-url", torch_index,
                        "--extra-index-url", "https://pypi.org/simple/",
                    ]
                else:
                    self._set_status(
                        "Downloading the speech engine (about 1 GB)…")
                    self._log("\n[Installing] whisperx + ffmpeg + customtkinter + Pillow...")
                    cmd = [
                        venv_pip, "install", "--timeout", "120",
                        "whisperx",
                        "imageio-ffmpeg",
                        "customtkinter>=5.2.2",
                        "tkinterdnd2>=0.4.2",
                        "sherpa-onnx>=1.13",
                        "python-bidi>=0.6",
                        "arabic-reshaper>=3.0",
                        "Pillow",
                    ]

                rc = self._run_pip(cmd)
                self._install_done = True
                if hasattr(self, "_prog"):
                    self.win.after(0, self._prog.stop)
                    self.win.after(0, lambda: self._prog.configure(mode="determinate"))
                if rc != 0:
                    raise RuntimeError(
                        "pip couldn't install the speech engine.")
                self._set_progress(0.85)
                self.win.after(0, lambda: self._set_step(1, "done"))
                self.win.after(0, lambda: self._set_step(2, "active"))
                if torch_index:
                    self._check_gpu(venv_py)

            # ffmpeg — checked independently so it installs even when whisperx
            # was already cached (handles upgrades from older app versions)
            if not self._pkg_installed(venv_pip, "imageio-ffmpeg"):
                self._set_status("Installing ffmpeg (audio decoder)...")
                self._log("\n[ffmpeg] Installing portable ffmpeg...")
                rc = self._run_pip([venv_pip, "install", "imageio-ffmpeg",
                                    "--quiet", "--timeout", "120"])
                if rc != 0:
                    raise RuntimeError(
                        "pip couldn't install ffmpeg.")
                self._log("\n[ffmpeg] Done.")
            else:
                self._log("\n[ffmpeg] Already installed — skipping.")
            self._set_progress(0.95)

            # 4. Mark complete
            flag = os.path.join(runtime, ".setup_complete")
            with open(flag, "w") as fh:
                fh.write("ok")

            self._set_progress(1.0)
            self.win.after(0, lambda: self._set_step(2, "done"))
            self._set_status("Setup complete!")
            self._log("\n✓ All done! Launching app...")
            self.win.after(800, self._show_done)

        except Exception as exc:
            import traceback as tb
            self._log(f"\nERROR: {exc}\n{tb.format_exc()}")
            self._set_status("Setup stopped")
            self.win.after(0, lambda msg=str(exc): self._show_error(msg))   # exc is unbound once except ends

    # ── Page 3: Done / Error ──────────────────────────────────────────────────

    def _show_done(self):
        self._drop_install_widgets()
        self._clear(self.body)
        self._set_title("Ready")
        for i in range(len(STEPS)):
            self._set_step(i, "done")
        if self._gpu_warning:
            gpu = ("GPU", "Runs on CPU", WARN, self._gpu_warning)
        elif self.torch_index:
            gpu = ("GPU", "CUDA works", OK, None)
        else:
            gpu = ("GPU", "None · CPU build", INERT, None)
        card, inner = self._make_card()
        r = self._add_rows(inner, [gpu, ("Engine", "Installed", OK, None)])
        _pill(inner, "Open WhisperX", self._finish).grid(row=r, column=0, sticky="e", pady=(16, 0))

    def _show_error(self, msg: str):
        states = getattr(self, "_step_states", [])
        self._set_step(states.index("active") if "active" in states else 1, "failed")
        self._failure_page(msg, self._retry, "Show details →", self._toggle_details, log=True)

    def _failure_page(self, msg: str, retry, link_text: str, link_cmd, log: bool) -> None:
        """Setup stopped: the reason, one orange 'Try again' and a quiet link."""
        self._drop_install_widgets()
        self._clear(self.body)
        self._set_title("Setup stopped")
        card, inner = self._make_card()
        reason = ctk.CTkTextbox(inner, height=84, fg_color=PANEL, border_width=0, corner_radius=0,
                                font=_F(13), text_color=INK, wrap="word")
        reason._textbox.configure(padx=0, pady=0)
        reason.insert("1.0", msg.strip())
        reason.configure(state="disabled")
        reason.bind("<Button-1>", lambda _e: reason.focus_set(), add="+")
        reason.grid(row=0, column=0, sticky="ew")
        self._hide_on_details = [reason]
        if log:
            self._make_log_box(inner, row=1, lines=getattr(self, "_log_lines", []))
        bottom = ctk.CTkFrame(inner, fg_color="transparent")
        bottom.grid(row=2, column=0, sticky="ew", pady=(16, 0))
        bottom.grid_columnconfigure(0, weight=1)
        link = _link(bottom, link_text, link_cmd)
        link.grid(row=0, column=0, sticky="w")
        if log:
            self._details_link = link
        _pill(bottom, "Try again", retry).grid(row=0, column=1, sticky="e")

    def _retry(self):
        self.python_path = _find_python_fresh()
        self._show_welcome()

    def _finish(self):
        self.win.destroy()
        self.on_complete()

    def run(self):
        self.win.mainloop()
