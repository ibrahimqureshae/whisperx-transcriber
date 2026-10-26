"""
WhisperX Transcriber — first-run setup wizard.

Downloads and installs the AI backend (torch + whisperx) into a local
runtime/ venv on first launch. Uses ONLY stdlib + customtkinter + tkinter
so it can run from the thin PyInstaller launcher without any ML packages.
"""
import os
import sys
import queue
import shutil
import subprocess
import tempfile
import threading
import urllib.request
import webbrowser
import customtkinter as ctk

_PYTHON_VERSION   = "3.13.7"
_PYTHON_URL       = f"https://www.python.org/ftp/python/{_PYTHON_VERSION}/python-{_PYTHON_VERSION}-amd64.exe"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


_F  = lambda sz, wt="normal": ctk.CTkFont(family="Segoe UI", size=sz, weight=wt)
_FM = lambda sz=10: ctk.CTkFont(family="Consolas", size=sz)

# Colours (dark, light) tuples
_BG    = ("#f5f5f5", "#161616")
_HDR   = ("#111111", "#0d0d0d")
_CARD  = ("#ffffff", "#1e1e1e")
_FTR   = ("#e8e8e8", "#111111")
_PRI   = ("#111111", "#ffffff")
_DIM   = ("#555555", "#888888")
_HINT  = ("#888888", "#555555")
_LOG   = ("#f0f0f0", "#0d0d0d")
_LTXT  = ("#333333", "#aaaaaa")


# ── Detection helpers ─────────────────────────────────────────────────────────

def _has_nvidia() -> bool:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=6)
        return r.returncode == 0
    except Exception:
        return False


def _cuda_torch_index() -> str | None:
    """
    Return the PyTorch wheel index URL for the detected CUDA driver, or None
    if no NVIDIA GPU is present.  Maps driver CUDA version → nearest PyTorch
    CUDA suffix (driver is always forward-compatible with older toolkits).
    """
    try:
        import re
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=6)
        if r.returncode != 0:
            return None
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", r.stdout)
        if not m:
            return None
        major, minor = int(m.group(1)), int(m.group(2))
        version = major * 10 + minor  # e.g. 12.8 → 128, 12.1 → 121
        if version >= 126:
            suffix = "cu128"
        elif version >= 121:
            suffix = "cu124"
        elif version >= 118:
            suffix = "cu118"
        else:
            return None  # too old, fall back to CPU
        return f"https://download.pytorch.org/whl/{suffix}"
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


# ── Wizard ────────────────────────────────────────────────────────────────────

class SetupWizard:
    def __init__(self, base_dir: str, on_complete):
        self.base_dir    = base_dir
        self.on_complete = on_complete
        self.runtime_dir = os.path.join(base_dir, "runtime")
        self.has_gpu     = _has_nvidia()
        self.torch_index = _cuda_torch_index()
        self.python_path = _find_python()
        self._q: queue.Queue[tuple[str, str]] = queue.Queue()

        self.win = ctk.CTk()
        self.win.title("WhisperX Transcriber — Setup")
        self.win.geometry("620x540")
        self.win.resizable(False, False)
        self.win.grid_columnconfigure(0, weight=1)
        self.win.grid_rowconfigure(0, weight=1)

        self._build_shell()
        self._show_welcome()
        self.win.after(100, self._drain_log)

    # ── Shell (header + swappable content + footer) ───────────────────────────

    def _build_shell(self):
        root = ctk.CTkFrame(self.win, fg_color=_BG, corner_radius=0)
        root.grid(row=0, column=0, sticky="nsew")
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)

        # Header
        hdr = ctk.CTkFrame(root, fg_color=_HDR, corner_radius=0, height=84)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="WhisperX Transcriber", font=_F(20, "bold"),
                     text_color="#ffffff").grid(row=0, column=0, pady=(20, 2))
        ctk.CTkLabel(hdr, text="First-time Setup", font=_F(12),
                     text_color="#666666").grid(row=1, column=0, pady=(0, 16))

        # Content (swappable)
        self.body = ctk.CTkFrame(root, fg_color="transparent")
        self.body.grid(row=1, column=0, sticky="nsew", padx=32, pady=22)
        self.body.grid_columnconfigure(0, weight=1)

        # Footer
        self.ftr = ctk.CTkFrame(root, fg_color=_FTR, corner_radius=0, height=66)
        self.ftr.grid(row=2, column=0, sticky="ew")
        self.ftr.grid_propagate(False)
        self.ftr.grid_columnconfigure(0, weight=1)

    def _clear(self, frame):
        for w in frame.winfo_children():
            w.destroy()

    # ── Page 1: Welcome ───────────────────────────────────────────────────────

    def _show_welcome(self):
        self._clear(self.body)
        self._clear(self.ftr)
        b = self.body

        ctk.CTkLabel(b, text="Welcome", font=_F(22, "bold"),
                     text_color=_PRI).grid(row=0, column=0, sticky="w", pady=(0, 6))
        ctk.CTkLabel(b,
                     text=("This wizard installs the AI transcription engine.\n"
                            "Includes: torch · whisperX · ffmpeg · all dependencies.\n"
                            "This is a one-time 1–3 GB download (2–5 GB on disk).\n"
                            "Subsequent launches open instantly."),
                     font=_F(13), text_color=_DIM,
                     wraplength=500, justify="left",
                     ).grid(row=1, column=0, sticky="w", pady=(0, 18))

        # GPU card
        gcard = ctk.CTkFrame(b, fg_color=_CARD, corner_radius=10)
        gcard.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        gcard.grid_columnconfigure(0, weight=1)

        if self.has_gpu and self.torch_index:
            suffix = self.torch_index.rstrip("/").split("/")[-1].upper()
            gpu_label = f"●  NVIDIA GPU detected — CUDA build will be installed ({suffix})"
            gpu_color = "#22cc66"
        elif self.has_gpu:
            gpu_label = "●  NVIDIA GPU detected — CUDA version too old, using CPU build"
            gpu_color = "#ffaa00"
        else:
            gpu_label = "●  No GPU detected — will run on CPU"
            gpu_color = "#888888"
        ctk.CTkLabel(gcard, text=gpu_label, font=_F(12, "bold"),
                     text_color=gpu_color).grid(row=0, column=0, padx=16, pady=12, sticky="w")

        # Python card
        pcard = ctk.CTkFrame(b, fg_color=_CARD, corner_radius=10)
        pcard.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        pcard.grid_columnconfigure(0, weight=1)

        if self.python_path:
            py_text  = f"●  Python found: {self.python_path}"
            py_color = "#22cc66"
        else:
            py_text  = "●  Python 3.13 not found — will be installed automatically"
            py_color = "#ffaa00"

        ctk.CTkLabel(pcard, text=py_text, font=_F(11),
                     text_color=py_color).grid(row=0, column=0, padx=16, pady=10, sticky="w")

        if not self.python_path:
            ctk.CTkLabel(pcard,
                         text="Python 3.13 will be downloaded and installed automatically.",
                         font=_F(10), text_color=_HINT,
                         ).grid(row=1, column=0, padx=16, pady=(0, 10), sticky="w")

        # Footer
        if self.python_path:
            ctk.CTkButton(self.ftr, text="Begin Setup →", width=160, height=40,
                          font=_F(13, "bold"), fg_color="#4a9eff", hover_color="#6ab0ff",
                          command=self._start_install,
                          ).grid(row=0, column=0, pady=13, padx=18, sticky="e")
        else:
            ctk.CTkLabel(self.ftr,
                         text="Python 3.13 not found — installing automatically...",
                         font=_F(11), text_color=_HINT,
                         ).grid(row=0, column=0, pady=18, padx=18)
            # Auto-start after a brief pause so the user can read the screen
            self.win.after(2000, self._auto_install_python)


    def _check_python_again(self):
        self.python_path = _find_python_fresh()
        self._show_welcome()

    # ── Auto Python install ───────────────────────────────────────────────────

    def _auto_install_python(self):
        self._show_python_installing()
        threading.Thread(target=self._run_python_install, daemon=True).start()

    def _show_python_installing(self):
        self._clear(self.body)
        self._clear(self.ftr)
        b = self.body

        ctk.CTkLabel(b, text="Installing Python...", font=_F(22, "bold"),
                     text_color=_PRI).grid(row=0, column=0, sticky="w", pady=(0, 6))

        self._py_status_var = ctk.StringVar(value=f"Downloading Python {_PYTHON_VERSION}...")
        ctk.CTkLabel(b, textvariable=self._py_status_var, font=_F(12),
                     text_color=_DIM).grid(row=1, column=0, sticky="w", pady=(0, 10))

        prog_row = ctk.CTkFrame(b, fg_color="transparent")
        prog_row.grid(row=2, column=0, sticky="ew")
        prog_row.grid_columnconfigure(0, weight=1)

        self._py_prog = ctk.CTkProgressBar(prog_row, height=8, corner_radius=4,
                                            fg_color=("#d0d0d0", "#1a1a1a"),
                                            progress_color="#4a9eff")
        self._py_prog.set(0)
        self._py_prog.grid(row=0, column=0, sticky="ew")

        self._py_pct = ctk.CTkLabel(prog_row, text="0%", font=_F(10),
                                     text_color=_HINT, width=36)
        self._py_pct.grid(row=0, column=1, padx=(8, 0))

        ctk.CTkLabel(self.ftr,
                     text="Python is being downloaded and installed — this takes about a minute.",
                     font=_F(11), text_color=_HINT,
                     ).grid(row=0, column=0, pady=18, padx=18)

    def _set_py_status(self, msg: str):
        if hasattr(self, "_py_status_var"):
            self.win.after(0, lambda: self._py_status_var.set(msg))

    def _set_py_progress(self, value: float):
        def _do():
            if hasattr(self, "_py_prog"):
                self._py_prog.set(max(0.0, min(1.0, value)))
            if hasattr(self, "_py_pct"):
                self._py_pct.configure(text=f"{int(value * 100)}%")
        self.win.after(0, _do)

    def _run_python_install(self):
        import time
        try:
            tmp = os.path.join(tempfile.gettempdir(), f"python-{_PYTHON_VERSION}-installer.exe")

            # Download
            def _progress(count, block_size, total):
                if total > 0:
                    self._set_py_progress(min(count * block_size / total, 0.95))

            urllib.request.urlretrieve(_PYTHON_URL, tmp, _progress)
            self._set_py_progress(1.0)

            # Run installer — InstallAllUsers=0 keeps it per-user (no UAC needed)
            self._set_py_status("Installing Python — please wait...")
            result = subprocess.run(
                [tmp, "/passive", "PrependPath=1", "Include_test=0",
                 "SimpleInstall=1", "InstallAllUsers=0"],
                timeout=300,
            )
            # 1638 = already installed (also fine)
            if result.returncode not in (0, 1638):
                raise RuntimeError(f"Installer exited with code {result.returncode}.")

            # Retry a few times — installer may not have flushed to disk yet
            self._set_py_status("Verifying installation...")
            for _ in range(6):
                self.python_path = _find_python_fresh()
                if self.python_path:
                    break
                time.sleep(2)

            if self.python_path:
                self._set_py_status("Python installed successfully!")
                self.win.after(800, self._show_welcome)
            else:
                raise RuntimeError("Python was installed — please restart the app to continue.")

        except Exception as exc:
            self._set_py_status(f"Failed: {exc}")
            self.win.after(0, lambda: self._show_py_error(str(exc)))

    def _show_py_error(self, msg: str):
        self._clear(self.ftr)
        self.ftr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.ftr, text=msg[:80], font=_F(10), text_color="#ff4d4d",
                     wraplength=400).grid(row=0, column=0, pady=8, padx=18, sticky="w")
        ctk.CTkButton(self.ftr, text="Try again", width=100, height=36,
                      font=_F(12), fg_color=("#d0d0d0", "#2a2a2a"),
                      hover_color=("#c0c0c0", "#3a3a3a"), text_color=_PRI,
                      command=self._show_welcome,
                      ).grid(row=0, column=1, pady=8, padx=(0, 8))
        ctk.CTkButton(self.ftr, text="Install manually →", width=150, height=36,
                      font=_F(12), fg_color="#4a9eff", hover_color="#6ab0ff",
                      command=lambda: webbrowser.open("https://www.python.org/downloads/"),
                      ).grid(row=0, column=2, pady=8, padx=(0, 18))

    # ── Page 2: Installing ────────────────────────────────────────────────────

    def _show_installing(self):
        self._clear(self.body)
        self._clear(self.ftr)
        b = self.body
        b.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(b, text="Installing...", font=_F(22, "bold"),
                     text_color=_PRI).grid(row=0, column=0, sticky="w", pady=(0, 4))

        self._status_var = ctk.StringVar(value="Preparing...")
        ctk.CTkLabel(b, textvariable=self._status_var, font=_F(12),
                     text_color=_DIM).grid(row=1, column=0, sticky="w", pady=(0, 8))

        prog_row = ctk.CTkFrame(b, fg_color="transparent")
        prog_row.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        prog_row.grid_columnconfigure(0, weight=1)

        self._prog = ctk.CTkProgressBar(prog_row, height=8, corner_radius=4,
                                         fg_color=("#d0d0d0", "#1a1a1a"),
                                         progress_color="#4a9eff",
                                         mode="determinate")
        self._prog.set(0)
        self._prog.grid(row=0, column=0, sticky="ew")

        self._pct_lbl = ctk.CTkLabel(prog_row, text="0%", font=_F(10),
                                      text_color=_HINT, width=36)
        self._pct_lbl.grid(row=0, column=1, padx=(8, 0))

        self._activity_lbl = ctk.CTkLabel(b, text="", font=_FM(9),
                                           text_color=_HINT, anchor="w",
                                           wraplength=540)
        self._activity_lbl.grid(row=3, column=0, sticky="ew", pady=(0, 6))

        self._log_box = ctk.CTkTextbox(b, font=_FM(10), fg_color=_LOG,
                                        text_color=_LTXT, corner_radius=8,
                                        state="disabled")
        self._log_box.grid(row=4, column=0, sticky="nsew")

        ctk.CTkLabel(self.ftr,
                     text="Please wait — typically 10–45 minutes depending on your connection speed.",
                     font=_F(11), text_color=_HINT,
                     ).grid(row=0, column=0, pady=18, padx=18)

    def _drain_log(self):
        while True:
            try:
                kind, text = self._q.get_nowait()
                if kind == "r":
                    if hasattr(self, "_activity_lbl"):
                        t = text.strip()[:90]
                        self.win.after(0, lambda s=t: self._activity_lbl.configure(text=s))
                elif kind == "p":
                    self.win.after(0, lambda v=text: self._set_install_progress(v))
                elif kind == "i":
                    self.win.after(0, self._set_install_indeterminate)
                else:
                    if hasattr(self, "_log_box") and text:
                        if "━" in text or "─" in text:
                            continue
                        self._log_box.configure(state="normal")
                        self._log_box.insert("end", text + "\n")
                        self._log_box.see("end")
                        self._log_box.configure(state="disabled")
            except queue.Empty:
                break
        self.win.after(100, self._drain_log)

    def _log(self, msg: str):
        self._q.put(("n", msg))

    def _set_status(self, msg: str):
        if hasattr(self, "_status_var"):
            self.win.after(0, lambda: self._status_var.set(msg))
        if hasattr(self, "_activity_lbl"):
            self.win.after(0, lambda: self._activity_lbl.configure(text=""))

    def _set_progress(self, value: float, step: str = ""):
        def _do():
            if hasattr(self, "_prog"):
                self._prog.set(max(0.0, min(1.0, value)))
            if hasattr(self, "_pct_lbl"):
                self._pct_lbl.configure(text=f"{int(value * 100)}%")
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
        """Parse pip's 'X/Y MB' download lines into a real progress fraction."""
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*[kKmMgG]i?B", line)
        if not m:
            return
        try:
            done, total = float(m.group(1)), float(m.group(2))
        except ValueError:
            return
        if total <= 0:
            return
        self._real_progress_seen = True
        self._q.put(("p", done / total))

    def _set_install_progress(self, frac: float):
        if hasattr(self, "_prog"):
            self._prog.configure(mode="determinate")
            self._prog.set(max(0.0, min(1.0, frac)))
        if hasattr(self, "_pct_lbl"):
            self._pct_lbl.configure(text=f"{int(frac * 100)}%")

    def _check_install_phase(self, line: str):
        """Downloads are done once pip starts installing — switch to an
        indeterminate bar so it doesn't sit frozen at a download percentage."""
        if "Installing collected packages" in line or "Building wheel" in line:
            self._q.put(("i", ""))

    def _set_install_indeterminate(self):
        if hasattr(self, "_prog"):
            self._prog.configure(mode="indeterminate")
            self._prog.start()
        if hasattr(self, "_pct_lbl"):
            self._pct_lbl.configure(text="")
        if hasattr(self, "_status_var"):
            self._status_var.set("Installing packages (this can take a few minutes)...")

    def _run_install(self):
        try:
            self._real_progress_seen = False
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
            #    Minimum torch version — pip will pick the latest available
            #    wheel for the current Python version.  Using >= instead of ==
            #    so new Python versions (3.14+) aren't blocked by missing exact
            #    wheels on the pytorch index.
            TORCH_MIN = "2.8.0"

            if self._pkg_installed(venv_pip, "whisperx"):
                self._log("\n[whisperx] Already installed — skipping.")
                self._set_progress(0.85)
            else:
                torch_index = _cuda_torch_index()
                self._install_done = False
                self.win.after(1000, lambda: self._animate_progress(0.10, 0.83))

                if torch_index:
                    self._set_status(
                        "Downloading AI engine and dependencies (1–3 GB, CUDA)...")
                    self._log(
                        f"\n[Installing] torch>={TORCH_MIN}+CUDA, whisperx, ffmpeg, deps...")
                    cmd = [
                        venv_pip, "install", "--timeout", "120",
                        f"torch>={TORCH_MIN}",
                        f"torchaudio>={TORCH_MIN}",
                        "whisperx",
                        "imageio-ffmpeg",
                        "customtkinter>=5.2.2",
                        "Pillow",
                        "--index-url", torch_index,
                        "--extra-index-url", "https://pypi.org/simple/",
                    ]
                else:
                    self._set_status(
                        "Downloading AI engine and dependencies (1–3 GB)...")
                    self._log("\n[Installing] whisperx + ffmpeg + customtkinter + Pillow...")
                    cmd = [
                        venv_pip, "install", "--timeout", "120",
                        "whisperx",
                        "imageio-ffmpeg",
                        "customtkinter>=5.2.2",
                        "Pillow",
                    ]

                rc = self._run_pip(cmd)
                self._install_done = True
                if hasattr(self, "_prog"):
                    self.win.after(0, self._prog.stop)
                    self.win.after(0, lambda: self._prog.configure(mode="determinate"))
                if rc != 0:
                    raise RuntimeError(
                        "pip install failed — see the log above for details.")
                self._set_progress(0.85)

            # ffmpeg — checked independently so it installs even when whisperx
            # was already cached (handles upgrades from older app versions)
            if not self._pkg_installed(venv_pip, "imageio-ffmpeg"):
                self._set_status("Installing ffmpeg (audio decoder)...")
                self._log("\n[ffmpeg] Installing portable ffmpeg...")
                rc = self._run_pip([venv_pip, "install", "imageio-ffmpeg",
                                    "--quiet", "--timeout", "120"])
                if rc != 0:
                    raise RuntimeError(
                        "ffmpeg install failed — see the log above for details.")
                self._log("\n[ffmpeg] Done.")
            else:
                self._log("\n[ffmpeg] Already installed — skipping.")
            self._set_progress(0.95)

            # 4. Mark complete
            flag = os.path.join(runtime, ".setup_complete")
            with open(flag, "w") as fh:
                fh.write("ok")

            self._set_progress(1.0)
            self._set_status("Setup complete!")
            self._log("\n✓ All done! Launching app...")
            self.win.after(800, self._show_done)

        except Exception as exc:
            import traceback as tb
            self._log(f"\nERROR: {exc}\n{tb.format_exc()}")
            self._set_status("Setup failed — see log above")
            self.win.after(0, lambda: self._show_error(str(exc)))

    # ── Page 3: Done / Error ──────────────────────────────────────────────────

    def _show_done(self):
        if hasattr(self, "_prog"):
            self._prog.stop()
        self._clear(self.body)
        self._clear(self.ftr)
        b = self.body

        ctk.CTkLabel(b, text="Setup Complete", font=_F(22, "bold"),
                     text_color="#22cc66").grid(row=0, column=0, sticky="w", pady=(0, 12))
        ctk.CTkLabel(b,
                     text=("The AI transcription engine is installed.\n\n"
                            "Click Launch to open WhisperX Transcriber.\n"
                            "Future launches will skip this screen and open instantly."),
                     font=_F(13), text_color=_DIM,
                     wraplength=500, justify="left",
                     ).grid(row=1, column=0, sticky="w")

        ctk.CTkButton(self.ftr, text="Launch App", width=160, height=40,
                      font=_F(13, "bold"), fg_color="#22cc66", hover_color="#2de07a",
                      text_color="#ffffff",
                      command=self._finish,
                      ).grid(row=0, column=0, pady=13, padx=18, sticky="e")

    def _show_error(self, msg: str):
        if hasattr(self, "_prog"):
            try:
                self._prog.stop()
            except Exception:
                pass
        self._clear(self.ftr)
        ctk.CTkLabel(self.ftr, text=f"Error: {msg[:100]}",
                     font=_F(11), text_color="#ff4d4d",
                     wraplength=550,
                     ).grid(row=0, column=0, pady=10, padx=18)
        ctk.CTkButton(self.ftr, text="Retry", width=90, height=36,
                      font=_F(12), fg_color=("#d0d0d0", "#2a2a2a"),
                      hover_color=("#c0c0c0", "#3a3a3a"),
                      command=self._retry,
                      ).grid(row=0, column=1, pady=10, padx=(0, 18))

    def _retry(self):
        self.python_path = _find_python_fresh()
        self._show_welcome()

    def _finish(self):
        self.win.destroy()
        self.on_complete()

    def run(self):
        self.win.mainloop()
