"""
WhisperX Transcriber — thin entry point.

On first launch:  shows setup_wizard.py to install the AI backend.
On later launches: checks for runtime/.setup_complete, then spawns
                   runtime/Scripts/python.exe app.py and exits.

This file and its PyInstaller bundle intentionally contain NO torch,
whisperx, ctranslate2, or any ML package — they download on first run.
"""
import os
import sys
import subprocess


def _base_dir() -> str:
    """Directory that contains (or will contain) runtime/ and app.py."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _runtime_python() -> str:
    base = _base_dir()
    pythonw = os.path.join(base, "runtime", "Scripts", "pythonw.exe")
    python  = os.path.join(base, "runtime", "Scripts", "python.exe")
    return pythonw if os.path.isfile(pythonw) else python


def _runtime_ready() -> bool:
    """Fast check — look for the flag file and verify ffmpeg is reachable."""
    py = os.path.join(_base_dir(), "runtime", "Scripts", "python.exe")
    if not os.path.isfile(py):
        return False
    flag = os.path.join(_base_dir(), "runtime", ".setup_complete")
    if not os.path.isfile(flag):
        return False
    # Ensure ffmpeg is available — bundled via imageio-ffmpeg or on system PATH
    ffmpeg_binaries = os.path.join(_base_dir(), "runtime", "Lib", "site-packages",
                                   "imageio_ffmpeg", "binaries")
    if os.path.isdir(ffmpeg_binaries):
        return True
    try:
        import shutil
        return bool(shutil.which("ffmpeg"))
    except Exception:
        return False


def _launch_app() -> None:
    """Replace this process with runtime/python app.py."""
    base = _base_dir()
    runtime_py = _runtime_python()
    app_script = os.path.join(base, "app.py")

    if not os.path.isfile(app_script):
        _fatal(f"app.py not found at:\n{app_script}\n\nReinstall the application.")

    if not os.path.isfile(runtime_py):
        _fatal("Runtime Python not found. Run the app again to reinstall the backend.")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    # Prepend bundled ffmpeg binaries to PATH so whisperx.load_audio can find ffmpeg
    ffmpeg_bin = os.path.join(base, "runtime", "Lib", "site-packages",
                              "imageio_ffmpeg", "binaries")
    if os.path.isdir(ffmpeg_bin):
        env["PATH"] = ffmpeg_bin + os.pathsep + env.get("PATH", "")

    subprocess.Popen(
        [runtime_py, app_script],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    sys.exit(0)


def _fatal(msg: str) -> None:
    """Show an error dialog then exit."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("WhisperX Transcriber", msg)
        root.destroy()
    except Exception:
        print("FATAL:", msg, file=sys.stderr)
    sys.exit(1)


DND_PIP_TIMEOUT = 60        # seconds for the pip call
DND_BUDGET = 70             # seconds for the whole one-time step, splash included
DND_TRIES = 3               # failed installs before the step is given up for good


def _splash(text: str):
    """Small borderless window while the one-time install runs; None if Tk is unavailable."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        tk.Label(root, text=text, padx=28, pady=18, bg="#16181d", fg="#e8e6e1",
                 font=("Segoe UI", 11)).pack()
        root.update_idletasks()
        root.geometry(f"+{(root.winfo_screenwidth() - root.winfo_width()) // 2}"
                      f"+{(root.winfo_screenheight() - root.winfo_height()) // 3}")
        root.update()
        return root
    except Exception:
        return None


def _ensure_dnd() -> None:
    _ensure_pkg("tkinterdnd2", "tkinterdnd2>=0.4.2", ".dnd_checked", "Adding drag-and-drop support…")


def _ensure_speakers() -> None:
    _ensure_pkg("sherpa_onnx", "sherpa-onnx>=1.13", ".sherpa_checked", "Adding speaker identification…")


def _ensure_rtl() -> None:
    _ensure_pkg("bidi, arabic_reshaper", ["python-bidi>=0.6", "arabic-reshaper>=3.0"],
                ".rtl_checked", "Adding right-to-left text support…")


def _ensure_pkg(module: str, spec, marker_name: str, splash_text: str) -> None:
    """Best-effort: add a package to an existing runtime once. The marker is written on
    success, or after DND_TRIES failed attempts (it then holds the count); never blocks > DND_BUDGET s."""
    specs = [spec] if isinstance(spec, str) else list(spec)
    runtime = os.path.join(_base_dir(), "runtime")
    marker = os.path.join(runtime, marker_name)
    py = os.path.join(runtime, "Scripts", "python.exe")
    if not os.path.isfile(py):
        return
    fails = 0
    if os.path.exists(marker):
        try:
            with open(marker, encoding="utf-8") as f:
                txt = f.read().strip()
        except OSError:
            txt = ""
        if not txt.isdigit() or int(txt) >= DND_TRIES:      # empty = marker of an earlier release
            return
        fails = int(txt)
    import time
    t0 = time.monotonic()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    ok = False
    splash = None
    try:
        r = subprocess.run([py, "-c", f"import {module}"], timeout=8,
                           capture_output=True, creationflags=flags)
        ok = r.returncode == 0
        if not ok:
            splash = _splash(splash_text)
            left = max(1, min(DND_PIP_TIMEOUT, DND_BUDGET - (time.monotonic() - t0)))
            r = subprocess.run([py, "-m", "pip", "install", *specs, "--quiet",
                                "--timeout", str(DND_PIP_TIMEOUT)], timeout=left,
                               capture_output=True, creationflags=flags)
            ok = r.returncode == 0
    except Exception:
        pass
    finally:
        if splash is not None:
            try:
                splash.destroy()
            except Exception:
                pass
    if not ok:
        fails += 1
    try:
        with open(marker, "w", encoding="utf-8") as f:
            f.write("done" if ok else str(fails))      # digits below DND_TRIES = retry next launch
    except OSError:
        pass


def main() -> None:
    if _runtime_ready():
        _ensure_dnd()
        _ensure_speakers()
        _ensure_rtl()
        _launch_app()
    else:
        try:
            from setup_wizard import SetupWizard
        except ImportError as e:
            _fatal(f"Cannot load setup wizard: {e}")
        wizard = SetupWizard(base_dir=_base_dir(), on_complete=_launch_app)
        wizard.run()


if __name__ == "__main__":
    main()
