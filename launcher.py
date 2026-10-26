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
import glob
import subprocess

IS_WIN = sys.platform == "win32"


def _base_dir() -> str:
    """Directory that contains (or will contain) runtime/ and app.py."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _venv_python(base: str, *, console: bool = False) -> str:
    """Path to the runtime venv's interpreter (platform-aware).

    Windows uses runtime/Scripts/python(w).exe; POSIX uses runtime/bin/python.
    `console=True` always returns the console interpreter (used for checks).
    """
    if IS_WIN:
        python  = os.path.join(base, "runtime", "Scripts", "python.exe")
        if console:
            return python
        pythonw = os.path.join(base, "runtime", "Scripts", "pythonw.exe")
        return pythonw if os.path.isfile(pythonw) else python
    return os.path.join(base, "runtime", "bin", "python")


def _ffmpeg_bin_dir(base: str) -> str:
    """Folder with the bundled imageio-ffmpeg binaries, or '' if not present."""
    if IS_WIN:
        cand = os.path.join(base, "runtime", "Lib", "site-packages",
                            "imageio_ffmpeg", "binaries")
        return cand if os.path.isdir(cand) else ""
    for cand in glob.glob(os.path.join(base, "runtime", "lib", "python*",
                                       "site-packages", "imageio_ffmpeg",
                                       "binaries")):
        if os.path.isdir(cand):
            return cand
    return ""


def _runtime_python() -> str:
    return _venv_python(_base_dir())


def _runtime_ready() -> bool:
    """Fast check — look for the flag file and verify ffmpeg is reachable."""
    base = _base_dir()
    py = _venv_python(base, console=True)
    if not os.path.isfile(py):
        return False
    flag = os.path.join(base, "runtime", ".setup_complete")
    if not os.path.isfile(flag):
        return False
    # Ensure ffmpeg is available — bundled via imageio-ffmpeg or on system PATH
    if _ffmpeg_bin_dir(base):
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
    ffmpeg_bin = _ffmpeg_bin_dir(base)
    if ffmpeg_bin:
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


def main() -> None:
    if _runtime_ready():
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
