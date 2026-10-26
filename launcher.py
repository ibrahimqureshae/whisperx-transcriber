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
