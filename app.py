"""WhisperX Transcriber — thin entry point; the UI lives in ui/shell.py."""
import sys

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Muqaddimah.WhisperXTranscriber")
    except Exception:
        pass

from ui.shell import Shell

def _already_open():
    from tkinter import Tk, messagebox
    root = Tk()
    root.withdraw()
    messagebox.showinfo("WhisperX", "WhisperX is already open.")
    root.destroy()


if __name__ == "__main__":
    from core import settings
    from ui.state import base_dir
    _lock = settings.acquire_lock(base_dir())   # held for the process lifetime
    if _lock is None:
        _already_open()
        sys.exit(0)
    Shell().mainloop()
