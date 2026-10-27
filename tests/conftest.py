"""Shared test setup for the GUI modules, which each build their own Tk root in one interpreter.

* A CTkFont cached against a destroyed root breaks the next module: the theme font cache starts
  empty for every module.
* Creating a root re-reads Tk's script files; when another process briefly holds or replaces
  them Tk fails with "Can't find a usable tk.tcl". Root creation retries that error a few times."""
import time
import tkinter

import pytest

_tk_init = tkinter.Tk.__init__


def _tk_init_retry(self, *args, **kwargs):
    for attempt in range(5):
        try:
            return _tk_init(self, *args, **kwargs)
        except tkinter.TclError as e:
            if "tk.tcl" not in str(e) or attempt == 4:
                raise
            time.sleep(1.0)


tkinter.Tk.__init__ = _tk_init_retry


@pytest.fixture(autouse=True, scope="module")
def _fresh_font_cache():
    from ui import theme
    theme._font_cache.clear()
    yield
    theme._font_cache.clear()


@pytest.fixture(autouse=True)
def _no_real_folder_windows(monkeypatch):
    """Tests must never open Explorer/Finder windows on their temp folders (the open-folder
    setting is on by default); record the requests instead."""
    opened = []
    import ui.state as st
    monkeypatch.setattr(st, "open_folder", opened.append)
    try:
        import ui.transcript as tr
        monkeypatch.setattr(tr, "_open_folder", opened.append)
    except Exception:
        pass
    yield opened
