"""Shell: stacked screens, fixed 1366x768 window. Needs a display; skipped without one."""
import sys
import time
import tkinter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def shell(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("base"))
    import ui.state as st
    import ui.shell as sh
    mp = pytest.MonkeyPatch()
    mp.setattr(st, "base_dir", lambda: tmp)
    mp.setattr(sh, "base_dir", lambda: tmp)
    try:
        app = sh.Shell()
    except tkinter.TclError as e:               # no display
        mp.undo()
        pytest.skip(f"no display: {e}")
    app.withdraw()
    yield app
    app.app.shutdown()
    app.destroy()
    mp.undo()


def _top(shell):
    """The built screen highest in the host's stacking order (winfo children: lowest first)."""
    built = {str(w): n for n, w in shell._frames.items() if w.winfo_manager()}
    return [built[c] for c in shell.tk.splitlist(shell.tk.call("winfo", "children", shell._host))
            if c in built][-1]


def test_show_raises_and_keeps_screens_gridded(shell):
    """H20: switching restacks (tkraise); every built screen stays gridded in the one cell."""
    shell.deiconify()
    order = ["transcribe", "queue", "transcript", "models", "settings", "log"]
    for name in order + order + order[::-1]:
        shell.show(name)
        shell.update()
        assert shell.current == name and _top(shell) == name
        built = [w for n, w in shell._frames.items() if n in order[:order.index(name) + 1] or w.winfo_manager()]
        assert all(w.winfo_manager() == "grid" and w.grid_info()["row"] == 0 for w in built)
        assert len(shell._host.grid_slaves()) == len(built)
    shell.withdraw()


def test_show_uses_tkraise(shell, monkeypatch):
    raised = []
    frame = shell._frames.get("queue") or (shell.show("queue"), shell._frames["queue"])[1]
    monkeypatch.setattr(frame, "tkraise", lambda *a: raised.append(a))
    shell.show("queue")
    assert raised and frame.winfo_manager() == "grid"


def test_on_show_fires(shell, monkeypatch):
    shell.show("models")
    calls = []
    monkeypatch.setattr(shell._frames["models"], "on_show", lambda: calls.append(1), raising=False)
    shell.show("transcribe")
    shell.show("models")
    shell.show("models")
    assert calls == [1, 1]


def test_fixed_window(shell):
    import customtkinter as ctk
    assert tuple(bool(v) for v in shell.resizable()) == (False, False)
    shell.deiconify()
    shell.update()
    k = ctk.ScalingTracker.get_window_scaling(shell)
    assert round(shell.winfo_width() / k) == 1366
    assert round(shell.winfo_height() / k) == 768
    shell.withdraw()


def test_rescale_keeps_screens_stacked(shell):
    import customtkinter as ctk
    shell.show("transcribe")
    shell.show("queue")
    ctk.set_widget_scaling(ctk.ScalingTracker.widget_scaling)
    shell.update()
    assert len(shell._host.grid_slaves()) == len([w for w in shell._frames.values() if w.winfo_manager()])
    assert _top(shell) == "queue"


@pytest.mark.parametrize("area, pin, want", [
    ((2560, 1440), "auto", 1.5), ((2560, 1392), "auto", 1.5),      # owner's monitor @100 %, with taskbar
    ((1366, 768), "auto", 0.95), ((1920, 1040), "auto", 1.0), ((3840, 2100), "auto", 2.0),
    ((1200, 700), "auto", 0.85), ((2560, 1440), "1.25", 1.25), ((1366, 768), "1.5", 1.5),
    ((2560, 1440), "junk", 1.5)])
def test_pick_scale(area, pin, want):
    import ui.shell as sh
    assert sh.Shell._pick_scale(*area, pin) == pytest.approx(want)


_AUTO_SCALE = """
import sys, time, tkinter
sys.path.insert(0, {root!r})
import customtkinter as ctk
import ui.state as st, ui.shell as sh
st.base_dir = sh.base_dir = lambda: {base!r}
sh.Shell._work_area = lambda self, s: {area!r}
try:
    app = sh.Shell()
except tkinter.TclError:
    sys.exit(77)
end = time.time() + 1.5       # covers CTk's delayed _set_scaled_min_max
sizes = []
for stop in (0, end):
    app.update()
    while time.time() < stop:
        app.update(); time.sleep(0.02)
    k = ctk.ScalingTracker.get_window_scaling(app)
    sizes.append((round(app.winfo_width() / k), round(app.winfo_height() / k)))
print(sizes)
app.app.shutdown()
sys.exit(0 if sizes == [(1366, 768)] * 2 else 1)
"""


@pytest.mark.parametrize("area", [(0, 0, 1200, 700), (0, 0, 1280, 720), (0, 0, 2560, 1392)])
def test_auto_scale_keeps_logical_size(area, tmp_path):
    """Own process: a second Tk root in the test process is unreliable."""
    import subprocess
    root = str(Path(__file__).parent.parent)
    code = _AUTO_SCALE.format(root=root, base=str(tmp_path), area=area)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    if r.returncode == 77:
        pytest.skip("no display")
    assert r.returncode == 0, r.stdout + r.stderr


# ── H8c: status topic, title, subtitle, activity log ──────────────────────────
def test_status_topic_and_percent_spacing(shell):
    shell.app.emit("status", text="Skipped 2: not audio/video")
    assert shell._status_lbl.cget("text") == "Skipped 2: not audio/video"
    job = shell.app.queue.add("clip.wav", {})
    shell._on_runner(job, {"t": "status", "m": "Transcribing… 35%"})
    shell._on_runner(job, {"t": "progress"})
    assert shell._status_lbl.cget("text") == "Skipped 2: not audio/video"      # say it once
    shell.app.emit("notice", text="Saved")
    assert shell._status_lbl.cget("text") == "Saved"


def test_no_running_subtitle_and_no_session_block(shell):
    assert not hasattr(shell, "_sub") and not hasattr(shell, "_sess")


def test_save_to_only_on_transcribe(shell):
    shell.show("transcribe")
    shell.update_idletasks()
    assert shell._save.winfo_manager() == "pack"
    for name in ("queue", "transcript", "models", "settings"):
        shell.show(name)
        assert shell._save.winfo_manager() == ""
    shell.show("transcribe")
    assert shell._save.winfo_manager() == "pack"


class FakeTaskbar:
    def __init__(self):
        self.calls = []

    def progress(self, f):
        self.calls.append(("progress", round(f, 2)))

    def error(self):
        self.calls.append(("error",))

    def clear(self):
        self.calls.append(("clear",))


def test_running_job_shows_dot_percent_and_taskbar(shell):
    tb, shell.taskbar = shell.taskbar, FakeTaskbar()
    try:
        item = shell._nav["transcribe"]
        job = shell.app.queue.add("clip.wav", {})
        shell.app.running, job.progress = job, 0.35
        shell._pct = None
        shell._on_runner(job, {"t": "progress"})
        assert item.count.cget("text") == "35 %" and item._phase is not None
        assert item._dot.winfo_manager() == "pack"
        assert shell.taskbar.calls == [("progress", 0.35)]
        shell._on_runner(job, {"t": "progress"})            # same percent: nothing repeated
        assert len(shell.taskbar.calls) == 1
        job.progress = 0.5
        shell._on_runner(job, {"t": "progress"})
        assert shell.taskbar.calls[-1] == ("progress", 0.5)
        shell.app.running, job.status = None, "done"
        shell._refresh_jobs()
        shell._on_finished(job)
        assert item.count.cget("text") == "" and item._phase is None
        assert item._dot.winfo_manager() == ""
        assert shell.taskbar.calls[-1] == ("clear",)
    finally:
        shell.taskbar = tb


def test_failed_job_flags_taskbar_error_then_clears(shell):
    tb, shell.taskbar = shell.taskbar, FakeTaskbar()
    try:
        job = shell.app.queue.add("bad.wav", {})
        job.status = "failed"
        shell._refresh_jobs()
        shell._on_finished(job)
        assert shell.taskbar.calls[-1] == ("error",) and shell._tb_err is not None
        shell._refresh_jobs()                               # idle refresh keeps the red
        assert shell.taskbar.calls[-1] == ("error",)
        shell._tb_error_done()
        assert shell.taskbar.calls[-1] == ("clear",) and shell._tb_err is None
    finally:
        shell.taskbar = tb


def test_taskbar_com_calls_and_failure_swallowed():
    import ui.shell as sh
    t = sh.Taskbar(lambda: 77)
    t._dead, t._obj = False, object()
    seen = []
    t._call = lambda name, *a: seen.append((name,) + a)
    t.progress(0.4)
    t.error()
    t.clear()
    assert seen == [("SetProgressState", 77, 2), ("SetProgressValue", 77, 400, 1000),
                    ("SetProgressValue", 77, 1000, 1000), ("SetProgressState", 77, 4),
                    ("SetProgressState", 77, 0)]

    def boom(*a):
        raise OSError("com")
    t._call = boom
    t.progress(0.5)                                         # swallowed, then inert
    assert t._dead
    t._call = lambda *a: seen.append("again")
    t.progress(0.6)
    assert "again" not in seen
    assert sh.Taskbar(lambda: 1)._dead == (not sh.IS_WIN)


def test_window_title_has_the_version(shell):
    assert shell.title() == (f"WhisperX Transcriber \u2014 {shell.version}" if shell.version
                             else "WhisperX Transcriber")


def test_activity_log_lines_carry_the_file_name(shell):
    job = shell.app.queue.add(r"C:\talks\interview.m4a", {})
    shell._on_runner(job, {"t": "warn", "m": "low memory"})
    last = shell._log.get("1.0", "end").strip().splitlines()[-1]
    assert "interview.m4a  WARN  low memory" in last


# --- starting a run shows the Transcribe screen (T10) ---

def _recorder(shell, monkeypatch):
    """Replace the transcribe subscribers (they'd start a real job) with one recorder."""
    seen = []
    monkeypatch.setitem(shell.app._subs, "transcribe", [lambda **kw: seen.append(shell.current)])
    return seen


def _ctrl_enter(shell):
    shell.deiconify()
    shell.update()
    shell.focus_force()
    shell.update()
    shell.event_generate("<Control-Return>")
    shell.update()


@pytest.mark.parametrize("start", ["Settings", "Queue", "Transcribe"])
def test_footer_button_leaves_expected_screen(shell, monkeypatch, start):
    seen = _recorder(shell, monkeypatch)
    expect = "queue" if start == "Queue" else "transcribe"
    try:
        shell.show(start.lower())
        shell._go.invoke()
        assert shell.current == expect
        assert len(seen) == 1
    finally:
        shell.show("transcribe")


@pytest.mark.parametrize("start", ["settings", "queue", "transcribe"])
def test_ctrl_enter_leaves_expected_screen(shell, monkeypatch, start):
    seen = _recorder(shell, monkeypatch)
    expect = "queue" if start == "queue" else "transcribe"
    try:
        shell.show(start)
        _ctrl_enter(shell)
        assert shell.current == expect
        assert len(seen) == 1
    finally:
        shell.withdraw()
        shell.show("transcribe")


def test_footer_button_switches_before_emitting_start(shell, monkeypatch):
    seen = _recorder(shell, monkeypatch)
    try:
        shell.show("settings")
        shell._go.invoke()
        assert seen == ["transcribe"]
    finally:
        shell.show("transcribe")


def test_ctrl_enter_switches_before_emitting_start(shell, monkeypatch):
    seen = _recorder(shell, monkeypatch)
    try:
        shell.show("settings")
        _ctrl_enter(shell)
        assert seen == ["transcribe"]
    finally:
        shell.withdraw()
        shell.show("transcribe")


def test_window_icon_photo_keeps_source_resolution(tmp_path, monkeypatch):
    import ui.state as st
    import ui.shell as sh
    from PIL import ImageTk

    recorded = {}

    class FakePhoto:
        def __init__(self, image, master=None):
            recorded["photo_size"] = image.size

    def fake_iconphoto(self, default, image):
        recorded["iconphoto"] = default

    monkeypatch.setattr(st, "base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sh, "base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sh.os.path, "isfile", lambda path: False)
    monkeypatch.setattr(ImageTk, "PhotoImage", FakePhoto)
    monkeypatch.setattr(sh.Shell, "iconphoto", fake_iconphoto)

    try:
        app = sh.Shell()
    except tkinter.TclError as e:
        pytest.skip(f"no display: {e}")
    app.withdraw()
    try:
        assert recorded["iconphoto"] is True
        assert recorded["photo_size"][0] >= 128
    finally:
        app.app.shutdown()
        app.destroy()
