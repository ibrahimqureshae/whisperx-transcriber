"""H14: adding files is the obvious first action: empty state, slim row, Ctrl+V, whole-window drop."""
import sys
import tkinter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from ui import theme


@pytest.fixture(scope="module")
def shell(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("base"))
    import ui.state as st
    import ui.shell as sh
    mp = pytest.MonkeyPatch()
    mp.setattr(st, "base_dir", lambda: tmp)
    mp.setattr(sh, "base_dir", lambda: tmp)
    mp.setenv("WHISPERX_RUNNER", str(Path(__file__).parent / "fake_runner.py"))
    theme._font_cache.clear()
    try:
        app = sh.Shell()
    except tkinter.TclError as e:
        mp.undo()
        if "display" not in str(e).lower():
            raise
        pytest.skip(f"no display: {e}")
    app.withdraw()
    yield app
    app.app.shutdown()
    app.destroy()
    theme._font_cache.clear()
    mp.undo()


@pytest.fixture
def scr(shell):
    import ui.transcribe as tr
    shell.show("transcribe")
    s = next(cb.__self__ for cb in shell.app._subs["browse"] if isinstance(cb.__self__, tr.Screen))
    shell.app.queue.jobs.clear()
    s._refresh_lists()
    shell.update()
    return s


def _wav(tmp_path, name="a.wav"):
    p = tmp_path / name
    p.write_bytes(b"RIFF")
    return str(p)


def test_empty_state_big_area_heat_add_and_secondary_transcribe(shell, scr):
    heat = theme.C["heat"]
    assert scr._empty and scr.drop.pack_info()["expand"]
    assert scr.add_btn.cget("fg_color") == heat
    assert shell._go.cget("fg_color") != heat
    assert scr.title.cget("text") in ("Drop audio or video", "Add audio or video")


def test_adding_a_file_collapses_to_the_slim_row_and_transcribe_is_primary_again(shell, scr, tmp_path):
    scr.add([_wav(tmp_path)])
    shell.update()
    assert not scr._empty and not scr.drop.pack_info()["expand"]
    assert scr.add_btn.cget("fg_color") == theme.C["raised"]
    assert shell._go.cget("fg_color") == theme.C["heat"]
    assert scr.title.cget("text") in ("Drop more here", "Add more here")


def test_transcribe_with_no_files_opens_the_picker(shell, scr, monkeypatch):
    calls = []
    monkeypatch.setattr(scr, "browse", lambda **_: calls.append(1))
    scr._run()
    assert calls == [1] and str(shell._go.cget("state")) == "normal"


def test_ctrl_v_adds_copied_files(shell, scr, tmp_path, monkeypatch):
    import ui.shell as sh
    f = _wav(tmp_path)
    monkeypatch.setattr(sh, "_hdrop_paths", lambda: [f])
    monkeypatch.setattr(sh, "IS_WIN", True)
    assert shell._paste() == "break"
    assert [j.audio for j in shell.app.queue.jobs] == [f]


def test_ctrl_v_falls_back_to_text_paths_and_ignores_missing_ones(shell, scr, tmp_path, monkeypatch):
    import ui.shell as sh
    f = _wav(tmp_path, "b.wav")
    monkeypatch.setattr(sh, "_hdrop_paths", lambda: [])
    monkeypatch.setattr(shell, "clipboard_get", lambda: f'"{f}"\n{tmp_path / "gone.wav"}\n')
    shell._paste()
    assert [j.audio for j in shell.app.queue.jobs] == [f]


def test_ctrl_v_inside_an_entry_adds_nothing(shell, scr, tmp_path, monkeypatch):
    import ui.shell as sh
    shell.deiconify(); shell.update()
    import tkinter as tk
    e = tk.Entry(shell); e.pack(); e.focus_force(); shell.update()
    monkeypatch.setattr(sh, "_hdrop_paths", lambda: [_wav(tmp_path)])
    assert shell._paste() is None
    assert not shell.app.queue.jobs
    shell.withdraw()


def test_a_folder_adds_its_media_files_and_reports_counts(shell, scr, tmp_path):
    d = tmp_path / "talks"
    d.mkdir()
    for n in ("one.wav", "two.mp3", "notes.txt"):
        (d / n).write_bytes(b"x")
    scr.add([str(d)])
    assert sorted(Path(j.audio).name for j in shell.app.queue.jobs) == ["one.wav", "two.mp3"]
    assert shell._status_lbl.cget("text") == "Added 2 files from folder · skipped 1 not audio/video"


def test_root_drop_adds_files_switches_screen_and_clears_the_highlight(shell, scr, tmp_path):
    f = _wav(tmp_path)
    shell.show("settings")
    shell._drag(True)
    assert len(shell._edges) == 4
    ev = type("E", (), {"data": f.replace("\\", "/"), "action": "copy"})()
    assert shell._on_drop(ev) == "copy"
    assert shell.current == "transcribe" and shell._edges == []
    assert [Path(j.audio).name for j in shell.app.queue.jobs] == ["a.wav"]
