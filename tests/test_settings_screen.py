"""Settings screen: areas built lazily, every key has a control, external changes refresh controls.
Needs a display; skipped without one."""
import sys
import tkinter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def root():
    import customtkinter as ctk
    from ui import theme
    try:
        r = ctk.CTk()
    except tkinter.TclError as e:               # no display
        pytest.skip(f"no display: {e}")
    r.withdraw()
    theme.load_fonts()                          # theme caches fonts per interpreter: one root per module
    yield r
    r.destroy()


@pytest.fixture()
def screen(root, tmp_path):
    from ui import settings as S
    from ui.state import AppState
    state = AppState(lambda fn: root.after(0, fn), base=str(tmp_path))
    scr = S.build(root, state)
    scr.pack(fill="both", expand=True)
    root.update()
    yield S, scr, state, root
    scr.destroy()


def test_keys_cover_defaults():
    from core.settings import DEFAULTS
    from ui import settings as S
    assert S.KEYS_WITH_CONTROLS == set(DEFAULTS) - {"file", "theme", "last_dir", "ui_scale"}


def test_areas_build_lazily_and_switch(screen):
    S, scr, _state, root = screen
    assert set(scr._areas) == {0}               # only the first area exists after build
    for i in range(len(S.AREAS)):
        scr._show(i)
        root.update_idletasks()
        assert scr._cur == i
        assert [k for k, f in scr._areas.items() if f.winfo_manager()] == [i]
    assert set(scr._areas) == set(range(len(S.AREAS)))
    built = dict(scr._areas)
    scr._show(0)
    assert scr._areas == built                  # switching back reuses the frames


def test_external_set_updates_control(screen):
    S, scr, state, root = screen
    scr._show(0)
    state.set("beam", "7")
    state.set("decode", {**state.settings["decode"], "best_of": "9"})
    state.set("model", "small")
    scr._show(3)
    state.set("min_speakers", "2")
    root.update()
    ents = {k: e for k, _r, e in scr._refreshers if e is not None}
    assert ents["beam"].get() == "7"
    assert ents["decode.best_of"].get() == "9"
    assert ents["min_speakers"].get() == "2"


def test_commit_saves_and_stamps(screen):
    S, scr, state, root = screen
    root.deiconify()
    root.update()
    ents = {k: e for k, _r, e in scr._refreshers if e is not None}
    for key, text in (("beam", "3"), ("decode.best_of", "4")):
        ents[key].focus_force()
        root.update()
        ents[key].set(text)
        ents[key].event_generate("<Return>")
        root.update()
    root.withdraw()
    assert state.settings["beam"] == "3"
    assert state.settings["decode"]["best_of"] == "4"
    assert scr._stamps[0].cget("text").startswith("Saved ")


def test_speakers_area_has_no_token_or_model_field(screen):
    S, scr, state, root = screen
    scr._show(3)
    keys = {k for k, _r, _e in scr._refreshers}
    assert "hf_token" not in keys and "diarize_model" not in keys
    assert {"min_speakers", "max_speakers", "diarize"} <= keys


def test_refusals_within_two_seconds_restore_the_real_tip(screen):
    S, scr, state, root = screen
    root.deiconify()
    root.update()
    beam = next(e for k, _r, e in scr._refreshers if k == "beam")
    beam.focus_force()
    root.update()
    for bad in ("abc", "99"):
        beam.set(bad)
        beam.event_generate("<Return>")
        root.update()
        assert scr._tips["beam"][0].text != S.TIPS["beam"]
    root.withdraw()
    end = root.tk.call("clock", "milliseconds") + 2300
    while root.tk.call("clock", "milliseconds") < end:
        root.update()
    assert scr._tips["beam"][0].text == S.TIPS["beam"]


def test_translate_guard_follows_model(screen):
    S, scr, state, root = screen
    state.set("task", "translate")
    state.set("model", "turbo")
    root.update()
    assert state.settings["task"] == "transcribe"
    assert scr._tips["task"][0].text == S.NO_TRANSLATE
    state.set("model", "medium")
    root.update()
    assert scr._tips["task"][0].text == S.TIPS["task"]


def test_condition_on_previous_text_is_a_switch(screen):
    S, scr, state, root = screen
    key = "decode.condition_on_previous_text"
    sw = next(r for k, r, e in scr._refreshers if k == key)
    assert _entry_free(scr, key)
    scr._save(0, key, "true")
    sw()
    assert state.settings["decode"]["condition_on_previous_text"] == "true"


def _entry_free(scr, key):
    return all(e is None for k, _r, e in scr._refreshers if k == key)


def test_every_field_row_fills_the_four_columns(screen):
    S, scr, state, root = screen
    for _title, lines in S.AREAS:
        for line in lines or []:
            if line[0][0] != "switch":
                assert sum(c[4] for c in line) == 4, line
            else:
                assert len(line) == 1


def test_dir_picker_uses_and_stores_last_dir(screen, monkeypatch, tmp_path):
    S, scr, state, root = screen
    root.deiconify()
    scr._show(5)
    root.update()
    seen = []
    monkeypatch.setattr(S.filedialog, "askdirectory", lambda **kw: seen.append(kw["initialdir"]) or str(tmp_path))
    state.settings["last_dir"] = "X:/last"
    state.settings["out_dir"] = ""
    ent = next(e for k, _r, e in scr._refreshers if k == "out_dir")
    pills = [w for w in ent.master.winfo_children() if w.winfo_class() == "Canvas"][0]
    pills.event_generate("<Button-1>", x=10, y=10)
    root.update()
    root.withdraw()
    assert seen == ["X:/last"]
    assert state.settings["last_dir"] == str(tmp_path)


def test_reset_before_reset_tab_is_built(screen, monkeypatch):
    S, scr, state, root = screen
    monkeypatch.setattr(S.messagebox, "askyesno", lambda *a, **k: True)
    state.settings["beam"] = "7"
    assert len(S.AREAS) - 1 not in scr._stamps
    scr._reset()
    assert state.settings["beam"] == "5"


def test_selected_check_chip_is_filled_ok_green(root):
    from ui import theme
    p = theme.light_pills(root, ["A", "B"], multi=True, check=True, selected=["A"])
    box = [i for i in p.find_all() if p.type(i) == "rectangle"][0]
    assert p.itemcget(box, "fill") == theme.C["ok"]
    p.destroy()
