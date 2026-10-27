"""H7: field rules (core.settings.validate / clamp), build_cfg clamping, masked token field."""
import copy
import sys
import tkinter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.settings import DEFAULTS, FIELD_RULES, clamp, validate  # noqa: E402

_NUMERIC = [k for k, r in FIELD_RULES.items() if r[0] in ("int", "float", "floats")]


# ── rules ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("key", _NUMERIC)
def test_boundaries_valid_and_just_outside_invalid(key):
    kind, lo, hi, _blank = FIELD_RULES[key]
    fmt = (lambda v: str(int(v))) if kind == "int" else (lambda v: repr(float(v)))
    assert validate(key, fmt(lo))[0] and validate(key, fmt(hi))[0]
    step = 1 if kind == "int" else 0.001
    ok, text, why = validate(key, fmt(lo - step))
    assert not ok and why and text == fmt(lo - step)
    assert not validate(key, fmt(hi + step))[0]


@pytest.mark.parametrize("key", _NUMERIC)
def test_junk_is_invalid(key):
    for bad in ("abc", "nan", "inf", "x1"):
        assert not validate(key, bad)[0], (key, bad)


def test_blank_allowed_only_where_ruled():
    for key, (_k, _lo, _hi, blank_ok) in FIELD_RULES.items():
        ok, text, why = validate(key, "  ")
        assert ok is blank_ok, key
        assert text == "" and (ok or why == "Required.")


def test_integer_rule_rejects_fractions():
    assert not validate("beam", "2.5")[0]
    assert validate("beam", " 7 ") == (True, "7", "")


def test_float_normalised():
    assert validate("vad_onset", "0.500") == (True, "0.5", "")
    assert not validate("vad_onset", "2")[0]


def test_temperatures_normalised_and_ranged():
    assert validate("decode.temperatures", "0,0.2") == (True, "0, 0.2", "")
    assert validate("decode.temperatures", "0,1.0") == (True, "0, 1", "")
    assert not validate("decode.temperatures", "hot")[0]
    assert not validate("decode.temperatures", "1,,2")[0]
    assert not validate("decode.temperatures", "0, 1.5")[0]


def test_condition_on_previous_text():
    assert validate("decode.condition_on_previous_text", "TRUE") == (True, "true", "")
    assert validate("decode.condition_on_previous_text", "false")[0]
    assert not validate("decode.condition_on_previous_text", "maybe")[0]


def test_bare_decode_name_and_unknown_key():
    assert not validate("best_of", "11")[0]
    assert validate("hotwords", "anything at all") == (True, "anything at all", "")


def test_min_max_speakers_cross_check():
    cur = {"min_speakers": "2", "max_speakers": "5"}
    assert not validate("max_speakers", "1", cur)[0]
    assert not validate("min_speakers", "6", cur)[0]
    assert validate("max_speakers", "2", cur)[0] and validate("min_speakers", "5", cur)[0]
    assert validate("max_speakers", "1", {"min_speakers": ""})[0]


def test_defaults_pass_their_own_rules():
    for key in FIELD_RULES:
        val = DEFAULTS["decode"][key[7:]] if key.startswith("decode.") else DEFAULTS[key]
        assert validate(key, val)[0], key


def test_clamp():
    assert clamp("beam", "99") == 10 and clamp("beam", "-3") == 1
    assert clamp("vad_onset", "2") == 0.95
    assert clamp("beam", "abc") is None and clamp("max_width", "") is None
    assert clamp("decode.temperatures", "-1, 0.5, 4") == [0.0, 0.5, 1.0]


# ── build_cfg ─────────────────────────────────────────────────────────────────
def test_build_cfg_clamps_a_bad_settings_dict():
    from ui.state import build_cfg
    s = copy.deepcopy(DEFAULTS)
    s.update(beam="abc", batch="-1", threads="0", vad_onset="2", vad_offset="0",
             chunk="999", device_index="x", min_speakers="5", max_speakers="2",
             max_width="5", max_count="50")
    s["decode"].update(temperatures="0, 7", best_of="99", patience="0", max_new_tokens="9999",
                       no_speech_threshold="3", length_penalty="x")
    cfg = build_cfg(s, "a.wav")
    assert (cfg["beam"], cfg["batch"], cfg["threads"], cfg["chunk"], cfg["device_index"]) == (5, 1, 1, 30, 0)
    assert (cfg["vad_onset"], cfg["vad_offset"]) == (0.95, 0.05)
    assert (cfg["min_speakers"], cfg["max_speakers"]) == (2, 5)
    assert (cfg["max_width"], cfg["max_count"]) == (10, 5)
    assert cfg["decode"] == {"temperatures": [0.0, 1.0], "best_of": 10, "patience": 0.1,
                             "max_new_tokens": 448, "no_speech_threshold": 1.0}


def test_build_cfg_blank_speakers_and_widths_are_none():
    from ui.state import build_cfg
    cfg = build_cfg(copy.deepcopy(DEFAULTS), "a.wav")
    assert cfg["min_speakers"] is None and cfg["max_speakers"] is None
    assert cfg["max_width"] is None and cfg["max_count"] is None


# ── Settings screen ───────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def root():
    import customtkinter as ctk
    from ui import theme
    try:
        r = ctk.CTk()
    except tkinter.TclError as e:
        pytest.skip(f"no display: {e}")
    r.withdraw()
    theme.load_fonts()
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
    yield scr, state, root
    scr.destroy()


def _ents(scr):
    return {k: e for k, _r, e in scr._refreshers if e is not None}


def _commit(root, entry, text):
    root.deiconify()                            # key events need a mapped, focused window
    root.update()
    entry.focus_force()
    root.update()
    entry.set(text)
    entry.event_generate("<Return>")
    root.update()
    root.withdraw()


def test_invalid_entry_restores_flashes_and_does_not_save(screen):
    from ui.theme import C
    scr, state, root = screen
    beam = _ents(scr)["beam"]
    _commit(root, beam, "abc")
    assert state.settings["beam"] == "5" and beam.get() == "5"
    assert scr._stamps[0].cget("text") == ""
    assert str(beam.cget("highlightbackground")) == C["bad"]
    assert scr._tips["beam"][0].text.startswith("Enter a whole number")


def test_border_and_tip_revert_after_flash(screen):
    from ui.settings import TIPS
    from ui.theme import C
    scr, state, root = screen
    beam = _ents(scr)["beam"]
    _commit(root, beam, "99")
    root.after(2200, root.quit)
    root.mainloop()
    assert str(beam.cget("highlightbackground")) == C["rule"]
    assert scr._tips["beam"][0].text == TIPS["beam"]


def test_valid_entry_is_normalised_and_saved(screen):
    scr, state, root = screen
    t = _ents(scr)["decode.temperatures"]
    _commit(root, t, "0,0.2")
    assert state.settings["decode"]["temperatures"] == "0, 0.2"
    assert t.get() == "0, 0.2"
    assert scr._stamps[0].cget("text").startswith("Saved ")


def test_min_above_max_refused(screen):
    scr, state, root = screen
    scr._show(3)
    state.set("max_speakers", "2")
    root.update()
    mn = _ents(scr)["min_speakers"]
    _commit(root, mn, "5")
    assert state.settings["min_speakers"] == "" and mn.get() == ""
