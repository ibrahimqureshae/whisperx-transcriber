"""Tests for the Setup window shell, theme, step tracker and welcome state in setup_wizard.py."""
import os
import sys
from pathlib import Path

import customtkinter as ctk
import pytest

# Make sure the project root is on the path when running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import setup_wizard

ROOT = Path(__file__).parent.parent
HEAT = "#ff6a1a"
PYTHON_EXE = "C:/fake/python.exe"


# -- helpers -------------------------------------------------------------------

def _patch_env(monkeypatch, python=PYTHON_EXE, gpu=False, index=None):
    monkeypatch.setattr(setup_wizard, "_has_nvidia", lambda: gpu)
    monkeypatch.setattr(setup_wizard, "_cuda_torch_index", lambda: index)
    monkeypatch.setattr(setup_wizard, "_find_python", lambda: python)


def _walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from _walk(child)


def _buttons(win):
    """[(text, fg_color)] for every CTkButton in the window."""
    return [(w.cget("text"), w.cget("fg_color")) for w in _walk(win)
            if isinstance(w, ctk.CTkButton)]


def _label_texts(win):
    return [w.cget("text") for w in _walk(win) if isinstance(w, ctk.CTkLabel)]


def _is_heat(color):
    values = color if isinstance(color, (tuple, list)) else (color,)
    return any(str(v).lower() == HEAT for v in values)


def _make_wizard(monkeypatch, **env):
    _patch_env(monkeypatch, **env)
    return setup_wizard.SetupWizard(str(ROOT), lambda: None)


@pytest.fixture
def wizard(monkeypatch):
    wiz = _make_wizard(monkeypatch)
    try:
        yield wiz
    finally:
        try:
            wiz.win.destroy()
        except Exception:
            pass


@pytest.fixture
def wizard_no_python(monkeypatch):
    wiz = _make_wizard(monkeypatch, python=None)
    try:
        yield wiz
    finally:
        try:
            wiz.win.destroy()
        except Exception:
            pass


# -- window: size, resizing, icon ----------------------------------------------

def test_window_is_720_by_540(wizard):
    wizard.win.update()
    assert (wizard.win._current_width, wizard.win._current_height) == (720, 540)


def test_window_is_not_resizable(wizard):
    assert not any(wizard.win.resizable())


def test_window_icon_is_the_app_icon(monkeypatch):
    icon = ROOT / "assets" / "icon.ico"
    assert icon.exists()
    calls = []
    original = ctk.CTk.iconbitmap

    def record(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ctk.CTk, "iconbitmap", record)
    wiz = _make_wizard(monkeypatch)
    try:
        wiz.win.update()
        paths = [str(a[0]) for a, _ in calls if a] + \
                [str(k.get("bitmap") or k.get("default")) for _, k in calls if k]
        want = os.path.normcase(os.path.normpath(str(icon)))
        assert want in [os.path.normcase(os.path.normpath(p)) for p in paths]
    finally:
        try:
            wiz.win.destroy()
        except Exception:
            pass


# -- _set_step -----------------------------------------------------------------

def test_set_step_updates_wizard_states_and_tracker(wizard):
    wizard._set_step(1, "active")
    assert wizard._step_states == ["pending", "active", "pending"]
    assert wizard._tracker.states == ["pending", "active", "pending"]


def test_set_step_works_without_tracker_or_tk():
    wiz = setup_wizard.SetupWizard.__new__(setup_wizard.SetupWizard)
    wiz._set_step(2, "done")
    assert wiz._step_states == ["pending", "pending", "done"]
    wiz._set_step(0, "failed")
    assert wiz._step_states == ["failed", "pending", "done"]


def test_set_step_rejects_an_invalid_state(wizard):
    with pytest.raises(ValueError):
        wizard._set_step(0, "bogus")


def test_set_step_rejects_an_invalid_state_without_tracker():
    wiz = setup_wizard.SetupWizard.__new__(setup_wizard.SetupWizard)
    with pytest.raises(ValueError):
        wiz._set_step(0, "bogus")


def test_step_tracker_set_state_rejects_an_invalid_state(wizard):
    with pytest.raises(ValueError):
        wizard._tracker.set_state(0, "bogus")


# -- welcome state -------------------------------------------------------------

def test_step_names_are_python_speech_engine_finishing():
    assert tuple(setup_wizard.STEPS) == ("Python", "Speech engine", "Finishing")


def test_welcome_shows_the_tracker_with_all_steps_pending(wizard):
    assert isinstance(wizard._tracker, setup_wizard.StepTracker)
    assert wizard._tracker.states == ["pending", "pending", "pending"]


def test_welcome_has_exactly_one_primary_button_labelled_install(wizard):
    primary = [text for text, fg in _buttons(wizard.win) if _is_heat(fg)]
    assert primary == ["Install"]


def test_welcome_shows_gpu_python_and_download_rows(wizard):
    labels = _label_texts(wizard.win)
    for row in ("GPU", "Python", "Download"):
        assert row in labels


def test_missing_python_hides_the_install_button_and_activates_step_zero(wizard_no_python):
    assert "Install" not in [text for text, _ in _buttons(wizard_no_python.win)]
    assert wizard_no_python._tracker.states[0] == "active"


# -- theme: no blue, no light colours ------------------------------------------

@pytest.mark.parametrize("colour", ["#4a9eff", "#6ab0ff", "#f5f5f5", "#e8e8e8", "#f0f0f0"])
def test_old_blue_and_light_theme_colours_are_gone(colour):
    text = (ROOT / "setup_wizard.py").read_text(encoding="utf-8").lower()
    assert colour not in text
