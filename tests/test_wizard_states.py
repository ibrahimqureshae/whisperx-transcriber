"""Tests for the install, done and error states of the Setup window in setup_wizard.py."""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import customtkinter as ctk
import pytest

# Make sure the project root is on the path when running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import setup_wizard

ROOT = Path(__file__).parent.parent
HEAT = "#ff6a1a"
PYTHON_EXE = "C:/fake/python.exe"
CU128 = "https://download.pytorch.org/whl/cu128"

# "1.2 GB / 2.5 GB" style size readout
_UNIT = r"(?:B|kB|KB|MB|GB)"
SIZES_RE = re.compile(rf"\d+(?:\.\d+)?\s*{_UNIT}\s*/\s*\d+(?:\.\d+)?\s*{_UNIT}")
PERCENT_48_RE = re.compile(r"(?<!\d)48\s*%")


# -- helpers: real window ------------------------------------------------------

def _walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from _walk(child)


def _label_texts(win):
    """Displayed text of every CTkLabel, including labels driven by a StringVar."""
    texts = []
    for w in _walk(win):
        if not isinstance(w, ctk.CTkLabel):
            continue
        texts.append(str(w.cget("text")))
        try:
            var = w.cget("textvariable")
            if var:
                texts.append(str(w.getvar(str(var))))
        except Exception:
            pass
    return texts


def _button_texts(win):
    return [w.cget("text") for w in _walk(win) if isinstance(w, ctk.CTkButton)]


def _is_heat(color):
    values = color if isinstance(color, (tuple, list)) else (color,)
    return any(str(v).lower() == HEAT for v in values)


def _heat_buttons(win):
    return [w for w in _walk(win)
            if isinstance(w, ctk.CTkButton) and _is_heat(w.cget("fg_color"))]


def _textboxes(win):
    return [w for w in _walk(win) if isinstance(w, ctk.CTkTextbox)]


def _mapped_textboxes(win):
    win.update()
    return [w for w in _textboxes(win) if w.winfo_ismapped()]


def _make_wizard(monkeypatch, on_complete=None):
    monkeypatch.setattr(setup_wizard, "_has_nvidia", lambda: False)
    monkeypatch.setattr(setup_wizard, "_cuda_torch_index", lambda: None)
    monkeypatch.setattr(setup_wizard, "_find_python", lambda: PYTHON_EXE)
    return setup_wizard.SetupWizard(str(ROOT), on_complete or (lambda: None))


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
def installing(wizard):
    wizard._show_installing()
    wizard.win.update()
    return wizard


# -- helpers: no-Tk harness ----------------------------------------------------

def _immediate_win():
    """A fake window: after(0, fn) runs fn at once, delayed callbacks are dropped."""
    def after(ms, fn=None, *args):
        if fn is not None and ms == 0:
            fn(*args)
    return SimpleNamespace(after=after)


def _bare_wizard():
    wiz = setup_wizard.SetupWizard.__new__(setup_wizard.SetupWizard)
    wiz.win = _immediate_win()
    wiz._step_states = ["pending"] * 3
    wiz._log = wiz._set_status = lambda *a, **k: None
    wiz._set_progress = lambda *a, **k: None
    return wiz


def _install_harness(monkeypatch, tmp_path, cuda_index):
    """Run SetupWizard._run_install (fresh install) with pip faked; record step states."""
    (tmp_path / "Scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "Scripts" / "python.exe").write_bytes(b"")
    wiz = _bare_wizard()
    wiz.python_path, wiz.runtime_dir = "python", str(tmp_path)
    wiz._show_done = lambda: None
    wiz._show_error = lambda msg: None
    wiz._pkg_installed = lambda venv_pip, pkg: False
    seen = {}

    def fake_pip(cmd):
        if "--upgrade" in cmd:
            seen["before_pip_upgrade"] = list(wiz._step_states)
        if "whisperx" in cmd and "install" in cmd:
            seen["at_main_install"] = list(wiz._step_states)
        return 0

    def fake_check_gpu(venv_py):
        seen["at_finishing_checks"] = list(wiz._step_states)

    wiz._run_pip = fake_pip
    wiz._check_gpu = fake_check_gpu
    monkeypatch.setattr(setup_wizard, "_cuda_torch_index", lambda: cuda_index)
    monkeypatch.setattr(setup_wizard.subprocess, "run",
                        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="12\n", stderr=""))
    wiz._run_install()
    return wiz, seen


# -- steps advance: Python install ---------------------------------------------

def test_python_install_marks_step_zero_active_while_running_and_done_on_success(monkeypatch):
    wiz = _bare_wizard()
    wiz.python_path = None
    wiz._set_py_status = lambda *a, **k: None
    wiz._set_py_progress = lambda *a, **k: None
    wiz._show_py_error = lambda msg: pytest.fail(f"unexpected error page: {msg}")
    seen = {}

    def fake_urlretrieve(url, dest, reporthook=None):
        seen["during_download"] = list(wiz._step_states)

    def fake_run(cmd, **kw):
        seen["during_installer"] = list(wiz._step_states)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(setup_wizard.urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(setup_wizard.subprocess, "run", fake_run)
    monkeypatch.setattr(setup_wizard, "_find_python_fresh", lambda: PYTHON_EXE)

    wiz._run_python_install()

    assert seen["during_download"][0] == "active"
    assert seen["during_installer"][0] == "active"
    assert wiz._step_states[0] == "done"


# -- steps advance: engine install ---------------------------------------------

def test_install_starts_with_python_done_and_the_engine_step_active(monkeypatch, tmp_path):
    _, seen = _install_harness(monkeypatch, tmp_path, cuda_index=CU128)
    assert seen["before_pip_upgrade"][:2] == ["done", "active"]


def test_main_engine_install_runs_with_python_done_and_engine_active(monkeypatch, tmp_path):
    _, seen = _install_harness(monkeypatch, tmp_path, cuda_index=CU128)
    assert seen["at_main_install"] == ["done", "active", "pending"]


def test_finishing_checks_run_with_engine_done_and_finishing_active(monkeypatch, tmp_path):
    _, seen = _install_harness(monkeypatch, tmp_path, cuda_index=CU128)
    assert seen["at_finishing_checks"] == ["done", "done", "active"]


@pytest.mark.parametrize("cuda_index", [None, CU128])
def test_successful_install_marks_all_three_steps_done(monkeypatch, tmp_path, cuda_index):
    wiz, _ = _install_harness(monkeypatch, tmp_path, cuda_index=cuda_index)
    assert wiz._step_states == ["done", "done", "done"]


# -- installing state: title and old texts -------------------------------------

def test_installing_state_has_the_setting_up_title(installing):
    assert installing._title.cget("text") == "Setting up WhisperX"


def test_installing_state_drops_the_old_heading_and_footer_hint(installing):
    texts = _label_texts(installing.win) + _button_texts(installing.win)
    assert not [t for t in texts if "Installing..." in t or "Please wait" in t]


def test_python_installing_state_drops_the_old_footer_hint_and_activates_step_zero(wizard):
    wizard._show_python_installing()
    wizard.win.update()
    texts = _label_texts(wizard.win) + _button_texts(wizard.win)
    assert not [t for t in texts if "Installing..." in t or "Please wait" in t]
    assert wizard._title.cget("text") == "Setting up WhisperX"
    assert wizard._step_states[0] == "active"


# -- progress readout ----------------------------------------------------------

def test_progress_tuple_shows_sizes_percent_and_speed(installing):
    installing._set_install_progress((1_200_000_000, 2_500_000_000, 8_100_000.0))
    installing.win.update()
    joined = "\n".join(_label_texts(installing.win))
    assert re.search(r"1\.2 GB\s*/\s*2\.5 GB", joined)
    assert PERCENT_48_RE.search(joined)
    assert re.search(r"8\.1 MB/s", joined)
    assert installing._prog.cget("mode") == "determinate"


def test_install_phase_line_switches_to_the_indeterminate_sweep(installing):
    installing._set_install_progress((1_200_000_000, 2_500_000_000, 8_100_000.0))
    installing._set_install_indeterminate()
    installing.win.update()
    assert installing._prog.cget("mode") == "indeterminate"


def test_install_phase_line_blanks_the_size_row(installing):
    installing._set_install_progress((1_200_000_000, 2_500_000_000, 8_100_000.0))
    installing.win.update()
    assert [t for t in _label_texts(installing.win) if SIZES_RE.search(t)]
    installing._set_install_indeterminate()
    installing.win.update()
    texts = _label_texts(installing.win)
    assert not [t for t in texts if SIZES_RE.search(t)]
    assert not [t for t in texts if PERCENT_48_RE.search(t)]
    assert not [t for t in texts if "MB/s" in t]


# -- details log ---------------------------------------------------------------

def test_log_is_hidden_by_default(installing):
    assert installing._details_open is False
    assert _mapped_textboxes(installing.win) == []
    assert any("Show details" in t for t in _label_texts(installing.win) + _button_texts(installing.win))


def test_toggle_details_shows_and_hides_the_log(installing):
    installing._log("pip says hello")
    installing._drain_log()

    installing._toggle_details()
    shown = _mapped_textboxes(installing.win)
    assert installing._details_open is True
    assert len(shown) == 1
    assert "pip says hello" in shown[0].get("1.0", "end")
    assert any("Hide details" in t for t in _label_texts(installing.win) + _button_texts(installing.win))

    installing._toggle_details()
    assert installing._details_open is False
    assert _mapped_textboxes(installing.win) == []
    assert any("Show details" in t for t in _label_texts(installing.win) + _button_texts(installing.win))


def test_log_lines_still_arrive_while_the_panel_is_hidden(installing):
    assert installing._details_open is False
    before = len(installing._log_lines)
    installing._log("first line")
    installing._log("second line")
    installing._log("━━━━━━━━ 1.0/2.0 MB 1.0 MB/s eta 0:00:01")
    installing._drain_log()
    assert installing._log_lines[before:] == ["first line", "second line"]
    assert _mapped_textboxes(installing.win) == []


# -- done state ----------------------------------------------------------------

def test_done_shows_the_ready_title(wizard):
    wizard._show_done()
    wizard.win.update()
    assert wizard._title.cget("text") == "Ready"


def test_done_has_exactly_one_primary_button_open_whisperx(wizard):
    wizard._show_done()
    wizard.win.update()
    assert [b.cget("text") for b in _heat_buttons(wizard.win)] == ["Open WhisperX"]


def test_open_whisperx_calls_on_complete(monkeypatch):
    calls = []
    wiz = _make_wizard(monkeypatch, on_complete=lambda: calls.append("done"))
    try:
        wiz._show_done()
        wiz.win.update()
        (button,) = _heat_buttons(wiz.win)
        button.invoke()
        assert calls == ["done"]
    finally:
        try:
            wiz.win.destroy()
        except Exception:
            pass


# -- error state ---------------------------------------------------------------

def test_error_marks_the_active_step_failed(wizard):
    wizard._set_step(0, "done")
    wizard._set_step(1, "active")
    wizard._show_error("pip exploded")
    assert wizard._step_states == ["done", "failed", "pending"]


def test_error_with_no_active_step_marks_the_engine_step_failed(wizard):
    wizard._show_error("pip exploded")
    assert wizard._step_states == ["pending", "failed", "pending"]


def test_error_shows_the_reason_and_the_stopped_title(wizard):
    wizard._show_error("pip exploded: no network")
    wizard.win.update()
    shown = _label_texts(wizard.win) + [tb.get("1.0", "end") for tb in _textboxes(wizard.win)]
    assert wizard._title.cget("text") == "Setup stopped"
    assert any("pip exploded: no network" in t for t in shown)


def test_error_has_exactly_one_primary_button_try_again(wizard):
    wizard._show_error("pip exploded")
    wizard.win.update()
    assert [b.cget("text") for b in _heat_buttons(wizard.win)] == ["Try again"]


def test_python_error_marks_step_zero_failed_and_offers_try_again(wizard):
    wizard._show_py_error("download failed")
    wizard.win.update()
    shown = _label_texts(wizard.win) + [tb.get("1.0", "end") for tb in _textboxes(wizard.win)]
    assert wizard._title.cget("text") == "Setup stopped"
    assert wizard._step_states[0] == "failed"
    assert any("download failed" in t for t in shown)
    assert [b.cget("text") for b in _heat_buttons(wizard.win)] == ["Try again"]
