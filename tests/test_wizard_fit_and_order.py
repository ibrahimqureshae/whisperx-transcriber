"""Final-review fixes: step order on an existing install, and the card fitting with details open."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import setup_wizard

ROOT = str(Path(__file__).parent.parent)


def _immediate_win():
    """after(0, fn) runs fn now; delayed callbacks are dropped."""
    def after(ms, fn=None, *args):
        if fn is not None and ms == 0:
            fn(*args)
    return SimpleNamespace(after=after)


# -- existing install: speakers and right-to-left packages belong to "Finishing" --

def test_existing_install_adds_speakers_and_rtl_under_the_finishing_step(monkeypatch, tmp_path):
    (tmp_path / "Scripts").mkdir()
    (tmp_path / "Scripts" / "python.exe").write_bytes(b"")
    wiz = setup_wizard.SetupWizard.__new__(setup_wizard.SetupWizard)
    wiz.win = _immediate_win()
    wiz._step_states = ["pending"] * 3
    wiz._log = wiz._set_status = lambda *a, **k: None
    wiz._set_progress = lambda *a, **k: None
    wiz._show_done = lambda: None
    wiz._show_error = lambda msg: pytest.fail(f"unexpected error page: {msg}")
    wiz.python_path, wiz.runtime_dir = "python", str(tmp_path)
    installed = {"whisperx", "tkinterdnd2", "imageio-ffmpeg"}
    wiz._pkg_installed = lambda venv_pip, pkg: pkg in installed
    wiz._check_gpu = lambda venv_py: None
    seen = {}

    def fake_pip(cmd):
        if any(str(c).startswith("sherpa-onnx") for c in cmd):
            seen["speakers"] = list(wiz._step_states)
        if any(str(c).startswith("python-bidi") for c in cmd):
            seen["rtl"] = list(wiz._step_states)
        return 0

    wiz._run_pip = fake_pip
    monkeypatch.setattr(setup_wizard, "_cuda_torch_index", lambda: None)
    monkeypatch.setattr(setup_wizard.subprocess, "run",
                        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="12\n", stderr=""))
    wiz._run_install()
    assert seen["speakers"] == ["done", "done", "active"]
    assert seen["rtl"] == ["done", "done", "active"]
    assert wiz._step_states == ["done", "done", "done"]


# -- the card fits its space with details open --------------------------------

@pytest.fixture
def wiz(monkeypatch):
    monkeypatch.setattr(setup_wizard, "_has_nvidia", lambda: True)
    monkeypatch.setattr(setup_wizard, "_cuda_torch_index", lambda: "https://download.pytorch.org/whl/cu128")
    monkeypatch.setattr(setup_wizard, "_find_python", lambda: "C:/fake/python.exe")
    monkeypatch.setattr(setup_wizard, "_python_minor", lambda path: "3.11")
    w = setup_wizard.SetupWizard(ROOT, on_complete=lambda: None)
    try:
        yield w
    finally:
        try:
            w.win.destroy()
        except Exception:
            pass


def _card_fits(w):
    w.win.update()
    card = w.body.winfo_children()[0]
    return card.winfo_reqheight() <= w.body.winfo_height(), (card.winfo_reqheight(), w.body.winfo_height())


def test_installing_card_fits_with_details_open(wiz):
    wiz._show_installing()
    wiz._set_install_progress((1_200_000_000, 2_500_000_000, 8_100_000.0))
    wiz._toggle_details()
    fits, sizes = _card_fits(wiz)
    assert fits, f"card needs {sizes[0]} px, body has {sizes[1]} px"


def test_error_card_fits_with_details_open(wiz):
    wiz._show_installing()
    wiz._log_lines = [f"line {i}" for i in range(60)]
    wiz._show_error("pip couldn't install the speech engine.")
    wiz._toggle_details()
    fits, sizes = _card_fits(wiz)
    assert fits, f"card needs {sizes[0]} px, body has {sizes[1]} px"


def test_closing_details_restores_the_progress_rows(wiz):
    wiz._show_installing()
    wiz._set_install_progress((1_200_000_000, 2_500_000_000, 8_100_000.0))
    wiz._toggle_details()
    wiz._toggle_details()
    wiz.win.update()
    assert wiz._prog.winfo_ismapped()
    assert "2.5 GB" in wiz._pct_lbl.cget("text")
