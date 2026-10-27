"""Python readout when the found Python's version can't be read."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import setup_wizard

ROOT = str(Path(__file__).parent.parent)


def test_unknown_python_version_leaves_the_readout_blank(monkeypatch):
    monkeypatch.setattr(setup_wizard, "_has_nvidia", lambda: False)
    monkeypatch.setattr(setup_wizard, "_cuda_torch_index", lambda: None)
    monkeypatch.setattr(setup_wizard, "_find_python", lambda: "C:/fake/python.exe")
    monkeypatch.setattr(setup_wizard, "_python_minor", lambda path: None)
    wiz = setup_wizard.SetupWizard(ROOT, on_complete=lambda: None)
    try:
        assert wiz._tracker._details[0] == ""
    finally:
        wiz.win.destroy()


def test_missing_python_shows_the_version_that_will_be_installed(monkeypatch):
    monkeypatch.setattr(setup_wizard, "_has_nvidia", lambda: False)
    monkeypatch.setattr(setup_wizard, "_cuda_torch_index", lambda: None)
    monkeypatch.setattr(setup_wizard, "_find_python", lambda: None)
    wiz = setup_wizard.SetupWizard(ROOT, on_complete=lambda: None)
    try:
        assert wiz._tracker._details[0] == "3.13"
    finally:
        wiz.win.destroy()
