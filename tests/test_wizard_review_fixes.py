"""Small fixes after the T3 cross-review: empty bar at start, metadata lines, error wording."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import setup_wizard

ROOT = str(Path(__file__).parent.parent)


def test_metadata_downloads_are_not_shown_as_the_package():
    line = "  Downloading whisperx-3.8.6-py3-none-any.whl.metadata (12 kB)"
    assert setup_wizard.parse_pip_detail(line) is None


def test_wheel_downloads_still_show_the_package():
    line = "  Downloading whisperx-3.8.6-py3-none-any.whl (2.1 MB)"
    assert setup_wizard.parse_pip_detail(line) == "whisperx 3.8.6 \u00b7 2.1 MB"


def test_a_new_progress_card_starts_empty(monkeypatch):
    monkeypatch.setattr(setup_wizard, "_has_nvidia", lambda: False)
    monkeypatch.setattr(setup_wizard, "_cuda_torch_index", lambda: None)
    monkeypatch.setattr(setup_wizard, "_find_python", lambda: "C:/fake/python.exe")
    monkeypatch.setattr(setup_wizard, "_python_minor", lambda path: "3.11")
    wiz = setup_wizard.SetupWizard(ROOT, on_complete=lambda: None)
    try:
        wiz._show_installing()
        assert wiz._prog.get() == 0
    finally:
        wiz.win.destroy()


def test_error_reasons_do_not_point_at_a_log_above():
    src = Path(ROOT, "setup_wizard.py").read_text(encoding="utf-8")
    assert "see the log above" not in src
