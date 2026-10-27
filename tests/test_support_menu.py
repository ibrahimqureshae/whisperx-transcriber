"""Support menu at the foot of the rail: the project's star and donation links. Needs a display."""
import sys
import time
import tkinter
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

LINKS = {
    "https://github.com/ibrahimqureshae/whisperx-transcriber",
    "https://github.com/sponsors/ibrahimqureshae",
    "https://buymeacoffee.com/mibrahim99",
    "https://www.paypal.me/mibrahimqr",
}


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


@pytest.fixture
def opened(monkeypatch):
    import ui.shell as sh
    calls = []
    monkeypatch.setattr(sh.webbrowser, "open", lambda url, new=0: calls.append(url) or True)
    return calls


def _settle(shell, secs=0.3):
    """Pump Tk for real time so the ~120 ms open/close animation finishes."""
    end = time.monotonic() + secs
    while time.monotonic() < end:
        shell.update()
        time.sleep(0.01)


def _close(shell):
    if shell._support_open:
        shell._toggle_support()
    _settle(shell)


def test_links_are_the_project_links():
    import ui.shell as sh
    assert {url for _, _, url in sh.SUPPORT_LINKS} == LINKS
    assert [label for _, label, _ in sh.SUPPORT_LINKS][0] == "Star on GitHub"


def test_menu_starts_closed(shell):
    assert shell._support_open is False
    assert shell._support_menu.winfo_manager() == ""


def test_toggle_opens_then_closes(shell):
    shell._toggle_support(); _settle(shell)
    assert shell._support_open is True
    assert shell._support_menu.winfo_manager() == "grid"
    shell._toggle_support(); _settle(shell)
    assert shell._support_open is False
    assert shell._support_menu.winfo_manager() == ""


def test_each_row_opens_its_own_link_and_closes_the_menu(shell, opened):
    for row in shell._support_rows:
        shell._toggle_support(); _settle(shell)
        row.invoke(); _settle(shell)
        assert shell._support_open is False
    assert opened == [url for _, _, url in __import__("ui.shell", fromlist=["x"]).SUPPORT_LINKS]


def test_escape_closes_the_menu_without_cancelling(shell, monkeypatch):
    cancels = []
    monkeypatch.setattr(shell.app, "cancel", lambda: cancels.append(1))
    shell._toggle_support(); _settle(shell)
    shell._escape(None); _settle(shell)
    assert shell._support_open is False and cancels == []
    shell._escape(None)                         # menu closed: Esc cancels the run as before
    assert cancels == [1]


def test_click_away_closes_but_click_inside_does_not(shell):
    shell._toggle_support(); _settle(shell)
    shell._click_away(SimpleNamespace(widget=shell._support_rows[1].label))
    assert shell._support_open is True
    shell._click_away(SimpleNamespace(widget=shell._host))
    _settle(shell)
    assert shell._support_open is False
    _close(shell)


def test_menu_fits_in_the_rail_when_open(shell):
    shell._toggle_support(); _settle(shell)
    menu, rail, log = shell._support_menu, shell._support_menu.master, shell._nav["log"]
    assert menu.winfo_width() <= rail.winfo_width()           # positions relative to the rail (window is hidden)
    assert menu.winfo_y() + menu.winfo_height() <= shell._support_btn.winfo_y()
    assert menu.winfo_y() > log.master.winfo_y() + log.winfo_y() + log.winfo_height()
    _close(shell)
