"""Launcher one-time drag-and-drop and speaker-package installs: marker rules, timeouts, splash. No real pip."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import launcher


@pytest.fixture
def rt(tmp_path, monkeypatch):
    (tmp_path / "runtime" / "Scripts").mkdir(parents=True)
    (tmp_path / "runtime" / "Scripts" / "python.exe").write_bytes(b"")
    monkeypatch.setattr(launcher, "_base_dir", lambda: str(tmp_path))
    splashes = []
    monkeypatch.setattr(launcher, "_splash", lambda text: splashes.append(text))
    calls = []

    def fake_run(cmd, timeout=None, **kw):
        calls.append((cmd, timeout))
        return subprocess.CompletedProcess(cmd, fake_run.codes.pop(0))
    fake_run.codes = []
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    return tmp_path / "runtime" / ".dnd_checked", fake_run, calls, splashes


def test_success_writes_marker_and_shows_splash(rt):
    marker, run, calls, splashes = rt
    run.codes = [1, 0]                       # import fails, pip succeeds
    launcher._ensure_dnd()
    assert marker.read_text() == "done"
    assert splashes == ["Adding drag-and-drop support…"]
    pip = calls[1]
    assert "pip" in pip[0] and pip[0][pip[0].index("--timeout") + 1] == "60" and pip[1] <= 60
    assert calls[0][1] + pip[1] <= launcher.DND_BUDGET


def test_already_installed_needs_no_pip_or_splash(rt):
    marker, run, calls, splashes = rt
    run.codes = [0]
    launcher._ensure_dnd()
    assert len(calls) == 1 and not splashes and marker.read_text() == "done"


def test_marker_after_three_failures_only(rt):
    marker, run, calls, _ = rt
    for n in (1, 2, 3):
        run.codes = [1, 1]
        launcher._ensure_dnd()
        assert marker.read_text() == str(n)
    n_calls = len(calls)
    launcher._ensure_dnd()                   # given up for good: nothing runs
    assert len(calls) == n_calls


def test_legacy_empty_marker_is_final(rt):
    marker, run, calls, _ = rt
    marker.write_text("")
    launcher._ensure_dnd()
    assert not calls


def test_timeout_counts_as_a_failure(rt, monkeypatch):
    marker, run, calls, _ = rt

    def boom(cmd, timeout=None, **kw):
        raise subprocess.TimeoutExpired(cmd, timeout)
    monkeypatch.setattr(launcher.subprocess, "run", boom)
    launcher._ensure_dnd()
    assert marker.read_text() == "1"


def test_speakers_package_installs_once_with_its_own_marker(rt):
    marker, run, calls, splashes = rt
    run.codes = [1, 0]                       # import fails, pip succeeds
    launcher._ensure_speakers()
    sherpa_marker = marker.parent / ".sherpa_checked"
    assert sherpa_marker.read_text() == "done" and not marker.exists()
    assert splashes == ["Adding speaker identification…"]
    assert calls[0][0][-1] == "import sherpa_onnx"
    assert "sherpa-onnx>=1.13" in calls[1][0]
    launcher._ensure_speakers()              # marker present: nothing runs again
    assert len(calls) == 2


def test_rtl_packages_install_once_with_both_specs_and_own_marker(rt):
    marker, run, calls, splashes = rt
    run.codes = [1, 0]                       # import fails, one pip call succeeds
    launcher._ensure_rtl()
    rtl_marker = marker.parent / ".rtl_checked"
    assert rtl_marker.read_text() == "done" and not marker.exists()
    assert splashes == ["Adding right-to-left text support…"]
    assert calls[0][0][-1] == "import bidi, arabic_reshaper"
    pip = calls[1][0]
    assert "python-bidi>=0.6" in pip and "arabic-reshaper>=3.0" in pip
    assert len(calls) == 2
    launcher._ensure_rtl()                   # marker present: nothing runs again
    assert len(calls) == 2


def test_rtl_already_installed_needs_no_pip(rt):
    marker, run, calls, splashes = rt
    run.codes = [0]
    launcher._ensure_rtl()
    assert len(calls) == 1 and not splashes
    assert (marker.parent / ".rtl_checked").read_text() == "done"


def test_rtl_marker_after_three_failures_only(rt):
    marker, run, calls, _ = rt
    rtl_marker = marker.parent / ".rtl_checked"
    for n in (1, 2, 3):
        run.codes = [1, 1]
        launcher._ensure_rtl()
        assert rtl_marker.read_text() == str(n)
    n_calls = len(calls)
    launcher._ensure_rtl()
    assert len(calls) == n_calls


def test_ensure_pkg_accepts_a_list_of_specs(rt):
    marker, run, calls, _ = rt
    run.codes = [1, 0]
    launcher._ensure_pkg("somemod", ["pkg-a>=1", "pkg-b>=2"], ".multi_checked", "Adding things…")
    pip = calls[1][0]
    assert "pkg-a>=1" in pip and "pkg-b>=2" in pip
    assert (marker.parent / ".multi_checked").read_text() == "done"


def test_main_ensures_rtl_on_an_existing_runtime(monkeypatch):
    seen = []
    monkeypatch.setattr(launcher, "_runtime_ready", lambda: True)
    monkeypatch.setattr(launcher, "_ensure_dnd", lambda: seen.append("dnd"))
    monkeypatch.setattr(launcher, "_ensure_speakers", lambda: seen.append("speakers"))
    monkeypatch.setattr(launcher, "_ensure_rtl", lambda: seen.append("rtl"), raising=False)
    monkeypatch.setattr(launcher, "_launch_app", lambda: seen.append("launch"))
    launcher.main()
    assert "rtl" in seen and seen.index("rtl") < seen.index("launch")
