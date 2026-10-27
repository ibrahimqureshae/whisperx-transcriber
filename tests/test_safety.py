"""Data-safety tests (H6): single-instance lock, no orphan runner, corrupt files kept,
save errors surfaced. No GUI, no torch."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core import settings
from core.jobqueue import JobQueue
from ui import state as st


# ── single-instance lock ──────────────────────────────────────────────────────

def test_lock_second_acquire_fails_until_released(tmp_path):
    first = settings.acquire_lock(str(tmp_path))
    assert first is not None
    assert settings.acquire_lock(str(tmp_path)) is None
    settings.release_lock(first)
    again = settings.acquire_lock(str(tmp_path))
    assert again is not None
    settings.release_lock(again)


def test_lock_creates_missing_base_dir_and_ignores_stale_file(tmp_path):
    base = tmp_path / "new" / "dir"
    h = settings.acquire_lock(str(base))
    assert h is not None
    settings.release_lock(h)
    assert (base / ".whisperx.lock").exists()       # stale file left behind
    h = settings.acquire_lock(str(base))
    assert h is not None
    settings.release_lock(h)


# ── stdin watchdog ────────────────────────────────────────────────────────────

_WATCHDOG_CHILD = """
import sys
sys.path.insert(0, {root!r})
from core import runner
sys.stdin.readline()
runner._start_stdin_watchdog()
open({flag!r}, "w").close()
import time
time.sleep(30)
"""


def test_runner_exits_when_stdin_closes(tmp_path):
    flag = tmp_path / "ready"
    script = tmp_path / "child.py"
    script.write_text(_WATCHDOG_CHILD.format(root=str(ROOT), flag=str(flag)))
    p = subprocess.Popen([sys.executable, str(script)], stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    try:
        p.stdin.write("{}\n")
        p.stdin.flush()
        for _ in range(100):
            if flag.exists():
                break
            time.sleep(0.1)
        assert flag.exists()
        assert p.poll() is None            # stdin open -> keeps running
        p.stdin.close()
        assert p.wait(timeout=2) == 3
    finally:
        if p.poll() is None:
            p.kill()


def test_runner_process_keeps_stdin_open(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERX_RUNNER", str(ROOT / "tests" / "fake_runner.py"))
    monkeypatch.setenv("FAKE_RUNNER_HOLD", "1")
    rp = st.RunnerProcess({"audio": str(tmp_path / "a.wav"), "out_dir": str(tmp_path)},
                          lambda ev: None, lambda rc: None)
    rp.start()
    try:
        assert not rp.proc.stdin.closed
    finally:
        rp.kill()
        rp.proc.wait(timeout=5)


# ── Job Object ────────────────────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects")
def test_job_close_kills_child_and_grandchild(tmp_path):
    import ctypes
    from ctypes import wintypes
    flag = tmp_path / "grandchild.pid"
    grand = "import time; time.sleep(60)"
    child = (
        "import subprocess, sys, time\n"
        f"g = subprocess.Popen([sys.executable, '-c', {grand!r}])\n"
        f"open({str(flag)!r}, 'w').write(str(g.pid))\n"
        "time.sleep(60)\n")
    # a private job (the app-wide one must live for the whole process)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    saved, st._job_handle = st._job_handle, None
    try:
        private = st._kill_on_close_job()
    finally:
        st._job_handle = saved
    assert private
    p = subprocess.Popen([sys.executable, "-c", child], creationflags=subprocess.CREATE_NO_WINDOW)
    st._assign_to_job(p, private)
    for _ in range(100):
        if flag.exists() and flag.read_text():
            break
        time.sleep(0.1)
    gpid = int(flag.read_text())
    k32.CloseHandle(private)
    assert p.wait(timeout=5) is not None
    out = subprocess.run(["tasklist", "/FI", f"PID eq {gpid}"],
                         capture_output=True, text=True).stdout
    for _ in range(20):
        if str(gpid) not in out:
            break
        time.sleep(0.2)
        out = subprocess.run(["tasklist", "/FI", f"PID eq {gpid}"],
                             capture_output=True, text=True).stdout
    assert str(gpid) not in out


# ── corrupt files are kept ────────────────────────────────────────────────────

def _corrupt_copies(folder, name):
    return [p for p in Path(folder).iterdir() if p.name.startswith(name + ".corrupt-")]


def test_corrupt_settings_renamed_and_noticed(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    assert settings.load(str(path)) == settings.DEFAULTS
    assert not path.exists()
    kept = _corrupt_copies(tmp_path, "settings.json")
    assert len(kept) == 1 and kept[0].read_text() == "{not json"
    assert settings.last_load_notice == (
        f"settings.json was unreadable; kept a copy as {kept[0].name} "
        "and started from defaults")


def test_wrong_toplevel_settings_renamed(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("[1, 2]", encoding="utf-8")
    settings.load(str(path))
    assert len(_corrupt_copies(tmp_path, "settings.json")) == 1


def test_good_and_missing_settings_leave_no_notice(tmp_path):
    settings.load(str(tmp_path / "settings.json"))
    assert settings.last_load_notice is None
    (tmp_path / "settings.json").write_text("{}", encoding="utf-8")
    settings.load(str(tmp_path / "settings.json"))
    assert settings.last_load_notice is None
    assert _corrupt_copies(tmp_path, "settings.json") == []


def test_corrupt_queue_renamed_and_noticed(tmp_path):
    path = tmp_path / "queue.json"
    path.write_text("garbage", encoding="utf-8")
    assert JobQueue.load(str(path)).jobs == []
    assert not path.exists()
    assert len(_corrupt_copies(tmp_path, "queue.json")) == 1
    assert JobQueue.last_load_notice.startswith("queue.json was unreadable")


def test_wrong_toplevel_queue_renamed(tmp_path):
    path = tmp_path / "queue.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    JobQueue.load(str(path))
    assert len(_corrupt_copies(tmp_path, "queue.json")) == 1


def test_non_dict_cfg_is_coerced_and_round_trips(tmp_path):
    q = JobQueue.from_json([{"id": "a", "audio": "x.wav", "cfg": 5}])
    assert q.jobs[0].cfg == {}
    q.jobs[0].cfg = 7                       # odd cfg set after load
    assert q.to_json()[0]["cfg"] == {}
    path = tmp_path / "queue.json"
    assert q.save(str(path)) is None
    assert JobQueue.load(str(path)).jobs[0].cfg == {}


# ── AppState: base dir, notices ───────────────────────────────────────────────

def test_appstate_creates_base_dir(tmp_path):
    base = tmp_path / "a" / "b"
    st.AppState(lambda fn: fn(), base=str(base))
    assert base.is_dir()


def test_appstate_emits_load_notice(tmp_path):
    (tmp_path / "settings.json").write_text("nope", encoding="utf-8")
    pending = []
    app = st.AppState(pending.append, base=str(tmp_path))
    got = []
    app.on("notice", lambda text: got.append(text))
    for fn in pending:
        fn()
    assert len(got) == 1 and got[0].startswith("settings.json was unreadable")


def test_save_failure_emits_throttled_notice(tmp_path):
    app = st.AppState(lambda fn: fn(), base=str(tmp_path))
    got = []
    app.on("notice", lambda text: got.append(text))
    app.settings_path = str(tmp_path / "missing" / "settings.json")
    app.queue_path = str(tmp_path / "missing" / "queue.json")
    app.save_settings()
    app.save_settings()
    assert len(got) == 1 and got[0].startswith("Could not save settings")
    app.queue.save(app.queue_path)
    assert len(got) == 2 and got[1].startswith("Could not save the queue")
    app._notice_times[got[0]] -= 61          # a minute later it may repeat
    app.save_settings()
    assert len(got) == 3


# ── Cancel kills the whole venv-launcher tree ─────────────────────────────────

def _children(pid):
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}').ProcessId"],
        capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def _alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                         capture_output=True, text=True).stdout
    return str(pid) in out


@pytest.mark.skipif(sys.platform != "win32", reason="venv launcher tree")
def test_cancel_kills_launcher_and_base_interpreter(tmp_path, monkeypatch):
    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setenv("WHISPERX_RUNNER", str(script))
    app = st.AppState(lambda fn: fn(), base=str(tmp_path))
    job = app.add_files([str(tmp_path / "a.wav")])[0]
    assert app.run_job(job)
    root = app._proc.proc.pid
    kids = []
    for _ in range(40):                      # venv launcher -> base interpreter
        kids = _children(root)
        if kids or sys.prefix == sys.base_prefix:
            break
        time.sleep(0.25)
    assert kids or sys.prefix == sys.base_prefix
    app.cancel()
    deadline = time.time() + 3
    while time.time() < deadline and (app.running is not None or _alive(root)
                                      or any(_alive(k) for k in kids)):
        time.sleep(0.1)
    assert app.running is None and job.status == "cancelled"
    assert not _alive(root) and not any(_alive(k) for k in kids)
