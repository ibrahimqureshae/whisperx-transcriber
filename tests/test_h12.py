"""H12 review fixes: translate reset, cancel races, unreadable queue/settings files,
atomic settings save, non-blocking kill, speaker renames on re-run, punkt prefetch."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import settings
from core.jobqueue import JobQueue
from ui import state as st
from tests.test_state import _Pump, _settings


# ── 1. translate reset lives in AppState ──────────────────────────────────────

def test_model_change_resets_translate_without_settings_screen(tmp_path):
    s = st.AppState(_Pump(), base=str(tmp_path))
    s.set("model", "large-v3")
    s.set("task", "translate")
    s.set("model", "large-v3-turbo")
    assert s.settings["task"] == "transcribe"
    assert json.loads((tmp_path / "settings.json").read_text())["task"] == "transcribe"


def test_model_change_keeps_translate_when_supported(tmp_path):
    s = st.AppState(_Pump(), base=str(tmp_path))
    s.set("model", "large-v3")
    s.set("task", "translate")
    s.set("model", "medium")
    assert s.settings["task"] == "translate"


# ── 2. cancel races ───────────────────────────────────────────────────────────

class _Proc:
    def __init__(self, alive):
        self.alive = alive

    def kill(self, block=False):
        return self.alive


def test_cancel_after_exit_does_not_flag_cancelled(tmp_path):
    s = st.AppState(_Pump(), base=str(tmp_path))
    s._proc = _Proc(False)
    s.cancel()
    assert s._cancelled is False
    s._proc = _Proc(True)
    s.cancel()
    assert s._cancelled is True


_DONE_THEN_HANG = """import json, sys, time
sys.stdin.readline()
print(json.dumps({"t": "stage", "name": "export", "state": "done", "files": ["a.srt"]}), flush=True)
print(json.dumps({"t": "done", "path": "a.srt"}), flush=True)
time.sleep(30)
"""


def test_cancel_after_done_event_still_ends_done(tmp_path, monkeypatch):
    script = tmp_path / "r.py"
    script.write_text(_DONE_THEN_HANG, encoding="utf-8")
    monkeypatch.setenv("WHISPERX_RUNNER", str(script))
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    done = []
    s.on("finished", lambda job: done.append(job))
    job = s.add_files([str(tmp_path / "x.wav")])[0]
    s.run_job(job)
    assert pump.drain_until(lambda: s._ended)
    s.cancel()
    assert pump.drain_until(lambda: done)
    assert job.status == "done" and job.outputs == ["a.srt"]


# ── 3. unreadable queue.json ──────────────────────────────────────────────────

def test_unreadable_queue_is_not_overwritten_or_swept(tmp_path):
    (tmp_path / "queue.json").mkdir()                    # opening a directory -> OSError
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "keep.json").write_text("[{}]")
    errors = []
    s = st.AppState(_Pump(), base=str(tmp_path))
    s.queue.on_save_error = errors.append
    assert JobQueue.last_load_notice.startswith("queue.json could not be read: ")
    assert (tmp_path / "transcripts" / "keep.json").exists()
    s.queue.save(s.queue_path)
    s.queue.save(s.queue_path)
    assert (tmp_path / "queue.json").is_dir()
    assert len(errors) == 1


def test_missing_queue_is_just_empty(tmp_path):
    q = JobQueue.load(str(tmp_path / "queue.json"))
    assert q.jobs == [] and JobQueue.last_load_notice is None and not q.save_blocked


# ── 4. atomic settings save ───────────────────────────────────────────────────

def test_settings_save_failure_keeps_old_file(tmp_path, monkeypatch):
    path = str(tmp_path / "settings.json")
    assert settings.save(path, {"model": "medium"}) is None
    before = Path(path).read_text()

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(json, "dump", boom)
    assert "disk full" in settings.save(path, {"model": "small"})
    assert Path(path).read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == ["settings.json"]


# ── 5. quarantine failure ─────────────────────────────────────────────────────

def test_quarantine_failure_does_not_claim_a_copy(tmp_path, monkeypatch):
    p = tmp_path / "queue.json"
    p.write_text("x")

    def locked(*a):
        raise OSError("locked")
    monkeypatch.setattr(os, "replace", locked)
    msg = settings.quarantine(str(p))
    assert "could not be set aside" in msg and "kept a copy" not in msg


# ── 6. kill does not block ────────────────────────────────────────────────────

def test_kill_returns_quickly_while_taskkill_is_slow(monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(st, "IS_WIN", True)
    monkeypatch.setattr(st.subprocess, "run", lambda *a, **k: gate.wait(5))
    rp = st.RunnerProcess({}, lambda ev: None, lambda rc: None)
    rp.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               stdin=subprocess.PIPE)
    try:
        t0 = time.monotonic()
        assert rp.kill() is True
        assert time.monotonic() - t0 < 1
    finally:
        gate.set()
        rp.proc.wait(10)
    assert rp.kill() is False


# ── 7. from_json coercion ─────────────────────────────────────────────────────

def test_from_json_coerces_bad_fields():
    q = JobQueue.from_json([
        {"id": "a", "audio": "x.mp4", "status": "bogus", "duration": "7", "progress": None,
         "started": True, "finished": {}, "outputs": ["ok", 3], "cfg": 5},
        {"id": "b", "audio": "y.mp4", "cfg": {}, "status": "done", "duration": 3.5, "outputs": ["o.srt"]}])
    a, b = q.jobs
    assert (a.status, a.error) == ("failed", "Unknown status")
    assert a.duration is None and a.progress == 0.0 and a.started is None and a.finished is None
    assert a.outputs == [] and a.cfg == {}
    assert b.status == "done" and b.duration == 3.5 and b.outputs == ["o.srt"]


# ── 8. speaker_names_in_cues ──────────────────────────────────────────────────

def test_build_cfg_passes_speaker_names_in_cues(tmp_path):
    a = str(tmp_path / "a.wav")
    assert st.build_cfg(_settings(speaker_names_in_cues=False), a)["speaker_names_in_cues"] is False
    assert st.build_cfg(_settings(), a)["speaker_names_in_cues"] is True


# ── 9. renames survive a re-run ───────────────────────────────────────────────

def test_relaunch_keeps_speaker_names(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERX_RUNNER", str(Path(__file__).parent / "fake_runner.py"))
    monkeypatch.setenv("FAKE_RUNNER_SECONDS", "0.4")
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    done = []
    s.on("finished", lambda job: done.append(job))
    job = s.add_files([str(tmp_path / "x.wav")])[0]
    job.cfg["speaker_names"] = {"SPEAKER_00": "Ana"}
    s.run_job(job)
    assert job.cfg["speaker_names"] == {"SPEAKER_00": "Ana"}
    assert pump.drain_until(lambda: done)


# ── 10. punkt prefetch ────────────────────────────────────────────────────────

def test_whisper_download_also_fetches_punkt(monkeypatch):
    pytest.importorskip("customtkinter")
    from ui import models
    calls = []
    monkeypatch.setattr(models.pipeline, "ensure_punkt",
                        lambda md, offline: calls.append((md, offline)))
    models._prefetch_punkt("/md")
    assert calls == [("/md", False)]

    def net(*a, **k):
        raise RuntimeError("net")
    monkeypatch.setattr(models.pipeline, "ensure_punkt", net)
    models._prefetch_punkt("/md")                       # best-effort: never raises
