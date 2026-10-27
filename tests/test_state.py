"""Tests for ui/state.py: settings -> runner cfg mapping, and the job lifecycle
with tests/fake_runner.py as the runner. No GUI."""
import copy
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.settings import DEFAULTS
from ui import state as st


def _settings(**over):
    s = copy.deepcopy(DEFAULTS)
    s.update(over)
    return s


# ── decode_options ────────────────────────────────────────────────────────────

def test_decode_options_blank_is_empty():
    assert st.decode_options(DEFAULTS["decode"]) == {}


def test_decode_options_types_values_and_skips_bad_ones():
    out = st.decode_options({"temperatures": "0, 0.2", "best_of": "3", "patience": "1.5",
                             "condition_on_previous_text": "True", "max_new_tokens": "x",
                             "no_speech_threshold": " 0.4 "})
    assert out == {"temperatures": [0.0, 0.2], "best_of": 3, "patience": 1.5,
                   "condition_on_previous_text": True, "no_speech_threshold": 0.4}


# ── build_cfg ─────────────────────────────────────────────────────────────────

def test_build_cfg_maps_v13_and_v14_keys(tmp_path):
    audio = str(tmp_path / "a.wav")
    s = _settings(language="Urdu", batch="8", max_width="42", hotwords=" WhisperX ",
                  threads="6", device_index="1", diarize=True,
                  min_speakers="2", max_speakers="", align_model="", model_dir="")
    cfg = st.build_cfg(s, audio)
    assert cfg["audio"] == audio
    assert cfg["language"] == "ur"
    assert cfg["batch"] == 8 and cfg["max_width"] == 42 and cfg["max_count"] is None
    assert cfg["out_dir"] == str(tmp_path)              # blank out_dir -> the file's folder
    assert cfg["model_dir"] == st.default_model_dir()
    assert cfg["hotwords"] == "WhisperX"
    assert cfg["threads"] == 6 and cfg["device_index"] == 1
    assert cfg["diarize"] is True and "hf_token" not in cfg and "diarize_model" not in cfg
    assert cfg["min_speakers"] == 2 and cfg["max_speakers"] is None
    assert cfg["align_model"] is None
    assert cfg["decode"] == {}


# ── job lifecycle with the fake runner ────────────────────────────────────────

class _Pump:
    """Collects scheduled callbacks from the reader thread; run() drains them."""
    def __init__(self):
        self.lock, self.items = threading.Lock(), []

    def __call__(self, fn):
        with self.lock:
            self.items.append(fn)

    def drain_until(self, cond, timeout=20):
        import time
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                items, self.items = self.items, []
            for fn in items:
                fn()
            if cond():
                return True
            time.sleep(0.02)
        return False


@pytest.fixture
def fake_runner(monkeypatch):
    monkeypatch.setenv("WHISPERX_RUNNER", str(Path(__file__).parent / "fake_runner.py"))
    monkeypatch.setenv("FAKE_RUNNER_SECONDS", "0.4")


def test_run_job_with_fake_runner_ends_done(tmp_path, fake_runner):
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    s.settings["formats"] = ["srt", "txt"]
    finished = []
    s.on("finished", lambda job: finished.append(job))
    job = s.add_files([str(tmp_path / "talk.m4a")])[0]
    assert s.run_job(job) is True
    assert pump.drain_until(lambda: finished)
    assert job.status == "done" and job.progress == 1.0
    assert job.language == "ur" and job.language_prob == 0.87
    assert [os.path.basename(p) for p in job.outputs] == ["talk.srt", "talk.txt"]
    assert s.transcripts[job.id]
    assert os.path.isfile(s.queue_path)


def test_failed_job_keeps_error(tmp_path, fake_runner, monkeypatch):
    monkeypatch.setenv("FAKE_RUNNER_FAIL", "1")
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    done = []
    s.on("finished", lambda job: done.append(job))
    job = s.add_files([str(tmp_path / "x.wav")])[0]
    s.run_job(job)
    assert pump.drain_until(lambda: done)
    assert job.status == "failed"
    assert job.error.startswith("CUDA out of memory")


def test_cancel_marks_cancelled(tmp_path, fake_runner, monkeypatch):
    monkeypatch.setenv("FAKE_RUNNER_HOLD", "1")
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    done = []
    s.on("finished", lambda job: done.append(job))
    job = s.add_files([str(tmp_path / "x.wav")])[0]
    s.run_job(job)
    assert pump.drain_until(lambda: job.progress > 0)
    assert s.run_job(job) is False            # one at a time
    s.cancel()
    assert pump.drain_until(lambda: done)
    assert job.status == "cancelled" and s.running is None


# ── back-to-back queue ────────────────────────────────────────────────────────

def test_queue_runs_files_back_to_back(tmp_path, fake_runner, monkeypatch):
    monkeypatch.setenv("FAKE_RUNNER_SECONDS", "0.2")
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    order = []
    s.on("finished", lambda job: order.append(job.audio))
    a, b = s.add_files([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")])
    assert s.start_queue() is a
    assert s.start_queue() is None            # one at a time
    assert pump.drain_until(lambda: len(order) == 2)
    assert (a.status, b.status) == ("done", "done")
    assert order == [a.audio, b.audio]
    assert s.running is None and s.start_queue() is None


def test_queue_autostart_off_runs_only_one(tmp_path, fake_runner, monkeypatch):
    monkeypatch.setenv("FAKE_RUNNER_SECONDS", "0.2")
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    s.settings["queue_autostart"] = False
    done = []
    s.on("finished", lambda job: done.append(job))
    a, b = s.add_files([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")])
    s.start_queue()
    assert pump.drain_until(lambda: done)
    pump.drain_until(lambda: False, timeout=0.5)
    assert a.status == "done" and b.status == "waiting" and s.running is None


def test_cancel_skips_to_next_file(tmp_path, fake_runner, monkeypatch):
    monkeypatch.setenv("FAKE_RUNNER_HOLD", "1")
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    a, b = s.add_files([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")])
    s.start_queue()
    assert pump.drain_until(lambda: a.progress > 0)
    s.cancel()
    assert pump.drain_until(lambda: b.status == "running" and b.progress > 0)
    assert a.status == "cancelled" and s.running is b
    s.cancel()
    assert pump.drain_until(lambda: b.status == "cancelled")


def test_failure_stops_queue_when_configured(tmp_path, fake_runner, monkeypatch):
    monkeypatch.setenv("FAKE_RUNNER_FAIL", "1")
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    s.settings["queue_on_failure"] = "stop"
    done = []
    s.on("finished", lambda job: done.append(job))
    a, b = s.add_files([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")])
    s.start_queue()
    assert pump.drain_until(lambda: done)
    pump.drain_until(lambda: False, timeout=0.5)
    assert a.status == "failed" and b.status == "waiting" and s.running is None


def test_failure_continues_by_default(tmp_path, fake_runner, monkeypatch):
    monkeypatch.setenv("FAKE_RUNNER_FAIL", "1")
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    a, b = s.add_files([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")])
    s.start_queue()
    assert pump.drain_until(lambda: a.status == "failed" and b.status == "failed")


# ── open folder when done / .en safety (final review) ────────────────────────

@pytest.fixture(autouse=True)
def _no_real_folder_windows(monkeypatch):
    """Never open real Explorer windows from tests; record the calls instead."""
    opened = []
    monkeypatch.setattr(st, "open_folder", opened.append)
    return opened


def _run_all(s, pump, n):
    done = []
    s.on("finished", lambda job: done.append(job))
    s.start_queue()
    assert pump.drain_until(lambda: len(done) == n, timeout=30)
    return done


def test_open_folder_per_file(tmp_path, fake_runner, _no_real_folder_windows):
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    s.settings.update(open_folder=True, open_folder_per_batch=False, out_dir="")
    s.add_files([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")])
    _run_all(s, pump, 2)
    assert _no_real_folder_windows == [str(tmp_path), str(tmp_path)]


def test_open_folder_once_per_batch(tmp_path, fake_runner, _no_real_folder_windows):
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    s.settings.update(open_folder=True, open_folder_per_batch=True, out_dir="")
    s.add_files([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")])
    _run_all(s, pump, 2)
    assert _no_real_folder_windows == [str(tmp_path)]


def test_open_folder_off_opens_nothing(tmp_path, fake_runner, _no_real_folder_windows):
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    s.settings.update(open_folder=False)
    s.add_files([str(tmp_path / "a.wav")])
    _run_all(s, pump, 1)
    assert _no_real_folder_windows == []


# ── H2: a native crash becomes a readable message ────────────────────────────

_CRASH = """import json, sys
sys.stdin.readline()
print(json.dumps({"t": "stage", "name": "align", "state": "start", "detail": ""}), flush=True)
sys.exit(3221225477)
"""


def test_native_crash_names_the_stage(tmp_path, monkeypatch):
    script = tmp_path / "crash_runner.py"
    script.write_text(_CRASH, encoding="utf-8")
    monkeypatch.setenv("WHISPERX_RUNNER", str(script))
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    done = []
    s.on("finished", lambda job: done.append(job))
    job = s.add_files([str(tmp_path / "x.wav")])[0]
    s.run_job(job)
    assert pump.drain_until(lambda: done)
    assert job.status == "failed"
    assert job.error == "The worker crashed during Align words (code 3221225477)."


# ── H5b: add_files hygiene, commit-before-run ────────────────────────────────

def test_add_files_filters_dedups_and_emits_one_event(tmp_path):
    s = st.AppState(_Pump(), base=str(tmp_path))
    jobs, status = [], []
    s.on("job", lambda job: jobs.append(job))
    s.on("status", lambda text: status.append(text))
    a = str(tmp_path / "a.wav")
    added = s.add_files([a, str(tmp_path / "notes.txt"), str(tmp_path / "pic.PNG"),
                         str(tmp_path / "sub" / ".." / "a.wav"), str(tmp_path / "B.MP4")])
    assert [os.path.basename(j.audio) for j in added] == ["a.wav", "B.MP4"]
    assert jobs == [None]                                   # one event for the whole batch
    assert status == ["Skipped 3: 2 not audio/video · 1 already queued"]
    s.add_files([a])                                        # already waiting
    assert len(s.queue.jobs) == 2 and status[-1] == "Skipped 1: 1 already queued"
    s.add_files([str(tmp_path / "c.mp3")])                  # nothing skipped: no status
    assert len(status) == 2


def test_add_files_allows_a_finished_file_again(tmp_path):
    s = st.AppState(_Pump(), base=str(tmp_path))
    first = s.add_files([str(tmp_path / "a.wav")])[0]
    first.status = "done"
    assert len(s.add_files([str(tmp_path / "a.wav")])) == 1


def test_run_job_emits_commit_before_building_cfg(tmp_path, fake_runner):
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    s.on("commit", lambda: s.set("hotwords", "Ibrahim, Whisper"))
    job = s.add_files([str(tmp_path / "a.wav")])[0]
    assert s.run_job(job) is True
    assert job.cfg["hotwords"] == "Ibrahim, Whisper"
    s.cancel()
    pump.drain_until(lambda: s.running is None)


def test_start_queue_commits_only_when_starting(tmp_path, fake_runner):
    pump = _Pump()
    s = st.AppState(pump, base=str(tmp_path))
    n = []
    s.on("commit", lambda: n.append(1))
    assert s.start_queue() is None and n == []
    s.add_files([str(tmp_path / "a.wav")])
    assert s.start_queue() is not None and n == [1]
    s.cancel()
    pump.drain_until(lambda: s.running is None)


# ── stored transcripts ────────────────────────────────────────────────────────

def _segs():
    return [{"start": 0.0, "end": 1.0, "text": "سلام", "speaker": "SPEAKER_00",
             "words": [{"word": "سلام", "start": 0.0, "end": 1.0, "score": 0.2}]}]


def test_transcript_event_is_written_and_reloads_in_a_fresh_state(tmp_path):
    from ui.transcript import _load_segments
    s = st.AppState(lambda f: f(), base=str(tmp_path))
    job = s.add_files([str(tmp_path / "a.wav")])[0]
    s._on_event(job, {"t": "transcript", "segments": _segs()})
    assert s.transcripts[job.id] == _segs()
    assert (tmp_path / "transcripts" / f"{job.id}.json").is_file()
    s2 = st.AppState(lambda f: f(), base=str(tmp_path))
    assert s2.transcripts == {}
    assert _load_segments(s2, job) == _segs()


def test_removing_and_clearing_delete_the_stored_transcript(tmp_path):
    s = st.AppState(lambda f: f(), base=str(tmp_path))
    a, b = s.add_files([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")])
    for j in (a, b):
        s._on_event(j, {"t": "transcript", "segments": _segs()})
    s.queue.remove(a.id)
    s.queue.save(s.queue_path)
    assert not (tmp_path / "transcripts" / f"{a.id}.json").exists()
    b.status = "done"
    s.queue.clear_finished()
    s.queue.save(s.queue_path)
    assert not (tmp_path / "transcripts" / f"{b.id}.json").exists()


def test_corrupt_or_missing_transcript_file_is_no_data(tmp_path):
    from ui.transcript import _load_segments
    s = st.AppState(lambda f: f(), base=str(tmp_path))
    job = s.add_files([str(tmp_path / "a.wav")])[0]
    assert _load_segments(s, job) is None
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / f"{job.id}.json").write_text("{not json", encoding="utf-8")
    assert _load_segments(s, job) is None


def test_transcript_files_go_only_after_a_good_queue_save(tmp_path):
    s = st.AppState(lambda f: f(), base=str(tmp_path))
    a = s.add_files([str(tmp_path / "a.wav")])[0]
    s._on_event(a, {"t": "transcript", "segments": _segs()})
    f = tmp_path / "transcripts" / f"{a.id}.json"
    s.queue.remove(a.id)
    assert f.exists()                       # not before the queue is saved
    s.queue.save(s.queue_path)
    assert not f.exists()


def test_orphan_transcripts_are_swept_on_load(tmp_path):
    s = st.AppState(lambda f: f(), base=str(tmp_path))
    a = s.add_files([str(tmp_path / "a.wav")])[0]
    s._on_event(a, {"t": "transcript", "segments": _segs()})
    s.queue.save(s.queue_path)
    (tmp_path / "transcripts" / "ghost.json").write_text("[]", encoding="utf-8")
    st.AppState(lambda f: f(), base=str(tmp_path))
    assert sorted(os.listdir(tmp_path / "transcripts")) == [f"{a.id}.json"]


@pytest.mark.parametrize("raw", ["[null]", "[1]", '[{"text": "x"}, 2]'])
def test_transcript_list_must_hold_only_dicts(tmp_path, raw):
    from core.jobqueue import load_transcript
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "j.json").write_text(raw, encoding="utf-8")
    assert load_transcript(str(tmp_path), "j") is None


# ── v1.5 H19: progress estimate and the fake runner's wave ───────────────────

def test_estimate_advances_between_events_at_the_recent_speed():
    e = st.ProgressEstimate()
    e.add(0.10, 0.0)
    e.add(0.20, 1.0)                        # 0.1 per second
    assert e.value(1.5) == pytest.approx(0.25)


def test_estimate_never_passes_the_cap():
    e = st.ProgressEstimate()
    e.add(0.10, 0.0)
    e.add(0.15, 1.0)                        # previous jump 0.05
    assert e.value(100.0) == pytest.approx(0.20)
    e.add(0.50, 2.0)                        # jump 0.35, but never more than 0.15 ahead
    assert e.value(100.0) == pytest.approx(0.65)
    e.add(0.99, 3.0)
    assert e.value(100.0) == pytest.approx(0.995)     # below 100 % until done


def test_estimate_never_goes_backwards():
    e = st.ProgressEstimate()
    e.add(0.10, 0.0)
    e.add(0.30, 1.0)
    ahead = e.value(1.9)
    assert ahead > 0.30
    assert e.add(0.31, 2.0) >= ahead        # a later report below the shown value holds it
    assert e.add(0.20, 3.0) >= ahead        # and a lower report is ignored
    vals = [e.value(3.0 + i / 10) for i in range(30)]
    assert vals == sorted(vals)


def test_estimate_counts_a_burst_of_reports_as_one_step():
    e = st.ProgressEstimate()
    e.add(0.10, 0.0)
    for i in range(5):
        e.add(0.10 + 0.02 * (i + 1), 10.0 + i * 0.01)
    assert e.jump == pytest.approx(0.10)
    assert e.value(15.0) == pytest.approx(0.20 + 0.01 * 5, abs=0.002)


def test_running_job_progress_moves_between_runner_events(tmp_path):
    import time
    s = st.AppState(lambda f: f(), base=str(tmp_path))
    job = s.add_files([str(tmp_path / "a.wav")])[0]
    s.running, job.progress = job, 0.0
    t = time.monotonic()
    s._est.add(0.1, t - 2.0)
    s._on_event(job, {"t": "progress", "v": 0.2})
    seen = []
    s.on("runner", lambda job, ev: seen.append(ev))
    time.sleep(0.2)
    s._est_tick(job)
    assert 0.2 < job.progress < 0.3 and seen[-1].get("est")
    s._on_event(job, {"t": "done", "path": None})
    assert job.progress == 1.0


def test_fake_runner_sends_the_wave_before_any_load_event(tmp_path, monkeypatch):
    import json
    import subprocess
    monkeypatch.setenv("FAKE_RUNNER_SECONDS", "0.2")
    out = subprocess.run([sys.executable, str(Path(__file__).parent / "fake_runner.py")],
                         input=json.dumps({"audio": "a.wav", "out_dir": str(tmp_path)}) + "\n",
                         capture_output=True, text=True, timeout=30).stdout
    ev = [json.loads(line) for line in out.splitlines() if line.strip()]
    kinds = [e["t"] for e in ev if e["t"] in ("wave", "stage")]
    assert kinds[0] == "wave" and len(ev[[e["t"] for e in ev].index("wave")]["bins"]) == 200
    vals = [e["v"] for e in ev if e["t"] == "progress"]
    assert vals == sorted(vals) and vals[-1] == 1.0


# ── R7: the wave event's bins are stored on the job ───────────────────────────

def test_wave_event_stores_bins_on_the_job_before_screens_hear_of_it(tmp_path):
    s = st.AppState(lambda f: f(), base=str(tmp_path))
    job = s.add_files([str(tmp_path / "a.wav")])[0]
    seen = []
    s.on("runner", lambda job, ev, **_: seen.append(list(job.bins)))
    s._on_event(job, {"t": "wave", "bins": [0.1, 0.9, 0.4], "seconds": 3.0})
    assert job.bins == [0.1, 0.9, 0.4]
    assert seen == [[0.1, 0.9, 0.4]]


def test_wave_bins_are_per_job_and_a_missing_list_is_empty(tmp_path):
    s = st.AppState(lambda f: f(), base=str(tmp_path))
    a, b = s.add_files([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")])
    s._on_event(a, {"t": "wave", "bins": [0.2, 0.3]})
    s._on_event(b, {"t": "wave", "bins": [0.8]})
    assert (a.bins, b.bins) == ([0.2, 0.3], [0.8])
    s._on_event(b, {"t": "wave"})
    assert b.bins == []


def test_wave_bins_survive_a_save_and_a_fresh_state(tmp_path):
    s = st.AppState(lambda f: f(), base=str(tmp_path))
    job = s.add_files([str(tmp_path / "a.wav")])[0]
    s._on_event(job, {"t": "wave", "bins": [0.5, 0.25]})
    job.status = "done"
    s.queue.save(s.queue_path)
    s2 = st.AppState(lambda f: f(), base=str(tmp_path))
    assert s2.queue.get(job.id).bins == [0.5, 0.25]
