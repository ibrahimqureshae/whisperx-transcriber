"""Tests for core/jobqueue.py (batch job queue with persistence). No GUI, no network."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.jobqueue import Job, JobQueue, queue_path


def _ids(q):
    return [j.id for j in q.jobs]


def _three(cfg=None):
    q = JobQueue()
    a = q.add("a.wav", cfg or {})
    b = q.add("b.wav", cfg or {})
    c = q.add("c.wav", cfg or {})
    return q, a, b, c


# ── queue_path ────────────────────────────────────────────────────────────────

def test_queue_path_is_queue_json_in_base_dir(tmp_path):
    assert queue_path(str(tmp_path)) == os.path.join(str(tmp_path), "queue.json")


# ── add / get ─────────────────────────────────────────────────────────────────

def test_add_returns_waiting_job_with_given_audio_and_cfg():
    q = JobQueue()
    cfg = {"model": "small", "formats": ["srt"]}
    job = q.add("talk.wav", cfg)
    assert isinstance(job, Job)
    assert job.audio == "talk.wav"
    assert job.cfg == cfg
    assert job.status == "waiting"
    assert job.progress == 0.0
    assert job.outputs == []
    assert q.jobs == [job]


def test_add_gives_unique_non_empty_string_ids():
    q, a, b, c = _three()
    ids = [a.id, b.id, c.id]
    assert all(isinstance(i, str) and i for i in ids)
    assert len(set(ids)) == 3


def test_add_appends_in_order():
    q, a, b, c = _three()
    assert _ids(q) == [a.id, b.id, c.id]


def test_get_returns_job_or_none():
    q, a, b, _ = _three()
    assert q.get(b.id) is b
    assert q.get("no-such-id") is None


# ── next_waiting ──────────────────────────────────────────────────────────────

def test_next_waiting_returns_first_waiting_job():
    q, a, b, c = _three()
    assert q.next_waiting() is a


def test_next_waiting_skips_non_waiting_jobs():
    q, a, b, c = _three()
    a.status = "done"
    b.status = "running"
    assert q.next_waiting() is c


def test_next_waiting_none_when_empty_or_nothing_waiting():
    q = JobQueue()
    assert q.next_waiting() is None
    job = q.add("a.wav", {})
    job.status = "failed"
    assert q.next_waiting() is None


# ── move_up ───────────────────────────────────────────────────────────────────

def test_move_up_swaps_with_previous_job():
    q, a, b, c = _three()
    q.move_up(c.id)
    assert _ids(q) == [a.id, c.id, b.id]
    q.move_up(c.id)
    assert _ids(q) == [c.id, a.id, b.id]


def test_move_up_top_job_is_noop():
    q, a, b, c = _three()
    q.move_up(a.id)
    assert _ids(q) == [a.id, b.id, c.id]


def test_move_up_unknown_id_is_noop():
    q, a, b, c = _three()
    q.move_up("nope")
    assert _ids(q) == [a.id, b.id, c.id]


# ── remove ────────────────────────────────────────────────────────────────────

def test_remove_deletes_job():
    q, a, b, c = _three()
    q.remove(b.id)
    assert _ids(q) == [a.id, c.id]
    assert q.get(b.id) is None


def test_remove_unknown_id_is_noop():
    q, a, b, c = _three()
    q.remove("nope")
    assert _ids(q) == [a.id, b.id, c.id]


# ── clear_finished ────────────────────────────────────────────────────────────

def test_clear_finished_removes_done_failed_cancelled_keeps_rest():
    q = JobQueue()
    jobs = [q.add(f"{i}.wav", {}) for i in range(5)]
    waiting, running, done, failed, cancelled = jobs
    running.status = "running"
    done.status = "done"
    failed.status = "failed"
    cancelled.status = "cancelled"
    q.clear_finished()
    assert _ids(q) == [waiting.id, running.id]


# ── JSON round-trip ───────────────────────────────────────────────────────────

def _rich_queue():
    q = JobQueue()
    done = q.add("one.wav", {"model": "small", "formats": ["srt", "txt"]})
    done.status = "done"
    done.progress = 1.0
    done.language = "en"
    done.language_prob = 0.97
    done.outputs = ["C:/out/one.srt", "C:/out/one.txt"]
    done.started = 1000.5
    done.finished = 1042.25
    failed = q.add("two.wav", {"model": "tiny"})
    failed.status = "failed"
    failed.error = "boom"
    failed.started = 2000.0
    failed.finished = 2001.0
    q.add("three.wav", {"model": "base"})
    return q


def test_to_json_is_serialisable_and_has_every_job_field():
    q = _rich_queue()
    data = q.to_json()
    json.dumps(data)
    assert isinstance(data, list) and len(data) == 3
    fields = {"id", "audio", "cfg", "status", "progress", "language",
              "language_prob", "outputs", "started", "finished", "error"}
    for entry in data:
        assert isinstance(entry, dict)
        assert fields <= set(entry)


def test_from_json_rebuilds_same_jobs():
    q = _rich_queue()
    q2 = JobQueue.from_json(q.to_json())
    assert isinstance(q2, JobQueue)
    assert q2.jobs == q.jobs


def test_save_load_round_trip_keeps_all_fields(tmp_path):
    q = _rich_queue()
    p = tmp_path / "queue.json"
    q.save(str(p))
    assert p.exists()
    loaded = JobQueue.load(str(p))
    assert loaded.jobs == q.jobs
    first = loaded.jobs[0]
    assert first.outputs == ["C:/out/one.srt", "C:/out/one.txt"]
    assert first.started == 1000.5
    assert first.finished == 1042.25
    assert first.language == "en"
    assert first.language_prob == 0.97
    assert loaded.jobs[1].error == "boom"


def test_round_trip_preserves_order(tmp_path):
    q, a, b, c = _three()
    q.move_up(c.id)
    p = tmp_path / "queue.json"
    q.save(str(p))
    assert _ids(JobQueue.load(str(p))) == [a.id, c.id, b.id]


# ── crash recovery ────────────────────────────────────────────────────────────

def test_from_json_resets_running_job_to_waiting():
    q = JobQueue()
    job = q.add("a.wav", {})
    job.status = "running"
    job.progress = 0.6
    q2 = JobQueue.from_json(q.to_json())
    got = q2.get(job.id)
    assert got.status == "waiting"
    assert got.progress == 0.0


def test_load_resets_running_job_and_keeps_other_statuses(tmp_path):
    q = JobQueue()
    r = q.add("r.wav", {})
    r.status = "running"
    r.progress = 0.4
    d = q.add("d.wav", {})
    d.status = "done"
    d.progress = 1.0
    p = tmp_path / "queue.json"
    q.save(str(p))
    loaded = JobQueue.load(str(p))
    assert loaded.get(r.id).status == "waiting"
    assert loaded.get(r.id).progress == 0.0
    assert loaded.get(d.id).status == "done"
    assert loaded.get(d.id).progress == 1.0


# ── load / save robustness ────────────────────────────────────────────────────

def test_load_missing_file_returns_empty_queue(tmp_path):
    q = JobQueue.load(str(tmp_path / "queue.json"))
    assert isinstance(q, JobQueue)
    assert q.jobs == []


def test_load_corrupt_file_returns_empty_queue(tmp_path):
    p = tmp_path / "queue.json"
    p.write_text("{not valid json", encoding="utf-8")
    q = JobQueue.load(str(p))
    assert isinstance(q, JobQueue)
    assert q.jobs == []


def test_save_to_unwritable_path_does_not_raise(tmp_path):
    # An existing directory cannot be opened as a file.
    q = _rich_queue()
    q.save(str(tmp_path))




# ── R7: waveform bins live on the job ─────────────────────────────────────────

def test_job_bins_default_is_an_independent_empty_list():
    a, b = Job(id="1", audio="a.wav", cfg={}), Job(id="2", audio="b.wav", cfg={})
    assert a.bins == [] and b.bins == []
    a.bins.append(0.5)
    assert b.bins == []


def test_bins_round_trip_through_save_and_load(tmp_path):
    q = JobQueue()
    a, b = q.add("a.wav", {}), q.add("b.wav", {})
    a.bins = [0.0, 0.25, 0.5]
    b.bins = [1.0, 0.75]
    p = str(tmp_path / "queue.json")
    q.save(p)
    q2 = JobQueue.from_json(json.loads(Path(p).read_text(encoding="utf-8")))
    assert [j.bins for j in q2.jobs] == [[0.0, 0.25, 0.5], [1.0, 0.75]]


def test_queue_json_without_bins_loads_with_empty_bins():
    q = JobQueue.from_json([{"id": "x", "audio": "x.wav", "cfg": {}, "status": "done"}])
    assert q.jobs[0].bins == []


def test_saved_bins_are_capped_at_200_and_rounded_to_3_decimals():
    q = JobQueue()
    j = q.add("a.wav", {})
    j.bins = [i / 7777 for i in range(1, 501)]
    d = q.to_json()[0]
    assert len(d["bins"]) == 200
    assert all(v == round(v, 3) for v in d["bins"])
    assert any(v != 0 for v in d["bins"])
    short = q.add("b.wav", {})
    short.bins = [0.123456, 0.987654]
    assert q.to_json()[1]["bins"] == [0.123, 0.988]
