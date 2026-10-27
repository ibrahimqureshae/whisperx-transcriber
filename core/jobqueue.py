"""
Batch job queue — stdlib only, no GUI imports.
Persisted as queue.json beside settings.json (see .orchestrate/plan-v1-4.md).
"""
import json
import os
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

from core.settings import quarantine

STATUSES = ("waiting", "running", "done", "failed", "cancelled")


@dataclass
class Job:
    id: str
    audio: str
    cfg: dict
    status: str = "waiting"
    progress: float = 0.0
    language: Optional[str] = None
    language_prob: Optional[float] = None
    duration: Optional[float] = None      # audio seconds, from the runner's vad stage
    outputs: list = field(default_factory=list)
    started: Optional[float] = None
    finished: Optional[float] = None
    error: Optional[str] = None
    bins: list = field(default_factory=list)   # waveform thumbnail, up to 200 floats


def transcript_path(base_dir: str, job_id: str) -> str:
    """os.path.join(base_dir, 'transcripts', '<job id>.json')"""
    return os.path.join(base_dir, "transcripts", f"{job_id}.json")


def save_transcript(base_dir: str, job_id: str, segments) -> None:
    """Atomic write of a job's segments; never raises."""
    tmp = None
    try:
        path = transcript_path(base_dir, job_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[queue] could not save transcript {job_id}: {e}", file=sys.stderr)
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def load_transcript(base_dir: str, job_id: str):
    """The stored segments, or None when the file is missing or unusable."""
    try:
        with open(transcript_path(base_dir, job_id), encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    ok = isinstance(data, list) and data and all(isinstance(x, dict) for x in data)
    return data if ok else None


def delete_transcript(base_dir: str, job_id: str) -> None:
    try:
        os.remove(transcript_path(base_dir, job_id))
    except OSError:
        pass


def queue_path(base_dir: str) -> str:
    """os.path.join(base_dir, 'queue.json')"""
    return os.path.join(base_dir, "queue.json")


def _num_or(v, default):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else default


def _coerce(job: Job) -> None:
    """Repair hand-edited / wrong-typed fields so a bad entry can't crash the UI."""
    if not isinstance(job.cfg, dict):
        job.cfg = {}
    if job.status not in STATUSES:
        job.status, job.error = "failed", "Unknown status"
    job.progress = _num_or(job.progress, 0.0)
    job.duration = _num_or(job.duration, None)
    job.started = _num_or(job.started, None)
    job.finished = _num_or(job.finished, None)
    if not (isinstance(job.outputs, list) and all(isinstance(o, str) for o in job.outputs)):
        job.outputs = []
    if not (isinstance(job.bins, list) and all(isinstance(b, (int, float)) for b in job.bins)):
        job.bins = []


def _saved_bins(bins) -> list:
    """At most 200 values (peak of each chunk), rounded to 3 decimals, for queue.json."""
    bins = list(bins or [])
    if len(bins) > 200:
        n = len(bins)
        bins = [max(bins[i * n // 200:(i + 1) * n // 200]) for i in range(200)]
    return [round(b, 3) for b in bins]


class JobQueue:
    # Set by every load(): text for the UI when queue.json was unreadable, else None.
    last_load_notice = None

    def __init__(self):
        self.jobs: list = []
        self.on_save_error = None     # optional callback(text) when save() fails
        self.base_dir = None          # set by AppState: remove/clear delete stored transcripts there
        self.save_blocked = False     # queue.json was unreadable: never overwrite it this session
        self._block_reported = False
        self._pending_delete = []     # transcript ids removed from jobs, deleted after the next good save

    def add(self, audio: str, cfg: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], audio=audio, cfg=cfg)
        self.jobs.append(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return next((j for j in self.jobs if j.id == job_id), None)

    def next_waiting(self) -> Optional[Job]:
        return next((j for j in self.jobs if j.status == "waiting"), None)

    def move_up(self, job_id: str) -> None:
        for i, j in enumerate(self.jobs):
            if j.id == job_id:
                if i > 0:
                    self.jobs[i - 1], self.jobs[i] = self.jobs[i], self.jobs[i - 1]
                return

    def _drop_transcripts(self, ids) -> None:
        self._pending_delete.extend(ids if self.base_dir else ())

    def sweep_transcripts(self) -> None:
        """Delete stored transcripts whose job is not in the queue."""
        d = os.path.join(self.base_dir or "", "transcripts")
        keep = {j.id for j in self.jobs}
        try:
            names = os.listdir(d) if self.base_dir else []
        except OSError:
            return
        for n in names:
            if n.endswith(".json") and n[:-5] not in keep:
                delete_transcript(self.base_dir, n[:-5])

    def remove(self, job_id: str) -> None:
        self._drop_transcripts([job_id])
        self.jobs = [j for j in self.jobs if j.id != job_id]

    def clear_finished(self) -> None:
        self._drop_transcripts(j.id for j in self.jobs
                               if j.status in ("done", "failed", "cancelled"))
        self.jobs = [j for j in self.jobs
                     if j.status not in ("done", "failed", "cancelled")]

    def to_json(self) -> list:
        """Job dicts for queue.json."""
        out = [asdict(j) for j in self.jobs]
        for d, j in zip(out, self.jobs):
            d["bins"] = _saved_bins(j.bins)
            if not isinstance(d.get("cfg"), dict):
                d["cfg"] = {}
        return out

    @classmethod
    def from_json(cls, data) -> "JobQueue":
        q = cls()
        known = {f.name for f in fields(Job)}
        for d in data if isinstance(data, list) else []:
            if not isinstance(d, dict) or not d.get("id") or not d.get("audio"):
                continue
            try:
                job = Job(**{k: v for k, v in d.items() if k in known})
            except TypeError:
                continue
            _coerce(job)
            if job.status == "running":  # app died mid-job
                job.status = "waiting"
                job.progress = 0.0
            q.jobs.append(job)
        return q

    def save(self, path: str):
        """Write to_json() to path; never raises. On failure returns the error text
        (and passes it to on_save_error), else None."""
        if self.save_blocked:
            msg = "queue.json could not be read, so the queue is not being saved this session"
            if self.on_save_error and not self._block_reported:
                self.on_save_error(msg)
            self._block_reported = True
            return msg
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                                       suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_json(), f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
            pending, self._pending_delete = self._pending_delete, []
            for i in pending:
                delete_transcript(self.base_dir, i)
        except Exception as e:
            print(f"[queue] could not save {path}: {e}", file=sys.stderr)
            msg = f"Could not save the queue: {getattr(e, 'strerror', None) or e}"   # no temp-file name: keeps the notice throttle working
            if self.on_save_error:
                self.on_save_error(msg)
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            return msg

    @classmethod
    def load(cls, path: str) -> "JobQueue":
        """Missing file -> empty queue. Any other read error -> empty queue that refuses to
        save (save_blocked) plus a notice. A corrupt file is renamed aside first and
        JobQueue.last_load_notice is set."""
        JobQueue.last_load_notice = None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return cls()
        except OSError as e:
            JobQueue.last_load_notice = f"queue.json could not be read: {e}"
            q = cls()
            q.save_blocked = True
            return q
        except ValueError:
            JobQueue.last_load_notice = quarantine(path)
            return cls()
        if not isinstance(data, list):
            JobQueue.last_load_notice = quarantine(path)
            return cls()
        return cls.from_json(data)
