"""Transcript and Models screens: size formatting, row order, and a smoke build at 1366x768."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import pipeline


def _models():
    pytest.importorskip("customtkinter")
    from ui import models
    return models


# ── Size formatting ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("nbytes, text", [
    (0, "0 KB"), (500_000, "500 KB"), (999_000, "999 KB"), (999_499, "999 KB"),
    (999_500, "1.0 MB"), (9_950_000, "10 MB"), (999_500_000, "1.0 GB"),
    (1_000_000, "1.0 MB"), (5_700_000, "5.7 MB"), (9_949_999, "9.9 MB"),
    (10_000_000, "10 MB"), (464_000_000, "464 MB"), (999_000_000, "999 MB"),
    (1_200_000_000, "1.2 GB"), (3_000_000_000, "3.0 GB"),
])
def test_fmt_size(nbytes, text):
    assert _models().fmt_size(nbytes) == text


# ── Row order ─────────────────────────────────────────────────────────────────
def _fake_cache(md, repos):
    for repo in repos:
        d = Path(md) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / "x"
        d.mkdir(parents=True)
        (d / "model.bin").write_bytes(b"x")          # tiny; sizes are stubbed where it matters


def test_whisper_first_then_word_timing_cached_first_then_name(tmp_path, monkeypatch):
    models = _models()
    _fake_cache(tmp_path, ["Systran/faster-whisper-small", "Systran/faster-whisper-tiny",
                           "jonatasgrosman/wav2vec2-large-xlsr-53-english",
                           "pyannote/segmentation-3.0"])
    rows = models._collect(str(tmp_path))
    assert {"small", "tiny", "pyannote/segmentation-3.0"} <= {r["name"] for r in rows if r["cached"]}
    kinds = [r["kind"] for r in rows]
    assert kinds[:len(pipeline.MODEL_INFO)] == ["whisper"] * len(pipeline.MODEL_INFO)    # whisper first
    assert kinds[-1] != "align" and "align" in kinds       # the rest follows word timing
    for kind in ("whisper", "align"):
        grp = [r for r in rows if r["kind"] == kind]
        assert [(not r["cached"], r["name"]) for r in grp] == sorted((not r["cached"], r["name"]) for r in grp)
    assert [r["group"] for r in rows if r["kind"] == "align"] == ["Word timing"]
    assert len(rows) == len(pipeline.MODEL_INFO) + 2 + 2     # + the two non-whisper cached models + 2 speaker rows


def test_speaker_models_are_their_own_group_and_follow_the_files(tmp_path):
    models = _models()
    rows = [r for r in models._collect(str(tmp_path)) if r["kind"] == "speaker"]
    assert [r["group"] for r in rows] == ["Speakers", "Speakers"]
    assert not any(r["cached"] for r in rows)
    for key in pipeline.SPEAKER_MODELS:
        path = pipeline.speaker_model_path(str(tmp_path), key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(b"x" * 10)
    rows = [r for r in models._collect(str(tmp_path)) if r["kind"] == "speaker"]
    assert all(r["cached"] and r["size"] == 10 for r in rows)


def test_collect_uses_cached_size(tmp_path, monkeypatch):
    models = _models()
    _fake_cache(tmp_path, ["Systran/faster-whisper-small"])
    monkeypatch.setattr(pipeline, "_dir_size", lambda p: 464_000_000)    # stub: no big files on disk
    small = next(r for r in models._collect(str(tmp_path)) if r["name"] == "small")
    assert small["cached"] and small["size"] == 464_000_000 and not small["estimate"]


# ── Screens build (own process: a second Tk root in the test process is unreliable) ──
_SCREENS = """
import os, sys, time, tkinter
sys.path.insert(0, {root!r})
import ui.state as st, ui.shell as sh
st.base_dir = sh.base_dir = lambda: {base!r}
from core import pipeline
pipeline._dir_size = lambda p: 1
sh.Shell._pick_scale = staticmethod(lambda *a: 1.0)     # probes below use 1x row geometry
try:
    app = sh.Shell()
except tkinter.TclError:
    sys.exit(77)
app.app.settings["model_dir"] = {md!r}
q = app.app.queue
app.update()

# Models: one table canvas, one row per model, drawn without page scrolling
t = time.perf_counter()
app.show("models"); app.update_idletasks(); app.update()
ms = app._frames["models"]
n = len(ms._rows)
assert n == len(pipeline.MODEL_INFO) + 2 + 2, n         # + 2 speaker rows
assert len(ms._cells) == 3                       # a share meter only for cached rows
assert len(ms._slot) == n and max(ms._slot.values()) == n + 1    # two divider slots ("Word timing", "Speakers")
assert ms._cv.winfo_width() > 800

# Transcript, empty state: no job open, the finished list is shown
app.show("transcript"); app.update()
tr = [w for w in app._frames["transcript"].winfo_children()]
assert any(w.winfo_ismapped() for w in tr)

# Transcript with a finished job: body is mapped, inspector tabs switch
job = q.add({audio!r}, {{"formats": ["srt"], "model": "tiny"}})
job.status, job.started, job.finished, job.duration = "done", time.time() - 5, time.time(), 60.0
app.app.transcripts[job.id] = [
    {{"start": 0.0, "end": 4.0, "text": "hello there", "speaker": "SPEAKER_00",
     "words": [{{"word": "hello", "start": 0.0, "end": 1.0, "score": 0.9}},
               {{"word": "there", "start": 1.0, "end": 2.0, "score": 0.3}}]}},
    {{"start": 5.0, "end": 8.0, "text": "hi", "speaker": "SPEAKER_01"}},
]
app.app.emit("open-transcript", job=job)
app.update()
import ui.transcript as T
scr = next(o for o in __import__("gc").get_objects() if isinstance(o, T.Screen))
assert scr.job is job and scr.body.winfo_ismapped() and not scr.empty.winfo_ismapped()
assert len(scr.low) == 1
assert "Doubtful 1" in scr._tab_of and "Re-run" in T.TABS and "Run" not in T.TABS
assert scr.h_span.cget("text").startswith("· ") and not hasattr(scr, "h_cap")
assert set(scr.name_entries) == {{"SPEAKER_00", "SPEAKER_01"}}
for tab in T.TABS:
    scr._show_tab(tab); app.update()
    assert scr._pages[tab].winfo_ismapped(), tab
    assert sum(p.winfo_ismapped() for p in scr._pages.values()) == 1

# Models: names cut by pixels (tooltip keeps the full name); the whole pill is the click area
from types import SimpleNamespace as NS
app.deiconify(); app.show("models"); app.update()
cv = ms._cv
import ui.models as M
M._col_x = lambda width, k=1.0: [18, 330, 480, 800]       # a 300 px name column
ms._paint_table(); app.update()
assert any(v.startswith('someorg/') for v in ms._clipped.values()) and 'tiny' not in ms._clipped.values()
ri, full = next((i, v) for i, v in ms._clipped.items() if v.startswith('someorg/'))
xs = M._col_x(cv.winfo_width())
from ui import theme as TH
k = TH._k(cv)                                   # rows scale with the UI (175 % displays, 1.5x on big screens)
ymid = int((ri * M._ROW_H + M._ROW_H / 2) * k)
assert ms._probe(NS(x=xs[0] + 20, y=ymid)) == full and ms._probe(NS(x=xs[1] + 5, y=ymid)) == ""
tiny = ms._slot["tiny"]
calls = []
ms._download = lambda name, md: calls.append(name)
ms.refresh(); app.update()
cx = int(xs[3] + M._COLS[3][1] * k / 2)
cy = int((tiny * M._ROW_H + M._ROW_H / 2) * k)
for dx, dy in ((30 * k, -11 * k), (-30 * k, 11 * k)):        # corners of the pill, away from the text glyph
    cv.event_generate("<Motion>", x=cx + dx, y=cy + dy); app.update()
    cv.event_generate("<ButtonPress-1>", x=cx + dx, y=cy + dy); app.update()
assert calls == ["tiny", "tiny"], calls

# Transcript: lanes beyond three scroll with X11 wheel buttons and by dragging the thumb
app.app.transcripts[job.id] += [
    {{"start": 9.0, "end": 12.0, "text": "a", "speaker": "SPEAKER_02"}},
    {{"start": 13.0, "end": 14.0, "text": "b", "speaker": "SPEAKER_03"}},
    {{"start": 15.0, "end": 16.0, "text": "c", "speaker": "SPEAKER_04"}}]
scr.open(job); app.update()
assert len(scr.spk) == 5 and scr.vy == 0
scr._wheel(NS(x=10, y=40, delta=0, num=5)); assert scr.vy == 1
scr._wheel(NS(x=10, y=40, delta=0, num=4)); assert scr.vy == 0
w = scr.cv.winfo_width()
kk = TH._k(scr.cv)
scr._down(NS(x=w - 4, y=round(kk * T.RULER_H) + 1)); assert scr._sb and scr.vy == 0
scr._press(NS(x=w - 4, y=round(kk * (T.RULER_H + 3 * (T.LANE_H + T.LANE_GAP))) - 2)); assert scr.vy == 2
scr._sb = False
scr._down(NS(x=w // 2, y=40)); assert not scr._sb and scr.vy == 2
# playhead glide: reaches the target, then the loop is gone
scr._set_pos(5.0, scroll=True, glide=True)
assert scr._glide.running
for _ in range(60):
    app.update(); time.sleep(0.02)
assert not scr._glide.running and scr.pos == 5.0
app.app.shutdown()
print("ok")
"""


def test_screens_build(tmp_path):
    md = tmp_path / "Models"
    _fake_cache(md, ["Systran/faster-whisper-small", "pyannote/segmentation-3.0",
                          "someorg/a-very-long-model-name-that-cannot-fit-the-name-column-at-all"])
    audio = tmp_path / "a.mp4"
    audio.write_bytes(b"\0")
    code = _SCREENS.format(root=str(Path(__file__).parent.parent), base=str(tmp_path),
                           md=str(md), audio=str(audio))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180,
                       env={**os.environ, "TMP": str(tmp_path), "TEMP": str(tmp_path)})
    if r.returncode == 77:
        pytest.skip("no display")
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


# ── A long job history must not slow the screens down ─────────────────────────
_HISTORY = """
import json, os, sys, time, tkinter
sys.path.insert(0, {root!r})
json.dump([{{"id": "q%03d" % i, "audio": "bulk_%03d.wav" % i, "cfg": {{}}, "status": "done",
            "progress": 1.0, "finished": 1000.0 + i, "started": 990.0 + i}} for i in range(200)],
          open(os.path.join({base!r}, "queue.json"), "w"))
import ui.state as st, ui.shell as sh
st.base_dir = sh.base_dir = lambda: {base!r}
try:
    app = sh.Shell()
except tkinter.TclError:
    sys.exit(77)
app.update()
for name in ("queue", "transcribe", "transcript", "queue", "transcribe", "transcript"):
    t = time.perf_counter(); app.show(name); app.update()
    assert time.perf_counter() - t < 3.0, name
import gc
import ui.transcript as T
import ui.transcribe as TR
scr = next(o for o in gc.get_objects() if isinstance(o, T.Screen))
scr.open(None); app.update()
assert len(scr.empty_list.winfo_children()) <= 8, len(scr.empty_list.winfo_children())
tr = next(o for o in gc.get_objects() if isinstance(o, TR.Screen))
assert len(tr.file_list.winfo_children()) <= TR.FINISHED_ROWS
app.app.shutdown()
print("ok")
"""


def test_long_history_stays_fast(tmp_path):
    pytest.importorskip("customtkinter")
    code = _HISTORY.format(root=str(Path(__file__).parent.parent), base=str(tmp_path))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180,
                       env={**os.environ, "TMP": str(tmp_path), "TEMP": str(tmp_path)})
    if r.returncode == 77:
        pytest.skip("no display")
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


# ── Transcript timeline geometry scales with the canvas ──────────────────────
_TRANSCRIPT_SCALE = """
import gc, os, sys, time, tkinter
from types import SimpleNamespace as NS
sys.path.insert(0, {root!r})
os.makedirs({base!r}, exist_ok=True)
import ui.state as st, ui.shell as sh
st.base_dir = sh.base_dir = lambda: {base!r}
from core import pipeline
pipeline._dir_size = lambda p: 1
from ui import theme
theme._k = lambda w: {k!r}
sh.Shell._pick_scale = staticmethod(lambda *a: 1.0)
try:
    app = sh.Shell()
except tkinter.TclError:
    sys.exit(77)
app.app.settings["model_dir"] = {md!r}
q = app.app.queue
job = q.add({audio!r}, {{"formats": ["srt"], "model": "tiny"}})
job.status, job.started, job.finished, job.duration = "done", time.time() - 5, time.time(), 240.0
long = "SPEAKER_000000000000000000_LONG_NAME"
app.app.transcripts[job.id] = [
    {{"start": 0.0, "end": 3.0, "text": "long", "speaker": long,
     "words": [{{"word": "long", "start": 0.0, "end": 1.0, "score": 0.9}}]}},
    {{"start": 4.0, "end": 8.0, "text": "hello", "speaker": "SPEAKER_00"}},
    {{"start": 9.0, "end": 12.0, "text": "speech", "speaker": "Speech"}},
    {{"start": 13.0, "end": 16.0, "text": "one", "speaker": "SPEAKER_01"}},
    {{"start": 17.0, "end": 20.0, "text": "two", "speaker": "SPEAKER_02"}},
]
app.show("transcript")
app.app.emit("open-transcript", job=job)
app.update_idletasks(); app.update()
import ui.transcript as T
scr = next(o for o in gc.get_objects() if isinstance(o, T.Screen))
scr.cv.update_idletasks()
{body}
app.app.shutdown()
print("ok")
"""


def _run_transcript_scale_case(tmp_path, name, k, body):
    base = tmp_path / f"{name}_{str(k).replace('.', '_')}"
    md = base / "Models"
    audio = base / "a.mp4"
    base.mkdir()
    audio.write_bytes(b"\0")
    code = _TRANSCRIPT_SCALE.format(root=str(Path(__file__).parent.parent), base=str(base),
                                    md=str(md), audio=str(audio), k=k, body=body)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180,
                       env={**os.environ, "TMP": str(tmp_path), "TEMP": str(tmp_path)})
    if r.returncode == 77:
        pytest.skip("no display")
    return r


def test_timeline_canvas_height_scales_with_theme_k(tmp_path):
    body = """
expected = round({k!r} * (24 + 3 * (46 + 4)))
actual = int(float(scr.cv.cget("height")))
assert abs(actual - expected) <= 1, (actual, expected)
"""
    for k in (1.0, 1.75):
        r = _run_transcript_scale_case(tmp_path, "timeline_height", k, body.format(k=k))
        assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_timeline_press_uses_scaled_name_column(tmp_path):
    body = """
inside_x = 140 if {k!r} == 1.0 else 200
outside_x = round({k!r} * 150) + round({k!r} * 10) + round({k!r} * 60)
scr._glide.stop()
scr.pos = scr.vs = 0.0
scr._press(NS(x=inside_x, y=80, num=0, delta=120))
app.update()
assert scr.pos == 0.0, scr.pos
scr._press(NS(x=outside_x, y=80, num=0, delta=120))
app.update()
assert scr.pos > 0.0, scr.pos
"""
    for k in (1.0, 1.75):
        r = _run_transcript_scale_case(tmp_path, "timeline_press", k, body.format(k=k))
        assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_timeline_wheel_uses_scaled_name_column(tmp_path):
    body = """
inside_x = 140 if {k!r} == 1.0 else 200
outside_x = 160 if {k!r} == 1.0 else round({k!r} * 150) + 10
scr.vy = 0
scr.vs = 0.0
scr._wheel(NS(x=inside_x, y=80, num=5, delta=0))
assert scr.vy == 1 and scr.vs == 0.0, (scr.vy, scr.vs)
scr.vy = 0
scr.vs = 0.0
scr._wheel(NS(x=outside_x, y=80, num=5, delta=0))
assert scr.vy == 0 and scr.vs > 0.0, (scr.vy, scr.vs)
"""
    for k in (1.0, 1.75):
        r = _run_transcript_scale_case(tmp_path, "timeline_wheel", k, body.format(k=k))
        assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_timeline_probe_uses_scaled_name_column(tmp_path):
    body = """
inside_x = 140 if {k!r} == 1.0 else 200
outside_x = 160 if {k!r} == 1.0 else round({k!r} * 150) + 10
y = round({k!r} * (24 + 46 / 2))
assert scr._probe(NS(x=inside_x, y=y, num=0, delta=120)) == long
assert scr._probe(NS(x=outside_x, y=y, num=0, delta=120)) == ""
"""
    for k in (1.0, 1.75):
        r = _run_transcript_scale_case(tmp_path, "timeline_probe", k, body.format(k=k))
        assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_lane_names_use_scaled_clip_width_and_speech_lane_cannot_be_renamed(tmp_path):
    body = """
scr._show_tab("Speakers")
app.update()
assert "Speech" in scr.spk
assert "Speech" not in scr.name_entries
texts = [scr.cv.itemcget(i, "text") for i in scr.cv.find_all() if scr.cv.type(i) == "text"]
long_drawn = next((t for t in texts if t != "SPEAKER_00" and t.startswith("SPEAKER")), "")
assert long_drawn, texts
assert long_drawn != long and long_drawn.endswith("…"), texts
assert "SPEAKER_00" in texts, texts
"""
    r = _run_transcript_scale_case(tmp_path, "lane_clip", 1.75, body)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


# ── Re-run panel: the speed ratio leaves out model-load time ─────────────────
_RUN_TOOK = """
def took():
    scr._paint_run()
    app.update()
    ws = [w for w in scr.c_run.winfo_children() if hasattr(w, "cget")]
    ts = [w.cget("text") for w in ws]
    return ts[ts.index("Took") + 1]
job.status, job.duration, job.started, job.finished = "done", 10.0, 100.0, 109.0
app.app.stages.pop(job.id, None)
assert took() == "0:09 · 1.1×", took()
app.app.stages[job.id] = {"transcribe": {"state": "done", "t0": 108.0, "t1": 109.0}}
assert took() == "0:09 · 10.0×", took()
app.app.stages[job.id] = {"transcribe": {"state": "running", "t0": 108.0, "t1": None}}
assert took() == "0:09 · 1.1×", took()
"""


def test_rerun_panel_ratio_uses_transcribe_stage_when_known(tmp_path):
    r = _run_transcript_scale_case(tmp_path, "rerun_took", 1.0, _RUN_TOOK)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


# ── Crisp scaled drawing contracts ───────────────────────────────────────────
_CRISP_CANVASES = """
import gc, os, sys, tkinter
sys.path.insert(0, {root!r})
os.makedirs({base!r}, exist_ok=True)
import customtkinter as ctk
import ui.state as st, ui.shell as sh
st.base_dir = sh.base_dir = lambda: {base!r}
from core import pipeline
pipeline._dir_size = lambda p: 1
sh.Shell._pick_scale = staticmethod(lambda *a: 1.0)
try:
    app = sh.Shell()
except tkinter.TclError:
    sys.exit(77)
app.app.settings["model_dir"] = {md!r}
app.show("models"); app.update()
ms = app._frames["models"]
app.show("transcript"); app.update()
import ui.transcript as T
scr = next(o for o in gc.get_objects() if isinstance(o, T.Screen))
assert isinstance(scr.cv, ctk.CTkCanvas), type(scr.cv)
assert isinstance(ms._head, ctk.CTkCanvas), type(ms._head)
assert isinstance(ms._cv, ctk.CTkCanvas), type(ms._cv)
app.app.shutdown()
print("ok")
"""


def test_transcript_and_models_canvases_are_ctk_canvas(tmp_path):
    md = tmp_path / "Models"
    code = _CRISP_CANVASES.format(root=str(Path(__file__).parent.parent), base=str(tmp_path), md=str(md))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180,
                       env={**os.environ, "TMP": str(tmp_path), "TEMP": str(tmp_path)})
    if r.returncode == 77:
        pytest.skip("no display")
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_timeline_rule_widths_and_lane_radius_scale_with_theme_rrect(tmp_path):
    body = """
line_calls = []
orig_line = scr.cv.create_line
def record_line(*args, **kwargs):
    line_calls.append(kwargs.copy())
    return orig_line(*args, **kwargs)
scr.cv.create_line = record_line

rrect_calls = []
orig_rrect = theme.rrect
def record_rrect(cv, x0, y0, x1, y1, r, fill, edge=None, b=1, tags=()):
    rrect_calls.append((x0, y0, x1, y1, r, fill, edge, b, tags))
    return orig_rrect(cv, x0, y0, x1, y1, r, fill, edge=edge, b=b, tags=tags)
theme.rrect = record_rrect

scr._draw()
assert line_calls
assert all(isinstance(c.get("width"), int) and c["width"] >= 1 for c in line_calls), line_calls
assert any(c[4] == round(8 * 1.5) for c in rrect_calls), rrect_calls
"""
    r = _run_transcript_scale_case(tmp_path, "timeline_crisp_rules", 1.5, body)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


# ── RTL transcript display order ─────────────────────────────────────────────
_RTL_TRANSCRIPT = """
import gc, os, sys, time, tkinter
from types import SimpleNamespace as NS
sys.path.insert(0, {root!r})
from bidi.algorithm import get_display
import arabic_reshaper
def display(text):
    return get_display(arabic_reshaper.reshape(text))
def labels(w):
    out = []
    for child in w.winfo_children():
        if hasattr(child, "cget"):
            try:
                out.append(child.cget("text"))
            except Exception:
                pass
        out.extend(labels(child))
    return out
import ui.state as st, ui.shell as sh
st.base_dir = sh.base_dir = lambda: {base!r}
from core import pipeline
pipeline._dir_size = lambda p: 1
sh.Shell._pick_scale = staticmethod(lambda *a: 1.0)
try:
    app = sh.Shell()
except tkinter.TclError:
    sys.exit(77)
app.geometry("1100x760")
app.app.settings["model_dir"] = {md!r}
q = app.app.queue
job = q.add({audio!r}, {{"formats": ["srt"], "model": "tiny"}})
job.status, job.started, job.finished, job.duration = "done", time.time() - 5, time.time(), 60.0
short = "یہ اردو جملہ ہے"
long = ("یہ اردو جملہ ہے اور دائیں سے بائیں لکھا جاتا ہے۔ " * 10).strip()
english = "hello world"
low_word = "جملہ"
app.app.transcripts[job.id] = [
    {{"start": 0.0, "end": 20.0, "text": short, "speaker": "SPEAKER_00",
     "words": [{{"word": "یہ", "start": 0.0, "end": 0.5, "score": 0.9}},
               {{"word": "اردو", "start": 0.5, "end": 1.0, "score": 0.9}},
               {{"word": low_word, "start": 1.0, "end": 1.5, "score": 0.3}},
               {{"word": "ہے", "start": 1.5, "end": 2.0, "score": 0.9}}]}},
    {{"start": 22.0, "end": 30.0, "text": long, "speaker": "SPEAKER_00"}},
    {{"start": 32.0, "end": 36.0, "text": english, "speaker": "SPEAKER_01"}},
]
app.show("transcript")
app.app.emit("open-transcript", job=job)
app.update_idletasks(); app.update()
import ui.transcript as T
from ui.waveform import timecode as tc
scr = next(o for o in gc.get_objects() if isinstance(o, T.Screen))

short_parts = scr.t.get("1.0", "1.end").split("\\t", 2)
assert short_parts[2] == display(short), short_parts

scr.pos = 0.5
scr._paint_word(False)
ranges = scr.t.tag_ranges("lit0")
assert ranges
lit = scr.t.get(ranges[0], ranges[1])
assert lit == display("اردو"), lit

lines = scr.t.get("1.0", "end-1c").split("\\n")
english_line_no, english_line = next((i, line) for i, line in enumerate(lines, 1) if english in line)
assert english_line == f"{{tc(32.0)}}\\tSPEAKER_01\\t{{english}}", english_line

long_lines = lines[1:english_line_no - 1]
assert len(long_lines) > 1, long_lines
first_text = long_lines[0].split("\\t", 2)[2]
prefixes = [" ".join(long.split()[:i]) for i in range(1, len(long.split()) + 1)]
assert first_text in {{display(p) for p in prefixes}}, first_text
for cont in long_lines[1:]:
    assert not cont.startswith(tc(22.0)), cont

calls = []
scr._set_pos = lambda t, **kw: calls.append(t)
for line_no in range(2, english_line_no):
    bbox = scr.t.bbox(f"{{line_no}}.0")
    if bbox is None:
        scr.t.see(f"{{line_no}}.0"); app.update()
        bbox = scr.t.bbox(f"{{line_no}}.0")
    assert bbox is not None, line_no
    scr._click_line(NS(x=bbox[0] + 1, y=bbox[1] + max(1, bbox[3] // 2)))
assert calls == [22.0] * len(long_lines), calls

scr._draw(); app.update()
canvas_text = [scr.cv.itemcget(i, "text") for i in scr.cv.find_all() if scr.cv.type(i) == "text"]
lane_texts = {{display(short)}} | {{display(short[:i].rstrip() + "…") for i in range(1, len(short))}}
lane_matches = [text for text in canvas_text if text in lane_texts]
assert lane_matches, canvas_text
assert any(display("یہ") in text for text in lane_matches), lane_matches

scr._show_tab("Doubtful"); app.update()
assert display(low_word) in labels(scr.c_low), labels(scr.c_low)
app.app.shutdown()
print("ok")
"""


def test_rtl_transcript_screen_uses_display_order_without_changing_ltr(tmp_path):
    md = tmp_path / "Models"
    audio = tmp_path / "rtl.mp4"
    audio.write_bytes(b"\0")
    code = _RTL_TRANSCRIPT.format(root=str(Path(__file__).parent.parent), base=str(tmp_path),
                                  md=str(md), audio=str(audio))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=180,
                       env={**os.environ, "TMP": str(tmp_path), "TEMP": str(tmp_path)})
    if r.returncode == 77:
        pytest.skip("no display")
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


_RTL_MIXED_TRANSCRIPT = """
import gc, os, sys, time, tkinter
sys.path.insert(0, {root!r})
from bidi.algorithm import get_display
import arabic_reshaper
def display(text):
    return get_display(arabic_reshaper.reshape(text))
import ui.state as st, ui.shell as sh
st.base_dir = sh.base_dir = lambda: {base!r}
from core import pipeline
pipeline._dir_size = lambda p: 1
sh.Shell._pick_scale = staticmethod(lambda *a: 1.0)
try:
    app = sh.Shell()
except tkinter.TclError:
    sys.exit(77)
app.geometry("1100x760")
app.app.settings["model_dir"] = {md!r}
q = app.app.queue
job = q.add({audio!r}, {{"formats": ["srt"], "model": "tiny"}})
job.status, job.started, job.finished, job.duration = "done", time.time() - 5, time.time(), 20.0
mixed = "یہ hello world ہے"
numeric = "سال 2024 میں"
app.app.transcripts[job.id] = [
    {{"start": 0.0, "end": 4.0, "text": mixed, "speaker": "SPEAKER_00",
     "words": [{{"word": "یہ", "start": 0.0, "end": 0.5, "score": 0.9}},
               {{"word": "hello", "start": 0.5, "end": 1.0, "score": 0.9}},
               {{"word": "world", "start": 1.0, "end": 1.5, "score": 0.9}},
               {{"word": "ہے", "start": 1.5, "end": 2.0, "score": 0.9}}]}},
    {{"start": 5.0, "end": 9.0, "text": numeric, "speaker": "SPEAKER_00",
     "words": [{{"word": "سال", "start": 5.0, "end": 5.5, "score": 0.9}},
               {{"word": "2024", "start": 5.5, "end": 6.0, "score": 0.9}},
               {{"word": "میں", "start": 6.0, "end": 6.5, "score": 0.9}}]}},
]
app.show("transcript")
app.app.emit("open-transcript", job=job)
app.update_idletasks(); app.update()
import ui.transcript as T
scr = next(o for o in gc.get_objects() if isinstance(o, T.Screen))

mixed_parts = scr.t.get("1.0", "1.end").split("\\t", 2)
numeric_parts = scr.t.get("2.0", "2.end").split("\\t", 2)
assert mixed_parts[2] == display(mixed), mixed_parts
assert numeric_parts[2] == display(numeric), numeric_parts

scr.pos = 0.5
scr._paint_word(False)
ranges = scr.t.tag_ranges("lit0")
assert ranges
lit = scr.t.get(ranges[0], ranges[1])
assert lit == "hello", lit
app.app.shutdown()
print("ok")
"""


def test_rtl_word_timed_mixed_direction_segments_keep_bidi_order(tmp_path):
    md = tmp_path / "Models"
    audio = tmp_path / "rtl-mixed.mp4"
    audio.write_bytes(b"\0")
    code = _RTL_MIXED_TRANSCRIPT.format(root=str(Path(__file__).parent.parent), base=str(tmp_path),
                                        md=str(md), audio=str(audio))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=180,
                       env={**os.environ, "TMP": str(tmp_path), "TEMP": str(tmp_path)})
    if r.returncode == 77:
        pytest.skip("no display")
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


# ── Transcript timeline: the visible span fits short files ───────────────────
_TL_HELPERS = """
def open_len(L, vs=0.0):
    job.duration = float(L)
    app.app.transcripts[job.id] = [
        {"start": 0.0, "end": L / 2, "text": "clip", "speaker": "SPEAKER_00"},
        {"start": L / 2, "end": float(L), "text": "tail", "speaker": "SPEAKER_01"},
    ]
    app.app.emit("open-transcript", job=job)
    app.update_idletasks(); app.update()
    scr.cv.update_idletasks()
    assert abs(scr.total - L) < 1e-6, (scr.total, L)
    scr.vs = vs
    scr._draw()
    app.update()

def lane_edges():
    k, nw = scr._g()[:2]
    return nw + 10 * k, scr.cv.winfo_width() - 4 * k

def ticks():
    rh = scr._g()[2]
    xs = []
    for i in scr.cv.find_all():
        if scr.cv.type(i) == "line":
            c = scr.cv.coords(i)
            if len(c) == 4 and c[0] == c[2] and max(c[1], c[3]) <= rh:
                xs.append(round(c[0]))
    return sorted(xs)

def arrows():
    ws = [w for w in scr.h_name.master.winfo_children() if hasattr(w, "cget")]
    out = []
    for w in ws:
        try:
            if w.cget("text") in ("\u2192", "\u2190"):
                out.append(w)
        except Exception:
            pass
    assert len(out) == 2, out
    return out
"""


def test_timeline_span_fills_lane_for_short_file(tmp_path):
    body = _TL_HELPERS + """
for L in (6, 34, 100, 150):
    open_len(L)
    span = max(10, min(150, L))
    left, right = lane_edges()
    assert abs(scr._x(0) - left) < 1e-6, (L, scr._x(0), left)
    assert abs(scr._x(span) - right) < 1.0, (L, span, scr._x(span), right)
    assert abs(scr._x(L) - right) < 1.0 or L < 10, (L, scr._x(L), right)
    assert abs(scr._x(span / 2) - (left + right) / 2) < 1.0, (L, scr._x(span / 2))
open_len(6)
assert abs(scr._x(6) - (lane_edges()[0] + 0.6 * (lane_edges()[1] - lane_edges()[0]))) < 1.0
"""
    r = _run_transcript_scale_case(tmp_path, "span_fit", 1.0, body)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_timeline_span_is_150_for_long_files_and_10_minimum(tmp_path):
    body = _TL_HELPERS + """
open_len(400)
left, right = lane_edges()
assert abs(scr._x(150) - right) < 1.0, (scr._x(150), right)
open_len(3)
left, right = lane_edges()
assert abs(scr._x(10) - right) < 1.0, (scr._x(10), right)
"""
    r = _run_transcript_scale_case(tmp_path, "span_bounds", 1.0, body)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_timeline_short_clip_segment_is_drawn_wide_enough_for_its_text(tmp_path):
    body = _TL_HELPERS + """
open_len(6)
texts = [scr.cv.itemcget(i, "text") for i in scr.cv.find_all() if scr.cv.type(i) == "text"]
assert "clip" in texts, texts
"""
    r = _run_transcript_scale_case(tmp_path, "span_clip_text", 1.0, body)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_timeline_ruler_step_is_smallest_with_at_most_16_ticks(tmp_path):
    body = _TL_HELPERS + """
# (file length, expected tick step in seconds)
for L, step in ((6, 1), (20, 2), (34, 5), (60, 5), (100, 10), (150, 10)):
    open_len(L)
    span = max(10, min(150, L))
    want = sorted(round(scr._x(t)) for t in range(0, int(span) + 1, step))
    got = ticks()
    assert got == want, (L, step, got, want)
    assert len(got) <= 17, (L, len(got))
"""
    r = _run_transcript_scale_case(tmp_path, "ruler_step", 1.0, body)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_timeline_ruler_ticks_follow_scroll_on_long_file(tmp_path):
    body = _TL_HELPERS + """
open_len(400, vs=45.0)
want = sorted(round(scr._x(t)) for t in range(50, 196, 10))
assert ticks() == want, (ticks(), want)
"""
    r = _run_transcript_scale_case(tmp_path, "ruler_scrolled", 1.0, body)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_timeline_short_file_does_not_scroll_and_hides_arrows(tmp_path):
    body = _TL_HELPERS + """
for L in (6, 34, 150):
    open_len(L)
    assert all(not w.winfo_ismapped() for w in arrows()), L
    scr._scroll(50.0)
    scr._scroll(-50.0)
    scr._scroll(1000.0)
    assert scr.vs == 0.0, (L, scr.vs)
    scr._wheel(NS(x=500, y=80, num=5, delta=0))
    assert scr.vs == 0.0, (L, scr.vs)
    scr._set_pos(L * 0.9, scroll=True)
    app.update()
    assert scr.vs == 0.0, (L, scr.vs)
"""
    r = _run_transcript_scale_case(tmp_path, "short_no_scroll", 1.0, body)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_timeline_long_file_shows_arrows_and_scrolls_half_a_span(tmp_path):
    body = _TL_HELPERS + """
open_len(400)
assert all(w.winfo_ismapped() for w in arrows())
fwd = next(w for w in arrows() if w.cget("text") == "\u2192")
back = next(w for w in arrows() if w.cget("text") == "\u2190")
fwd.invoke(); app.update()
assert scr.vs == 75.0, scr.vs
fwd.invoke(); app.update()
assert scr.vs == 150.0, scr.vs
back.invoke(); app.update()
assert scr.vs == 75.0, scr.vs
scr._scroll(10000.0)
assert scr.vs == 250.0, scr.vs
# arrows reappear after a short file was open in between
open_len(6)
assert all(not w.winfo_ismapped() for w in arrows())
open_len(400)
assert all(w.winfo_ismapped() for w in arrows())
"""
    r = _run_transcript_scale_case(tmp_path, "long_arrows", 1.0, body)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_timeline_click_seek_uses_the_same_span_as_drawing(tmp_path):
    body = _TL_HELPERS + """
for L in (6, 34, 100):
    open_len(L)
    span = max(10, min(150, L))
    scr._sb = False
    for frac in (0.25, 0.5, 0.9):
        t = min(span * frac, L)      # a click past the end of a short file lands on its end
        left, right = lane_edges()
        x = round(left + frac * (right - left))
        scr._glide.stop()
        scr.pos = 0.0
        scr._press(NS(x=x, y=80, num=0, delta=120))
        app.update()
        tol = span / (right - left) * 1.5
        assert abs(scr.pos - t) <= tol, (L, t, scr.pos, tol)
        want_x = x if span * frac <= L else scr._x(L)
        assert abs(scr._x(scr.pos) - want_x) <= 1.5, (L, want_x, scr._x(scr.pos))
    # the drawn playhead sits where a click at that time lands
    scr._glide.stop()
    scr.pos = span / 2
    scr._draw()
    app.update()
    px = [(c[0] + c[2]) / 2 for c in (scr.cv.coords(i) for i in scr.cv.find_all()
          if scr.cv.type(i) == "rectangle") if c[1] == 0 and c[3] >= scr.cv.winfo_height() - 1]
    assert px and abs(px[0] - scr._x(span / 2)) <= 1.5, (L, px, scr._x(span / 2))

open_len(400, vs=100.0)
scr._sb = False
scr._glide.stop()
scr.pos = 0.0
scr._press(NS(x=round(scr._x(175.0)), y=80, num=0, delta=120))
app.update()
assert abs(scr.pos - 175.0) <= 1.5, scr.pos
"""
    r = _run_transcript_scale_case(tmp_path, "click_span", 1.0, body)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_timeline_long_file_wheel_and_seek_still_use_150s_window(tmp_path):
    body = _TL_HELPERS + """
open_len(400)
scr.vs = 0.0
scr._wheel(NS(x=500, y=80, num=5, delta=0))
assert scr.vs == 30.0, scr.vs
scr._wheel(NS(x=500, y=80, num=4, delta=120))
assert scr.vs == 0.0, scr.vs
scr._set_pos(200.0, scroll=True)
app.update()
assert scr.vs == 190.0, scr.vs
"""
    r = _run_transcript_scale_case(tmp_path, "long_wheel", 1.0, body)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr
