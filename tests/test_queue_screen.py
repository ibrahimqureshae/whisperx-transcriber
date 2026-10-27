"""Queue screen: virtualized rows, tab counts under search, per-status actions. Needs a display."""
import sys
import time
import tkinter
from pathlib import Path

import pytest
from ui.theme import C

sys.path.insert(0, str(Path(__file__).parent.parent))


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
def scr(shell):
    shell.app.queue.jobs.clear()
    shell.show("queue")
    q = shell._frames["queue"].screen
    q.search.delete(0, "end")
    q.tab = q.top = 0
    return q


def _fill(shell, q, statuses, prefix="clip"):
    jobs = []
    for i, st in enumerate(statuses):
        j = shell.app.queue.add(f"{prefix}{i:03d}.mp4", {})
        j.status = st
        jobs.append(j)
    q._refresh()
    shell.update_idletasks()
    return jobs


def test_tab_counts_follow_search(shell, scr):
    _fill(shell, scr, ["done", "failed", "cancelled", "waiting", "running"], "alpha")
    _fill(shell, scr, ["done", "waiting"], "beta")
    assert scr._labels == ["All  7", "Waiting  3", "Finished  2", "Failed  2"]
    scr.search.insert(0, "beta")
    scr._search()
    assert scr._labels == ["All  2", "Waiting  1", "Finished  1", "Failed  0"]
    assert [j.audio for j in scr.shown] == ["beta000.mp4", "beta001.mp4"]


def test_actions_per_status(shell, scr):
    jobs = _fill(shell, scr, ["running", "waiting", "waiting", "waiting", "done", "failed", "cancelled"])
    verb = {j.id: (scr.bound[j.id].verb, scr.bound[j.id].move.winfo_manager() == "pack") for j in jobs}
    assert verb[jobs[0].id] == ("", False)           # Cancel lives in the bottom bar
    assert verb[jobs[1].id] == ("Remove", False)          # first waiting: no Move up
    assert verb[jobs[2].id] == ("Remove", True)
    assert verb[jobs[3].id] == ("Remove", True)           # Remove on every waiting row
    assert verb[jobs[4].id] == ("Open", False)
    assert verb[jobs[5].id] == ("Retry", False)
    assert verb[jobs[6].id] == ("Retry", False)           # cancelled jobs can retry
    scr.act(jobs[2], "Move up")
    scr.act(jobs[3], "Remove")
    shell.update()
    assert [j.id for j in shell.app.queue.jobs][:3] == [jobs[0].id, jobs[2].id, jobs[1].id]
    assert jobs[3].id not in scr.bound


def test_progress_updates_row_in_place(shell, scr):
    jobs = _fill(shell, scr, ["running", "waiting"])
    row = scr.bound[jobs[0].id]
    rows_before = list(scr.pool)
    jobs[0].progress = 0.4
    shell.app.emit("runner", job=jobs[0], ev={"t": "progress"})
    assert scr.bound[jobs[0].id] is row
    for _ in range(40):                                   # the percent glides to the new value
        shell.update()
        time.sleep(0.02)
    assert row.pct.cget("text") == "40 %" and not row.glide.running
    shell.app.emit("job", job=None)                      # batch event without a job
    shell.update()
    assert scr.bound[jobs[0].id] is row and scr.pool == rows_before


def test_only_a_pool_of_rows_exists(shell, scr):
    _fill(shell, scr, ["waiting"] * 200)
    assert len(scr.pool) <= 10 and len(scr.bound) <= 10
    scr._yview("scroll", 50, "units")
    assert scr.top == 50 and scr.bound[scr.shown[50].id].no.cget("text") == "51"
    scr._yview("moveto", 1.0)
    assert scr.top == 200 - scr._vis()
    assert len(scr.pool) <= 10


def test_last_job_reachable_when_scaled(shell, scr):
    import customtkinter as ctk
    shell.deiconify()
    shell.update()
    ctk.set_widget_scaling(1.5)
    ctk.set_window_scaling(1.5)
    try:
        shell.update()
        jobs = _fill(shell, scr, ["waiting"] * 200)
        scr._yview("moveto", 1.0)
        shell.update()
        row = scr.bound[jobs[-1].id]
        assert row.frame.winfo_y() + row.frame.winfo_height() <= scr.rows.winfo_height() + 2
    finally:
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)
        shell.withdraw()
        shell.update()


def _pads(widget):
    v = widget.grid_info()["padx"]
    return tuple(int(x) for x in (v if isinstance(v, (tuple, list)) else str(v).replace(",", " ").strip("()").split()))


@pytest.mark.parametrize("scale", [0.85, 1.0, 1.5])
def test_row_cell_paddings_scale_with_ui_scale(shell, scale):
    import customtkinter as ctk
    from ui import queue as qm, theme
    shell.deiconify()
    shell.update()
    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    subs = {t: list(shell.app._subs.get(t, [])) for t in ("job", "runner")}
    page = None
    try:
        shell.update()
        shell.app.queue.jobs.clear()
        page = qm.build(shell._host, shell.app)
        page.grid(row=0, column=0, sticky="nsew")
        page.tkraise()
        scr = page.screen
        shell.app.queue.add("clip.mp4", {}).status = "waiting"
        scr._refresh()
        shell.update()
        row = scr.bound[scr.shown[0].id]
        k = theme._k(page)
        p18, p12, p10 = round(18 * k), round(12 * k), round(10 * k)
        assert _pads(row.no) == (p18, p12)
        for w in (row.cells[0], row.lang, row.length, row.pill, row.outs):
            assert _pads(w) == (0, p12), (scale, w)
        assert _pads(row.cells[8]) == (0, p12)               # lane box
        assert _pads(row.cells[11]) == (0, p10)                      # actions box
        assert row.cw == [round(w * k) - p12 for w, _ in qm.COLS]
    finally:
        for t, cbs in subs.items():
            shell.app._subs[t][:] = cbs
        if page is not None:
            page.destroy()
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)
        shell.withdraw()
        shell.update()


def test_long_names_cut_by_pixels_and_columns_align(shell, scr):
    long = "a-very-long-interview-recording-name-that-cannot-possibly-fit-the-file-column-2026.mp4"
    jobs = _fill(shell, scr, ["waiting"] * 3, "x")[1:]        # both rows past the first: same actions
    jobs[1].audio = long
    scr._refresh()
    shell.update()
    short_row, long_row = scr.bound[jobs[0].id], scr.bound[jobs[1].id]
    assert long_row.name.cget("text").endswith("…") and len(long_row.name.cget("text")) < len(long)
    assert long_row.tips["name"].text == long
    assert short_row.name.cget("text") == "x001.mp4" and short_row.tips["name"].text == ""
    for col in range(10):
        assert short_row.frame.grid_bbox(col, 0)[0] == long_row.frame.grid_bbox(col, 0)[0], col
    assert long_row.name.winfo_reqwidth() + long_row.size.winfo_reqwidth() <= long_row.cw[1]


def test_lane_and_percent_glide_then_stop(shell, scr):
    jobs = _fill(shell, scr, ["running"])
    row = scr.bound[jobs[0].id]
    jobs[0].progress = 0.8
    row.live(jobs[0])
    assert row.glide.running and row.glide.value < 0.8
    for _ in range(60):
        shell.update()
        time.sleep(0.02)
    assert not row.glide.running and row.glide.value == 0.8 and row.pct.cget("text") == "80 %"
    assert row.glide._job is None


def test_size_uses_the_shared_formatter_and_done_shows_no_percent(shell, scr, tmp_path):
    f = tmp_path / "tiny.wav"
    f.write_bytes(b"x" * 1500)
    j = shell.app.queue.add(str(f), {})
    j.status = "done"
    scr._refresh()
    shell.update_idletasks()
    row = scr.bound[j.id]
    assert row.size.cget("text") == "2 KB"
    assert row.pct.cget("text") == ""


def test_name_cell_centred_when_no_error_line(shell, scr):
    j = shell.app.queue.add("clip.wav", {})
    j.status = "failed"
    j.error = "boom"
    scr._refresh()
    shell.update()
    row = scr.bound[j.id]
    assert row.err.winfo_manager() == "pack"
    j.error = ""
    scr._refresh()
    shell.update()
    top = row.cells[1]
    assert row.err.winfo_manager() == ""
    cell = top.master
    gap_top, gap_bottom = top.winfo_y(), cell.winfo_height() - top.winfo_y() - top.winfo_height()
    assert abs(gap_top - gap_bottom) <= 2


def test_hand_edited_bad_entry_loads_and_screen_builds(shell, scr):
    from core.jobqueue import JobQueue
    q = JobQueue.from_json([{"id": "a1", "audio": "x.mp4", "cfg": {}, "status": "weird",
                             "duration": "long", "progress": "half", "started": "x",
                             "finished": [1], "outputs": "oops"}])
    shell.app.queue.jobs[:] = q.jobs
    scr._refresh()
    shell.update_idletasks()
    assert [j.status for j in scr.shown] == ["failed"]


def test_language_cell_is_empty_until_detected_and_warns_when_weak(shell, scr):
    jobs = _fill(shell, scr, ["waiting", "done", "done"])
    jobs[1].language, jobs[1].language_prob = "bo", 0.3
    jobs[2].language, jobs[2].language_prob = "en", 0.97
    scr._refresh()
    shell.update_idletasks()
    a, b, c = (scr.bound[j.id] for j in jobs)
    assert a.lang.cget("text") == ""
    assert b.lang.cget("text").startswith("Tibetan") and b.lang.cget("fg") == C["warn"] and b.tips["lang"].text
    assert c.lang.cget("fg") == C["ink"] and not c.tips["lang"].text


def test_language_tooltip_names_uncertain_language(shell, scr):
    jobs = _fill(shell, scr, ["done"])
    jobs[0].language, jobs[0].language_prob = "bo", 0.17
    scr._refresh()
    shell.update_idletasks()
    row = scr.bound[jobs[0].id]
    assert row.lang.cget("text").startswith("Tibetan")
    assert row.tips["lang"].text == "The language guess is uncertain (Tibetan 17 %). Pick the language for a better result."


def _done_row(shell, scr, duration, started, finished, stage=None):
    j = shell.app.queue.add("speed.wav", {})
    j.status, j.duration, j.started, j.finished = "done", duration, started, finished
    shell.app.stages.pop(j.id, None)
    if stage is not None:
        shell.app.stages[j.id] = {"transcribe": stage}
    scr._refresh()
    shell.update_idletasks()
    return scr.bound[j.id]


def test_speed_uses_transcribe_stage_time(shell, scr):
    row = _done_row(shell, scr, 60, 100, 200, {"state": "done", "t0": 150, "t1": 170})
    assert row.tips["length"].text == "3.0× realtime"


def test_speed_ignores_model_load_time(shell, scr):
    row = _done_row(shell, scr, 10, 100, 109, {"state": "done", "t0": 108, "t1": 109})
    assert row.tips["length"].text == "10× realtime"


def test_speed_falls_back_to_whole_job_without_stage_record(shell, scr):
    row = _done_row(shell, scr, 10, 100, 109)
    assert row.tips["length"].text == "1.1× realtime"


def test_speed_falls_back_to_whole_job_when_stage_unfinished(shell, scr):
    row = _done_row(shell, scr, 10, 100, 109, {"state": "running", "t0": 108, "t1": None})
    assert row.tips["length"].text == "1.1× realtime"


def test_speed_suffix_has_no_decimal_from_ten_and_one_below(shell, scr):
    fast = _done_row(shell, scr, 100, 0.5, 10.5, {"state": "done", "t0": 1, "t1": 5})
    assert fast.tips["length"].text == "25× realtime"
    slow = _done_row(shell, scr, 12, 0.5, 10.5, {"state": "done", "t0": 1, "t1": 5})
    assert slow.tips["length"].text == "3.0× realtime"


def _ratio(started, finished, stage):
    from types import SimpleNamespace
    from ui.queue import _speed_ratio
    job = SimpleNamespace(id="j1", status="done", duration=10, started=started, finished=finished)
    return _speed_ratio(job, {"j1": {"transcribe": stage}} if stage else {})


def test_speed_ratio_prefers_stage_when_job_timestamps_missing():
    assert _ratio(None, None, {"t0": 108.0, "t1": 109.0}) == 10.0


def test_speed_ratio_prefers_stage_when_job_timestamps_inverted():
    assert _ratio(109, 100, {"t0": 108.0, "t1": 109.0}) == 10.0


def test_speed_ratio_is_none_without_stage_or_job_timestamps():
    assert _ratio(None, None, None) is None


# ── full error on hover (T11) ─────────────────────────────────────────────────
LONG_ERR = ("CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 8.00 GiB total capacity)\n"
            "Traceback: " + "x" * 100 + " end")


def _failed_row(shell, scr, error):
    j = shell.app.queue.add("clip.wav", {})
    j.status = "failed"
    j.error = error
    scr._refresh()
    shell.update()
    return scr.bound[j.id]


def test_failed_row_error_tooltip_is_the_full_two_line_error(shell, scr):
    row = _failed_row(shell, scr, "  " + LONG_ERR + "\n ")
    assert row.err.cget("text").endswith("…")
    assert row.tips["err"].text == LONG_ERR


def test_failed_row_with_short_error_has_no_error_tooltip(shell, scr):
    row = _failed_row(shell, scr, "boom")
    assert row.err.cget("text") == "boom" and row.tips["err"].text == ""


def test_row_without_error_has_no_error_tooltip(shell, scr):
    row = _failed_row(shell, scr, "")
    assert row.tips["err"].text == ""
    j = shell.app.queue.add("ok.wav", {})
    j.status = "done"
    j.error = LONG_ERR
    scr._refresh()
    shell.update()
    assert scr.bound[j.id].tips["err"].text == ""


@pytest.mark.parametrize("scale", [0.85, 1.0, 1.5])
def test_header_labels_sit_over_their_row_cells(shell, scale):
    import customtkinter as ctk
    from ui import queue as qm
    shell.deiconify()
    shell.update()
    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    subs = {t: list(shell.app._subs.get(t, [])) for t in ("job", "runner")}
    page = None
    try:
        shell.update()
        shell.app.queue.jobs.clear()
        page = qm.build(shell._host, shell.app)         # built at this scale, like a fresh launch
        page.grid(row=0, column=0, sticky="nsew")
        page.tkraise()
        scr = page.screen
        for i in range(3):
            shell.app.queue.add(f"clip{i}.mp4", {}).status = "waiting"
        scr._refresh()
        shell.update()
        head = page.grid_slaves(row=1)[0]
        row = scr.bound[scr.shown[0].id]
        cells = (row.no, row.cells[0], row.cells[8], row.lang, row.length, row.pill, row.outs)
        for i, cell in enumerate(cells):
            lab = head.grid_slaves(row=0, column=i)[0]
            if i == 4:                                   # right-aligned column: compare right edges
                got = lab.winfo_rootx() + lab.winfo_width(), cell.winfo_rootx() + cell.winfo_width()
            else:
                got = lab.winfo_rootx(), cell.winfo_rootx()
            assert abs(got[0] - got[1]) <= 1, (scale, i, got)
    finally:
        for t, cbs in subs.items():
            shell.app._subs[t][:] = cbs
        if page is not None:
            page.destroy()
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)
        shell.withdraw()
        shell.update()


# ── R7: each row draws its own job's bins ─────────────────────────────────────

def test_rows_draw_their_own_bins_even_when_the_screen_never_saw_the_event(shell, scr):
    a, b, c = _fill(shell, scr, ["done", "done", "done"], "wave")
    a.bins = [0.1, 0.2, 0.3]
    b.bins = [0.9, 0.8, 0.7]
    scr._refresh()
    shell.update_idletasks()
    assert list(scr.bound[a.id].lane._bins) == [0.1, 0.2, 0.3]
    assert list(scr.bound[b.id].lane._bins) == [0.9, 0.8, 0.7]
    assert not scr.bound[c.id].lane._bins       # empty bins: the lane draws its stub


def test_wave_event_reaches_the_row_through_the_job(shell, scr):
    (j,) = _fill(shell, scr, ["running"], "live")
    shell.app._on_event(j, {"t": "wave", "bins": [0.4, 0.6]})
    shell.update_idletasks()
    assert j.bins == [0.4, 0.6]
    assert list(scr.bound[j.id].lane._bins) == [0.4, 0.6]


def test_done_rows_keep_their_own_bins_after_save_and_reload(shell, scr):
    from core.jobqueue import JobQueue
    a, b = _fill(shell, scr, ["done", "done"], "persist")
    a.bins, b.bins = [0.1, 0.5, 0.2], [0.7, 0.3]
    path = shell.app.queue_path
    shell.app.queue.save(path)
    import json
    with open(path, encoding="utf-8") as f:
        fresh = JobQueue.from_json(json.load(f))
    old = shell.app.queue
    shell.app.queue = fresh
    try:
        fresh.jobs = [j for j in fresh.jobs if j.id in (a.id, b.id)]
        scr._refresh()
        shell.update_idletasks()
        la, lb = scr.bound[a.id].lane._bins, scr.bound[b.id].lane._bins
        assert list(la) == [0.1, 0.5, 0.2] and list(lb) == [0.7, 0.3]
        assert list(la) != list(lb)
    finally:
        shell.app.queue = old
        scr._refresh()
