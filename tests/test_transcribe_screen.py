"""Transcribe screen: commit-before-run from every start path, fixed layout. Needs a display."""
import sys
import tkinter
from pathlib import Path

import pytest

from ui import theme

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def shell(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("base"))
    import ui.state as st
    import ui.shell as sh
    mp = pytest.MonkeyPatch()
    mp.setattr(st, "base_dir", lambda: tmp)
    mp.setattr(sh, "base_dir", lambda: tmp)
    mp.setenv("WHISPERX_RUNNER", str(Path(__file__).parent / "fake_runner.py"))
    mp.setenv("FAKE_RUNNER_HOLD", "1")
    from ui import theme
    theme._font_cache.clear()           # fonts cached by an earlier, destroyed Tk root are dead
    try:
        app = sh.Shell()
    except tkinter.TclError as e:
        mp.undo()
        if "display" not in str(e).lower():
            raise
        pytest.skip(f"no display: {e}")
    app.withdraw()
    yield app
    app.app.cancel()
    app.app.shutdown()
    app.destroy()
    theme._font_cache.clear()
    mp.undo()


def _screen(shell):
    shell.show("transcribe")
    import ui.transcribe as tr
    # the Screen registers its bound methods on the bus; find it through the hotwords entry owner
    return next(cb.__self__ for cb in shell.app._subs["browse"] if isinstance(cb.__self__, tr.Screen))


def test_speaker_dropdown_sets_both_limits_and_clears_for_auto(shell):
    scr = _screen(shell)
    scr._pick_speakers("3 speakers")
    assert shell.app.settings["min_speakers"] == shell.app.settings["max_speakers"] == "3"
    assert scr.spk_box.text == "3 speakers"
    scr._pick_speakers("Auto")
    assert shell.app.settings["min_speakers"] == shell.app.settings["max_speakers"] == ""
    assert scr.spk_box.text == "Auto"
    shell.app.set("min_speakers", "2"); shell.app.set("max_speakers", "4")
    assert scr.spk_box.text == "2–4 speakers"
    scr._pick_speakers("1 speaker")
    assert shell.app.settings["max_speakers"] == "1"


def test_recognition_dropdowns_have_keys_and_values_align(shell):
    scr = _screen(shell)
    shell.deiconify(); shell.update()
    drops = (scr.lang, scr.model, scr.device, scr.task)
    assert [d.key for d in drops] == ["Language", "Model", "Device", "Task"]
    assert len({d.x for d in drops}) == 1
    shell.withdraw()


def test_selected_chip_box_is_ok_green(shell):
    scr = _screen(shell)
    shell.app.set("formats", ["txt"])
    cv = scr.chips
    shell.update_idletasks()
    fills = {cv.itemcget(i, "fill") for i in cv.find_all() if cv.type(i) == "rectangle"}
    assert theme.C["ok"] in fills and cv.get() == ["txt"]          # the ticked box (and its edge) are green


def test_screen_fits_without_page_scrolling(shell):
    shell.deiconify()
    shell.show("transcribe")
    shell.update()
    page = shell._frames["transcribe"]
    assert page.winfo_height() >= max(c.winfo_y() + c.winfo_height() for c in page.winfo_children())
    shell.withdraw()


def test_status_text_reaches_the_shell_status_bar(shell):
    _screen(shell)
    shell.app.emit("status", text="Skipped 2: not audio/video")
    assert shell._status_lbl.cget("text") == "Skipped 2: not audio/video"


def test_language_menu_shows_the_setting_not_the_detection(shell):
    scr = _screen(shell)            # the detection is in the session panel (Say-It-Once)
    shell.deiconify(); shell.update()
    job = shell.app.add_files(["x.wav"])[0]
    job.language, job.language_prob = "en", 0.93
    scr._paint_conf()
    shell.update()
    assert scr.lang.text == "Auto-detect"
    shell.withdraw()


# ── per-file stage dots ───────────────────────────────────────────────────────
def _job(status="running", **cfg):
    from core.jobqueue import Job
    return Job(id="j1", audio="a.wav", cfg=cfg, status=status, started=100.0, finished=102.0)


def _state():
    from ui.state import AppState
    return AppState(lambda f: f(), base="x")


def _stage(name, state, detail=""):
    return {"t": "stage", "name": name, "state": state, "detail": detail}


def test_dots_for_a_running_job_with_skipped_stages():
    import ui.transcribe as tr
    st, job = _state(), _job(align=True, diarize=False)
    for ev in (_stage("load", "start"), _stage("load", "done", "m · 8.6 s"), _stage("detect", "start"),
               _stage("detect", "done"), _stage("vad", "start")):
        st._track(job, ev)
    assert tr.stage_view(job, st.stages[job.id]) == [
        "done", "done", "now", "pending", "pending", "skip", "pending"]


def test_translate_and_aligner_warning_skip_align():
    import ui.transcribe as tr
    st = _state()
    assert tr.stage_view(_job("waiting", align=True, diarize=True, task="translate"), None)[4] == "skip"
    job = _job(align=True, diarize=True)
    st._track(job, {"t": "warn", "m": "Word timestamps are not available for Georgian"})
    assert tr.stage_view(job, st.stages[job.id])[4] == "skip"


def test_failed_job_marks_the_stopping_stage_bad():
    import ui.transcribe as tr
    st, job = _state(), _job("failed", align=True, diarize=False)
    job.error = "CUDA out of memory.\nTraceback"
    for ev in (_stage("load", "start"), _stage("load", "done"), _stage("transcribe", "start")):
        st._track(job, ev)
    st.stages[job.id]["transcribe"]["state"] = "stopped"      # what AppState._on_exit does
    states = tr.stage_view(job, st.stages[job.id])
    assert states[3] == "bad" and states[4] == "pending"
    assert tr.stage_label(job, states) == ("Transcribe · CUDA out of memory.", "bad")


def test_labels_per_status():
    import ui.transcribe as tr
    done = _job("done", align=False)
    assert tr.stage_view(done, None) == ["done"] * 4 + ["skip", "skip", "done"]
    assert tr.stage_label(done, tr.stage_view(done, None)) == ("Took 0:02", "ink-2")
    wait = _job("waiting")
    assert tr.stage_label(wait, tr.stage_view(wait, None))[0] == ""       # the QUEUED pill says it
    run = _job("running")
    run.language, run.language_prob = "ur", 0.87
    now = lambda: {"state": "now", "detail": "", "t0": 1.0, "t1": None, "skip": False}
    assert tr.stage_label(run, tr.stage_view(run, {"detect": now()}))[0] == "Detect language"
    run.progress = 0.42
    assert tr.stage_label(run, tr.stage_view(run, {"transcribe": now()}))[0] == "Transcribe"
    cut = _job("cancelled")
    rec = {"vad": {"state": "stopped", "detail": "", "t0": 1.0, "t1": 2.0, "skip": False}}
    assert tr.stage_label(cut, tr.stage_view(cut, rec))[0] == "Stopped at Find speech"


def test_open_on_a_done_row_switches_to_transcript_with_that_job(shell):
    scr = _screen(shell)
    job = shell.app.add_files(["done.wav"])[0]
    job.status, job.started, job.finished = "done", 1.0, 3.0
    seen = []
    shell.app.on("screen", lambda name, **_: seen.append(("screen", name)))
    shell.app.on("open-transcript", lambda job, **_: seen.append(("open", job)))
    scr._open(job)
    assert seen == [("screen", "transcript"), ("open", job)]


def test_speaker_dropdown_shows_only_with_speakers_on_and_translate_greys_for_turbo(shell):
    scr = _screen(shell)
    shell.deiconify(); shell.update()
    shell.app.set("diarize", False); shell.update()
    assert not scr.spk_box.winfo_ismapped()
    shell.app.set("diarize", True); shell.update()
    assert scr.spk_box.winfo_ismapped()
    shell.app.set("diarize", False)
    shell.app.set("model", "large-v3-turbo")
    assert scr.task._off("Translate to English") and not scr.task._off("Transcribe")
    assert scr._task_tip.text == "This model can only transcribe."
    shell.app.set("model", "large-v2")
    assert not scr.task._off("Translate to English")
    shell.withdraw()


def test_failed_alignment_is_a_bad_dot_but_unavailable_alignment_is_hollow():
    import ui.transcribe as tr
    st = _state()
    ran = _job("running", align=True, diarize=True)
    for ev in (_stage("align", "start"),
               {"t": "warn", "m": "Word timestamps failed: boom; exporting segment timing only."},
               _stage("align", "done", "failed")):
        st._track(ran, ev)
    assert tr.stage_view(ran, st.stages[ran.id])[4] == "bad"
    for msg in ("Word timestamps are not available for Georgian; exporting segment timing only.",
                "Word timestamps need the NLTK sentence tokenizer, which could not be downloaded"):
        job = _job("running", align=True, diarize=True)
        st._track(job, {"t": "warn", "m": msg})
        assert tr.stage_view(job, st.stages[job.id])[4] == "skip"


def test_stage_tooltip_shows_seconds_once():
    import ui.transcribe as tr
    st, job = _state(), _job("running")
    for ev in (_stage("load", "start"), _stage("load", "done", "large-v3-turbo · 8.6 s")):
        st._track(job, ev)
    rec = st.stages[job.id]
    tip = tr.stage_tip(job, tr.stage_view(job, rec), rec).splitlines()[0]
    assert tip == "Load model — large-v3-turbo · 8.6 s"


def test_browse_starts_in_and_stores_last_dir(shell, tmp_path, monkeypatch):
    import ui.transcribe as tr
    scr = _screen(shell)
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return (str(tmp_path / "a.wav"),)
    monkeypatch.setattr(tr.filedialog, "askopenfilenames", fake)
    monkeypatch.setattr(scr, "add", lambda paths: None)
    shell.app.set("last_dir", "")
    scr.browse()
    assert seen["initialdir"] is None
    assert shell.app.settings["last_dir"] == str(tmp_path)
    scr.browse()
    assert seen["initialdir"] == str(tmp_path)


def test_idle_session_panel_stays_hidden(shell, tmp_path):
    scr = _screen(shell)            # queued files are named in their rows (Say-It-Once)
    shell.app.queue.jobs.clear()
    scr.cur = None
    scr._refresh_lists()
    shell.update()
    assert not scr.session.winfo_manager()
    shell.app.add_files([str(tmp_path / "next-up.wav")])
    scr._refresh_lists()
    shell.update()
    assert not scr.session.winfo_manager()
    shell.app.queue.jobs.clear()


def test_restored_finished_jobs_keep_their_track_and_open(shell, tmp_path):
    import ui.transcribe as tr
    from core.jobqueue import Job
    scr = _screen(shell)
    app = shell.app
    app.queue.jobs.clear()
    app.stages.clear()
    done = Job(id="r1", audio=str(tmp_path / "d.wav"), cfg={"align": True, "diarize": True},
               status="done", started=100.0, finished=102.0)
    bad = Job(id="r2", audio=str(tmp_path / "f.wav"), cfg={}, status="failed", error="boom")
    app.queue.jobs.extend([done, bad])
    scr.cur = None
    scr._refresh_lists()
    shell.update()
    assert set(scr.tracks) == {"r1", "r2"}
    states = tr.stage_view(done, None)
    assert set(states) == {"done"} and tr.stage_label(done, states)[0] == "Took 0:02"
    assert tr.stage_label(bad, tr.stage_view(bad, None)) == ("boom", "bad")
    opened = []
    app.on("open-transcript", lambda job: opened.append(job))
    scr._open(done)
    assert opened == [done]
    app.queue.jobs.clear()
    scr._refresh_lists()


def test_files_list_caps_finished_rows_at_newest(shell, tmp_path):
    import time
    import ui.transcribe as tr
    from core.jobqueue import Job
    scr = _screen(shell)
    app = shell.app
    app.stages.clear()
    app.queue.jobs[:] = [Job(id=f"d{i}", audio=str(tmp_path / f"{i}.wav"), cfg={}, status="done",
                             finished=1000.0 + i) for i in range(300)]
    app.queue.jobs.append(Job(id="w", audio=str(tmp_path / "w.wav"), cfg={}))
    scr.cur = None
    t0 = time.perf_counter()
    scr._refresh_lists()
    ms = (time.perf_counter() - t0) * 1000
    print(f"build {ms:.0f} ms")
    assert set(scr.tracks) == {"w"} | {f"d{i}" for i in range(300 - tr.FINISHED_ROWS, 300)}
    assert ms < 1000                 # 9 rows of Tk widgets; 301 rows took several seconds
    app.queue.jobs.clear()
    scr._refresh_lists()


def test_panel_names_the_job_with_a_verb_and_no_stage(shell):
    scr = _screen(shell)
    for task, verb in (("transcribe", "Transcribing"), ("translate", "Translating")):
        scr.cur = _job("running", task=task)
        scr._paint_session()
        assert scr.s_pill.cget("text") == verb
        assert "Left" in scr.readouts._cells and scr.readouts._cells["Left"][1].winfo_manager()
    assert "Stage" not in scr.readouts._vals
    scr.cur = _job("done", task="transcribe")
    scr._paint_session()
    assert scr.s_pill.cget("text") == "Done"
    assert not scr.readouts._cells["Left"][1].winfo_manager()
    scr.cur = None
    scr._paint_session()


def _open_stage():
    return {"state": "now", "detail": "", "t0": 1.0, "t1": None, "skip": False}


def _done_stage():
    return {"state": "done", "detail": "", "t0": 1.0, "t1": 2.0, "skip": False}


def test_running_load_stage_names_session_pill_loading_model(shell):
    scr = _screen(shell)
    job = _job("running", task="transcribe")
    scr.cur = job
    scr.state.stages[job.id] = {"load": _open_stage()}
    try:
        scr._paint_session()
        assert scr.s_pill.cget("text") == "Loading model"
    finally:
        scr.state.stages.pop(job.id, None)
        scr.cur = None
        scr._paint_session()


@pytest.mark.parametrize("task, verb", [("transcribe", "Transcribing"), ("translate", "Translating")])
def test_running_transcribe_stage_names_session_pill_for_task(shell, task, verb):
    scr = _screen(shell)
    job = _job("running", task=task)
    scr.cur = job
    scr.state.stages[job.id] = {"load": _done_stage(), "transcribe": _open_stage()}
    try:
        scr._paint_session()
        assert scr.s_pill.cget("text") == verb
    finally:
        scr.state.stages.pop(job.id, None)
        scr.cur = None
        scr._paint_session()


@pytest.mark.parametrize(
    "stage, verb",
    [
        ("detect", "Detecting language"),
        ("vad", "Finding speech"),
        ("align", "Aligning words"),
        ("diarize", "Identifying speakers"),
        ("export", "Writing files"),
    ],
)
def test_running_stage_names_session_pill_with_stage_verb(shell, stage, verb):
    scr = _screen(shell)
    job = _job("running", align=True, diarize=True)
    scr.cur = job
    scr.state.stages[job.id] = {stage: _open_stage()}
    try:
        scr._paint_session()
        assert scr.s_pill.cget("text") == verb
    finally:
        scr.state.stages.pop(job.id, None)
        scr.cur = None
        scr._paint_session()


@pytest.mark.parametrize("status, verb", [("done", "Done"), ("failed", "Failed"), ("cancelled", "Cancelled")])
def test_finished_status_names_session_pill_even_when_stage_exists(shell, status, verb):
    scr = _screen(shell)
    job = _job(status, task="transcribe")
    scr.cur = job
    scr.state.stages[job.id] = {"load": _open_stage()}
    try:
        scr._paint_session()
        assert scr.s_pill.cget("text") == verb
    finally:
        scr.state.stages.pop(job.id, None)
        scr.cur = None
        scr._paint_session()


def test_running_load_stage_keeps_file_row_track_text(shell, tmp_path):
    scr = _screen(shell)
    app = shell.app
    app.queue.jobs.clear(); app.stages.clear()
    job = _job("running", task="transcribe")
    job.audio = str(tmp_path / "load.wav")
    app.queue.jobs[:] = [job]
    app.stages[job.id] = {"load": _open_stage()}
    try:
        scr.cur = None
        scr._refresh_lists()
        shell.update_idletasks()
        assert scr.tracks[job.id].itemcget(scr.tracks[job.id].label, "text") == "Load model"
    finally:
        app.stages.pop(job.id, None)
        app.queue.jobs.clear()
        scr.cur = None
        scr._refresh_lists()


def test_setup_panel_columns_share_one_grid(shell):
    scr = _screen(shell)
    shell.app.set("diarize", True)
    shell.update_idletasks()
    import types
    cv = scr.chips                       # one canvas; each chip is a box on it
    box = lambda f: cv._box(cv.keys.index(f))
    ch = {f: types.SimpleNamespace(winfo_rootx=lambda f=f: cv.winfo_rootx() + box(f)[0],
                                   winfo_rooty=lambda f=f: cv.winfo_rooty() + box(f)[1],
                                   winfo_width=lambda f=f: box(f)[2] - box(f)[0],
                                   winfo_height=lambda f=f: box(f)[3] - box(f)[1]) for f in cv.keys}
    y = lambda w: w.winfo_rooty()
    x = lambda w: w.winfo_rootx()
    assert x(ch["srt"]) == x(ch["txt"]) == x(ch["json"]) == x(ch["aud"])          # left column
    assert x(ch["vtt"]) == x(ch["tsv"]) == x(ch["word_json"]) and x(ch["vtt"]) > x(ch["srt"])
    assert abs(ch["srt"].winfo_width() - ch["vtt"].winfo_width()) <= 1             # equal columns
    assert y(ch["srt"]) == y(ch["vtt"]) and y(ch["txt"]) == y(ch["tsv"]) and y(ch["json"]) == y(ch["word_json"])
    assert y(ch["srt"]) < y(ch["txt"]) < y(ch["json"]) < y(ch["aud"])             # Audacity labels: row 4, left
    rows = [scr.lang, scr.model, scr.device, scr.task]
    for r, a, b in zip(rows, (ch["srt"], ch["txt"], ch["json"], ch["aud"]),
                       (scr.switches[k] for k in ("align", "diarize", "open_folder", "highlight"))):
        assert abs(y(r) - y(a)) <= 1 and abs(y(a) - y(b)) <= 1
    assert not hasattr(scr, "hot")
    assert abs(scr.lang.winfo_height() - ch["srt"].winfo_height()) <= 1


def test_more_settings_link_opens_settings_on_accuracy(shell):
    scr = _screen(shell)
    assert scr.more.cget("text").startswith("More settings")
    scr.more.invoke()
    shell.update_idletasks()
    assert shell.current == "settings"
    st = shell._frames["settings"]
    assert st._tabs.get() == "Accuracy" and st._cur == 0
    st.open_tab("Word timing")
    shell.show("transcribe")
    scr.more.invoke()
    assert st._tabs.get() == "Accuracy" and st._cur == 0 and st._areas[0].winfo_manager()


def test_transcript_add_to_hotwords_still_writes_the_setting(shell):
    import types
    import ui.transcript as tx
    shell.app.set("hotwords", "")
    fake = types.SimpleNamespace(job=types.SimpleNamespace(audio="x.wav"), low=[(0, "Zainab,", 0)],
                                 state=shell.app)
    tx.Screen._rerun(fake)
    assert "Zainab" in shell.app.settings["hotwords"]
    shell.app.cancel()


# ── H21: one-canvas chips, one percent in the header, row length, a quick switch ──────────────
def test_format_chips_canvas_toggles_writes_and_keeps_one(shell):
    import ui.transcribe as tr
    scr = _screen(shell)
    shell.deiconify(); shell.update()
    cv = scr.chips
    shell.app.set("formats", ["srt"])
    x0, y0, x1, y1 = cv._box(cv.keys.index("txt"))
    cv._click(cv._at(x0 + 4, y0 + 4))                      # hit-testing finds the chip under the pointer
    assert shell.app.settings["formats"] == ["srt", "txt"] and cv.get() == ["srt", "txt"]
    cv._click("txt"); cv._click("srt")                      # the last one stays ticked
    assert shell.app.settings["formats"] == ["srt"] and cv.get() == ["srt"]
    cv.cur = "srt"; cv._move(1); cv._click(cv.cur)          # keyboard: arrow to VTT, Space ticks it
    assert shell.app.settings["formats"] == ["srt", "vtt"]
    ax, ay = cv._box(cv.keys.index("aud"))[:2]
    cv.event_generate("<Motion>", x=ax + 4, y=ay + 4)
    assert cv.tip.text == tr.AUD_TIP
    cv.event_generate("<Motion>", x=x0 + 4, y=y0 + 4)
    assert cv.tip.text == ""
    shell.app.set("formats", ["srt", "vtt", "txt"])
    shell.withdraw()


def test_header_meta_has_no_percent_and_warns_when_unsure(shell):
    scr = _screen(shell)
    job = _job("running")
    job.duration, job.language, job.language_prob = 600.0, "en", 0.93
    scr.cur = job
    scr._paint_session()
    assert scr.s_meta.cget("text") == "10:00 · English"
    assert scr.s_meta.cget("fg") == theme.C["ink-3"] and scr.m_tip.text == ""
    job.language, job.language_prob = "bo", 0.17
    scr._paint_session()
    assert scr.s_meta.cget("text") == "10:00 · Tibetan" and scr.s_meta.cget("fg") == theme.C["warn"]
    assert scr.m_tip.text == "The language guess is uncertain (Tibetan 17 %). Pick the language for a better result."
    scr.cur = None
    scr._paint_session()


def test_lang_label_names_uncertain_language_in_warning():
    import ui.transcribe as tr
    job = _job("done")
    job.language, job.language_prob = "bo", 0.17
    assert tr.lang_label(job) == (
        "Tibetan",
        "The language guess is uncertain (Tibetan 17 %). Pick the language for a better result.",
    )


def test_lang_label_keeps_unknown_code_in_uncertain_warning():
    import ui.transcribe as tr
    job = _job("done")
    job.language, job.language_prob = "xx", 0.2
    assert tr.lang_label(job) == (
        "xx",
        "The language guess is uncertain (xx 20 %). Pick the language for a better result.",
    )


def _done_text(row):
    return next(i for i in row.find_all() if row.type(i) == "text" and row.itemcget(i, "text") == "  DONE  ")


def _tip_at(tips, row, x, y):
    import types
    ev = types.SimpleNamespace(x=x, y=y)
    for tip in tips:
        if tip.w is row and tip.probe_fn:
            tip.probe(tip.probe_fn(ev))
            if tip.text:
                return tip.text
    return ""


def _row_with_clip_probe(shell, tmp_path, monkeypatch, language_prob):
    import ui.transcribe as tr
    from core.jobqueue import Job

    tips = []

    class ProbeTip:
        def __init__(self, widget, text="", probe=None):
            self.w, self.text, self.probe_fn = widget, text, probe
            tips.append(self)

        def probe(self, text):
            self.text = text

    monkeypatch.setattr(tr, "ClipTip", ProbeTip)
    scr = _screen(shell)
    app = shell.app
    app.queue.jobs.clear(); app.stages.clear()
    job = Job(id="uncertain", audio=str(tmp_path / "uncertain.wav"), cfg={}, status="done",
              finished=100.0)
    job.language, job.language_prob = "bo", language_prob
    app.queue.jobs[:] = [job]
    scr.cur = None
    scr._refresh_lists()
    shell.update_idletasks()
    return scr.rows[job.id], tips


def test_done_file_row_warns_on_uncertain_language(shell, tmp_path, monkeypatch):
    row, tips = _row_with_clip_probe(shell, tmp_path, monkeypatch, 0.17)
    done = _done_text(row)
    x0, y0, x1, y1 = row.bbox(done)
    assert (
        row.itemcget(done, "fill"),
        _tip_at(tips, row, (x0 + x1) / 2, (y0 + y1) / 2),
    ) == (
        theme.C["warn"],
        "The language guess is uncertain (Tibetan 17 %). Pick the language for a better result.",
    )


def test_done_file_row_keeps_ok_pill_when_language_is_confident(shell, tmp_path, monkeypatch):
    row, tips = _row_with_clip_probe(shell, tmp_path, monkeypatch, 0.5)
    done = _done_text(row)
    x0, y0, x1, y1 = row.bbox(done)
    assert (row.itemcget(done, "fill"), _tip_at(tips, row, (x0 + x1) / 2, (y0 + y1) / 2)) == (theme.C["ok"], "")


def test_row_length_shows_after_the_wave_event(shell, tmp_path):
    scr = _screen(shell)
    app = shell.app
    app.queue.jobs.clear(); app.stages.clear()
    job = app.add_files([str(tmp_path / "w.wav")])[0]
    scr._refresh_lists()
    row = scr.rows[job.id]
    assert row.itemcget(row.len, "text") == "—"
    job.duration = 754.0                     # AppState sets it from the wave event before the screen hears it
    scr._on_runner(job, {"t": "wave", "bins": [0.5] * 20, "seconds": 754.0})
    assert row.itemcget(row.len, "text") == "12:34"
    scr.cur = scr.run_id = None
    app.queue.jobs.clear()
    scr._refresh_lists()


def test_transcribe_show_is_quick(shell):
    import time
    _screen(shell)
    shell.deiconify()
    ms = []
    for _ in range(3):
        shell.show("queue"); shell.update()
        t = time.perf_counter(); shell.show("transcribe"); shell.update()
        ms.append((time.perf_counter() - t) * 1000)
    shell.withdraw()
    assert min(ms) < 600, ms


# ── file rows measure in the scale they are drawn in ──────────────────────────
def _row(scr, tmp_path, monkeypatch, k):
    """A _Row for a job with a 120-character name and a real (sized) file, at a stubbed scale k.
    Returns (row, mini width the row asked for)."""
    import ui.transcribe as tr
    from core.jobqueue import Job
    monkeypatch.setattr(theme, "_k", lambda w: k)
    seen = {}

    class Mini(tr.MiniLane):
        def __init__(self, parent, width=150, **kw):
            seen["width"] = width
            super().__init__(parent, width=width, **kw)

    monkeypatch.setattr(tr, "MiniLane", Mini)
    path = tmp_path / ("n" * 120 + ".wav")
    path.write_bytes(b"0" * 2048)
    job = Job(id="r1", audio=str(path), cfg={}, status="waiting")
    row = tr._Row(scr.file_list, scr, 0, job)
    return row, seen["width"]


def _name_and_size(row):
    texts = [i for i in row.find_all() if row.type(i) == "text"]
    name = next(i for i in texts if row.itemcget(i, "text").startswith("n"))
    size = next(i for i in texts if row.itemcget(i, "text").endswith(" KB"))
    return name, size


@pytest.mark.parametrize("k", [1.5, 0.85, 1.0])
def test_row_name_is_clipped_with_the_drawn_font_inside_the_scaled_column(shell, tmp_path, monkeypatch, k):
    import ui.transcribe as tr
    row, _ = _row(_screen(shell), tmp_path, monkeypatch, k)
    name, size = _name_and_size(row)
    drawn = theme._tkfont("body", row)
    assert row.itemcget(name, "font") == row.itemcget(row.create_text(0, 0, font=drawn), "font")
    fnt = theme.measure_font(drawn, row)
    width = fnt.measure(row.itemcget(name, "text"))
    room = (tr.COLS["name"] - 10) * k - (theme.measure_font(theme._tkfont("hint", row), row).measure(
        row.itemcget(size, "text")) + 8 * k)
    assert width <= room + 1                       # budget: the scaled name column minus the scaled size label
    assert width >= room - 2 * fnt.measure("n")    # and the name is not cut short of it
    assert row.itemcget(name, "text").endswith("…")


@pytest.mark.parametrize("k", [1.5, 0.85, 1.0])
def test_row_size_label_stays_inside_the_scaled_name_column(shell, tmp_path, monkeypatch, k):
    import ui.transcribe as tr
    row, _ = _row(_screen(shell), tmp_path, monkeypatch, k)
    _, size = _name_and_size(row)
    assert row.bbox(size)[2] <= round(tr.COLS["idx"] * k) + round(tr.COLS["name"] * k) + 1


@pytest.mark.parametrize("k, width", [(1.5, 165), (0.85, 94), (1.0, 110)])
def test_row_mini_lane_width_is_scaled(shell, tmp_path, monkeypatch, k, width):
    import ui.transcribe as tr
    _, mini = _row(_screen(shell), tmp_path, monkeypatch, k)
    assert mini == round((tr.COLS["mini"] - 24) * k) == width


def test_mono_label_uses_the_scaled_tk_font(shell, monkeypatch):
    import tkinter as tk
    import ui.transcribe as tr
    scr = _screen(shell)
    monkeypatch.setattr(theme, "_k", lambda w: 1.5)
    lab = tr._mono(scr.page, "ink", "panel")
    want = tk.Label(scr.page, font=theme._tkfont("mono-sm", scr.page))
    assert str(lab.cget("font")) == str(want.cget("font"))


# ── full error on hover (T11) ─────────────────────────────────────────────────
LONG_ERR = ("CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 8.00 GiB total capacity)\n"
            "Traceback: " + "x" * 100 + " end")


def _failed_track(shell, tmp_path, error):
    from core.jobqueue import Job
    scr = _screen(shell)
    app = shell.app
    app.queue.jobs.clear(); app.stages.clear()
    job = Job(id="e1", audio=str(tmp_path / "f.wav"), cfg={}, status="failed", error=error)
    app.queue.jobs.append(job)
    scr.cur = None
    scr._refresh_lists()
    shell.update()
    scr._paint_track(job, tip=True)
    return scr, job


def _clear_jobs(shell, scr):
    shell.app.queue.jobs.clear()
    scr.cur = None
    scr._refresh_lists()


def test_failed_row_tooltip_ends_with_the_full_two_line_error(shell, tmp_path):
    scr, job = _failed_track(shell, tmp_path, "  " + LONG_ERR + "\n ")
    try:
        assert len(LONG_ERR) >= 180 and LONG_ERR.count("\n") == 1
        assert scr.tracks[job.id].tip.text.endswith(LONG_ERR)
    finally:
        _clear_jobs(shell, scr)


def test_failed_row_tooltip_does_not_repeat_a_short_error(shell, tmp_path):
    import ui.transcribe as tr
    scr, job = _failed_track(shell, tmp_path, "boom")
    try:
        text = scr.tracks[job.id].tip.text
        rec = shell.app.stages.get(job.id)
        assert text.count("boom") <= 1
        assert text == tr.stage_tip(job, tr.stage_view(job, rec), rec)
    finally:
        _clear_jobs(shell, scr)


def test_row_without_error_has_no_error_text_in_track_tooltip(shell, tmp_path):
    import ui.transcribe as tr
    scr, job = _failed_track(shell, tmp_path, "")
    try:
        rec = shell.app.stages.get(job.id)
        assert scr.tracks[job.id].tip.text == tr.stage_tip(job, tr.stage_view(job, rec), rec)
    finally:
        _clear_jobs(shell, scr)


# ── R7: rows read job.bins ────────────────────────────────────────────────────

def test_transcribe_rows_show_their_job_bins_without_a_screen_copy(shell):
    scr = _screen(shell)
    q = shell.app.queue
    q.jobs.clear()
    a, b, c = (q.add(f"r7{i}.wav", {}) for i in range(3))
    for j in (a, b, c):
        j.status = "done"
        j.finished = 1.0
    a.bins, b.bins = [0.1, 0.2], [0.9, 0.6, 0.3]
    scr._refresh_lists()
    assert list(scr.minis[a.id]._bins) == [0.1, 0.2]
    assert list(scr.minis[b.id]._bins) == [0.9, 0.6, 0.3]
    assert not scr.minis[c.id]._bins            # empty bins: the stub


def test_transcribe_wave_event_updates_the_mini_lane_from_the_job(shell):
    scr = _screen(shell)
    q = shell.app.queue
    q.jobs.clear()
    j = q.add("r7live.wav", {})
    j.status = "running"
    scr._refresh_lists()
    shell.app._on_event(j, {"t": "wave", "bins": [0.3, 0.8, 0.5]})
    assert j.bins == [0.3, 0.8, 0.5]
    assert list(scr.minis[j.id]._bins) == [0.3, 0.8, 0.5]
