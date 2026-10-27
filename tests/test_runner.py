"""
Runner tests: core.runner.run(cfg) with every pipeline function faked.
No torch/whisperx, no network, nothing outside tmp_path.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import runner


# ── helpers ───────────────────────────────────────────────────────────────────

class _Harness:
    def __init__(self):
        self.events = []
        self.load_kwargs = None
        self.load_args = None
        self.download_calls = []
        self.detect_calls = []
        self.transcribe_calls = []
        self.audio = "AUDIO"
        self.bins = [0.5] * 200
        self.bins_calls = []
        self.segments = []
        self.progress_calls = []     # percents the fake transcribe reports
        self.export_rows = []
        self.align_calls = []        # (lang, dev, audio, kwargs)
        self.export_calls = []       # (args, kwargs)
        self.diarize_calls = []      # ((audio, device), kwargs)
        self.assign_calls = []       # (diarize_segments, result, embeddings)
        self.diarize_df = "DIARIZE_DF"
        self.diarize_embeddings = {"SPEAKER_00": [0.1, 0.2]}
        self.diarize_error = None
        self.speakers_ready = True   # are the speaker models on disk?
        self.ensure_calls = []       # (model_dir, offline)
        self.ensure_error = None
        self.ensure_progress = ()    # fractions the fake download reports
        self.assigned_segments = None  # segments the fake assign_speakers returns

    def of(self, kind):
        return [e for e in self.events if e.get("t") == kind]


@pytest.fixture
def harness(monkeypatch, tmp_path):
    h = _Harness()
    h.model_dir = tmp_path / "models"
    h.model_dir.mkdir()
    h.cached = True
    h.detected = ("ur", 0.87)

    monkeypatch.setattr(runner, "_emit", h.events.append)
    monkeypatch.setenv("HF_HOME", str(h.model_dir))
    P = runner.pipeline
    monkeypatch.setattr(P, "detect_device", lambda req: ("cpu", "cpu", "cpu"))
    monkeypatch.setattr(P, "resolve_compute", lambda t, c: (c, False))

    def cache_dir(model, model_dir):
        d = tmp_path / ("cache_present" if h.cached else "cache_missing")
        if h.cached:
            d.mkdir(exist_ok=True)
        return str(d)

    monkeypatch.setattr(P, "model_cache_dir", cache_dir)
    monkeypatch.setattr(P, "download_model",
                        lambda *a, **k: h.download_calls.append((a, k)))

    def load_model(*a, **k):
        h.load_args, h.load_kwargs = a, k
        return object()

    monkeypatch.setattr(P, "load_model", load_model)
    def load_audio(path):
        h.events.append({"t": "_mark", "name": "load_audio"})
        return h.audio

    monkeypatch.setattr(P, "load_audio", load_audio)

    def waveform_bins(audio, n=200):
        h.bins_calls.append(audio)
        return h.bins

    monkeypatch.setattr(P, "waveform_bins", waveform_bins, raising=False)

    def detect_language(model, audio):
        h.detect_calls.append((model, audio))
        return h.detected

    monkeypatch.setattr(P, "detect_language", detect_language)

    def transcribe(model, audio, batch, language, **k):
        h.events.append({"t": "_mark", "name": "transcribe"})
        h.transcribe_calls.append((language, k))
        cb = k.get("progress_cb")
        if cb is not None:
            for p in h.progress_calls:
                cb(p)
        return ({"segments": list(h.segments), "language": language or "en"},
                language or "en")

    monkeypatch.setattr(P, "transcribe", transcribe)
    monkeypatch.setattr(P, "align_supported", lambda lang: True)
    def align(result, lang, dev, audio, **k):
        h.events.append({"t": "_mark", "name": "align"})
        h.align_calls.append((lang, dev, audio, k))
        if getattr(h, "align_error", None):
            raise h.align_error
        return result

    monkeypatch.setattr(P, "align", align)
    h.punkt = True
    h.punkt_calls = []

    def ensure_punkt(model_dir, offline):
        h.punkt_calls.append((model_dir, offline))
        return h.punkt

    monkeypatch.setattr(P, "ensure_punkt", ensure_punkt)
    monkeypatch.setattr(P, "unload_model",
                        lambda m=None: h.events.append({"t": "_mark", "name": "unload"}))

    def export(*a, **k):
        h.export_calls.append((a, k))
        return list(h.export_rows)

    monkeypatch.setattr(P, "export", export)

    def diarize(audio, model_dir, **k):
        h.diarize_calls.append(((audio, model_dir), k))
        if h.diarize_error is not None:
            raise h.diarize_error
        if k.get("return_embeddings"):
            return h.diarize_df, h.diarize_embeddings
        return h.diarize_df

    def assign_speakers(diarize_segments, result, speaker_embeddings=None):
        h.assign_calls.append((diarize_segments, result, speaker_embeddings))
        if h.assigned_segments is None:
            return result
        out = dict(result)
        out["segments"] = h.assigned_segments
        return out

    def ensure_speaker_models(model_dir, offline, progress=None):
        h.ensure_calls.append((model_dir, offline))
        if h.ensure_error is not None:
            raise h.ensure_error
        for f in h.ensure_progress:
            progress(f)
        h.speakers_ready = True

    monkeypatch.setattr(P, "diarize", diarize)
    monkeypatch.setattr(P, "speaker_models_ready", lambda model_dir: h.speakers_ready)
    monkeypatch.setattr(P, "ensure_speaker_models", ensure_speaker_models)
    monkeypatch.setattr(P, "assign_speakers", assign_speakers)
    h.tmp_path = tmp_path
    (tmp_path / "a.wav").write_bytes(b"RIFF")   # the audio file must exist (R2)
    return h


def _cfg(h, **over):
    cfg = {
        "device": "cpu", "compute": "int8", "language": "en", "model": "small",
        "model_dir": str(h.model_dir), "beam": 5, "batch": 4,
        "audio": str(h.tmp_path / "a.wav"), "align": False, "formats": [],
        "out_dir": str(h.tmp_path / "out"), "max_width": None, "max_count": None,
        "highlight": False,
    }
    cfg.update(over)
    return cfg


# ── options reach load_model ─────────────────────────────────────────────────

def test_new_options_reach_load_model(harness):
    decode = {"best_of": 2, "temperatures": [0.0, 0.3]}
    runner.run(_cfg(harness, hotwords="Kubernetes", suppress_numerals=True,
                    vad_method="silero", device_index=2, threads=6,
                    offline_only=True, decode=decode))
    kw = harness.load_kwargs
    assert kw["hotwords"] == "Kubernetes"
    assert kw["suppress_numerals"] is True
    assert kw["vad_method"] == "silero"
    assert kw["device_index"] == 2
    assert kw["threads"] == 6
    assert kw["local_files_only"] is True
    assert kw["decode"] == decode


def test_load_model_defaults_when_options_absent(harness):
    runner.run(_cfg(harness))
    kw = harness.load_kwargs
    assert kw["hotwords"] is None
    assert kw["suppress_numerals"] is False
    assert kw["vad_method"] == "pyannote"
    assert kw["device_index"] == 0
    assert kw["threads"] == 4
    assert kw["local_files_only"] is False
    assert kw["decode"] is None


# ── offline ──────────────────────────────────────────────────────────────────

def test_offline_only_uncached_model_fails_before_loading(harness):
    harness.cached = False
    with pytest.raises(runner.UserError, match="Model 'small' is not downloaded. "
                       "Download it in Models or turn off Offline only."):
        runner.run(_cfg(harness, offline_only=True))
    assert harness.download_calls == []
    assert harness.load_kwargs is None


def test_online_downloads_uncached_model(harness):
    harness.cached = False
    runner.run(_cfg(harness))
    assert len(harness.download_calls) == 1


# ── language event ───────────────────────────────────────────────────────────

def test_auto_language_detects_emits_event_and_transcribes_with_code(harness):
    runner.run(_cfg(harness, language="auto"))
    assert len(harness.detect_calls) == 1
    assert harness.detect_calls[0][1] == "AUDIO"
    assert harness.of("language") == [{"t": "language", "code": "ur", "prob": 0.87}]
    assert harness.transcribe_calls[0][0] == "ur"


def test_explicit_language_emits_prob_one_and_skips_detection(harness):
    runner.run(_cfg(harness, language="fr"))
    assert harness.detect_calls == []
    assert harness.of("language") == [{"t": "language", "code": "fr", "prob": 1.0}]
    assert harness.transcribe_calls[0][0] == "fr"


# ── v1.4 T2: progress and stage events ───────────────────────────────────────

def _stages(h):
    return [e for e in h.events if e.get("t") == "stage"]


def _stage(h, name, state):
    found = [e for e in _stages(h) if e["name"] == name and e["state"] == state]
    assert len(found) == 1, f"expected exactly one {name} {state}, got {found}"
    return found[0]


def test_progress_callback_drives_progress_and_status_events(harness):
    harness.progress_calls = [50]
    runner.run(_cfg(harness))
    assert harness.transcribe_calls[0][1]["progress_cb"] is not None
    assert pytest.approx(0.04 + 0.94 * 0.5) in [e["v"] for e in harness.of("progress")]
    assert {"t": "status", "m": "Transcribing… 50%"} in harness.of("status")


def test_progress_callback_switches_to_determinate_once(harness):
    harness.progress_calls = [10, 25, 100]
    runner.run(_cfg(harness))
    # overall: load 0-4 %, transcribe 4-98 % (nothing else on), export the last 2 %
    assert [e["v"] for e in harness.of("progress")] == pytest.approx(
        [0.0, 0.04, 0.134, 0.275, 0.98, 1.0])
    det = [e for e in harness.of("pmode") if e["mode"] == "determinate"]
    assert len(det) == 1
    msgs = [e["m"] for e in harness.of("status")]
    assert "Transcribing… 10%" in msgs
    assert "Transcribing… 25%" in msgs
    assert "Transcribing… 100%" in msgs


def test_every_stage_event_has_t_name_state_detail(harness):
    harness.export_rows = [("srt", str(harness.tmp_path / "out" / "a.srt"), None)]
    runner.run(_cfg(harness, align=True, formats=["srt"]))
    stages = _stages(harness)
    assert stages
    for e in stages:
        assert {"t", "name", "state", "detail"} <= set(e)
        assert e["t"] == "stage"
        assert e["state"] in ("start", "done")
        assert isinstance(e["detail"], str)


def test_stage_events_emitted_in_order_for_full_run(harness):
    harness.export_rows = [("srt", str(harness.tmp_path / "out" / "a.srt"), None)]
    runner.run(_cfg(harness, language="auto", align=True, formats=["srt"]))
    order = [(e["name"], e["state"]) for e in _stages(harness)]
    assert order == [
        ("load", "start"), ("load", "done"),
        ("vad", "start"), ("vad", "done"),
        ("detect", "start"), ("detect", "done"),
        ("transcribe", "start"), ("transcribe", "done"),
        ("align", "start"), ("align", "done"),
        ("export", "start"), ("export", "done"),
    ]


def test_load_stage_reports_model_name_and_seconds(harness):
    import re
    runner.run(_cfg(harness, model="small"))
    assert _stage(harness, "load", "start")["detail"] == "small"
    done = _stage(harness, "load", "done")["detail"]
    assert re.fullmatch(r"small · \d+\.\d s", done), done


def test_audio_loads_before_the_model_and_vad_reports_duration(harness):
    harness.audio = [0.0] * 32000          # 2 s at 16 kHz
    runner.run(_cfg(harness))
    ev = harness.events
    i_mark = ev.index({"t": "_mark", "name": "load_audio"})
    assert i_mark < ev.index(_stage(harness, "load", "start"))
    assert _stage(harness, "vad", "done")["detail"] == "2 s audio"


def test_wave_event_carries_200_bins_and_seconds_before_any_load_event(harness):
    harness.audio = [0.0] * 16000
    harness.bins = [i / 199 for i in range(200)]
    runner.run(_cfg(harness))
    assert harness.bins_calls == [harness.audio]
    (wave,) = harness.of("wave")
    assert wave["bins"] == harness.bins and wave["seconds"] == 1.0
    first_load = next(e for e in _stages(harness) if e["name"] == "load")
    assert harness.events.index(wave) < harness.events.index(first_load)


# ── v1.5 H19: one overall progress ───────────────────────────────────────────

def _progress(h):
    return [e["v"] for e in h.of("progress")]


def test_overall_progress_is_monotonic_and_reaches_one_only_after_export(harness, monkeypatch):
    def align(result, lang, dev, audio, **k):
        for p in (20, 60, 100):
            k["progress"](p)
        return result

    def diarize(audio, model_dir, **k):
        for f in (0.3, 1.0):
            k["progress"](f)
        return "DF"

    monkeypatch.setattr(runner.pipeline, "align", align)
    monkeypatch.setattr(runner.pipeline, "diarize", diarize)
    harness.progress_calls = [30, 100]
    runner.run(_cfg(harness, language="fr", align=True, diarize=True))
    vals = _progress(harness)
    assert vals == sorted(vals) and len(set(vals)) == len(vals)
    assert vals[-1] == 1.0 and all(v < 1.0 for v in vals[:-1])
    ev = harness.events
    last = len(ev) - 1 - ev[::-1].index({"t": "progress", "v": 1.0})
    assert ev.index(_stage(harness, "export", "done")) < last < ev.index(harness.of("done")[0])


@pytest.mark.parametrize("align,diarize", [(False, False), (True, False), (False, True), (True, True)])
def test_spans_exist_only_for_stages_that_are_on(align, diarize):
    o = runner._Overall({"align": align, "diarize": diarize}, "cuda")
    assert ("align" in o.spans, "diarize" in o.spans) == (align, diarize)
    spans = list(o.spans.values())
    assert spans[0] == (0.0, 0.04) and spans[-1] == pytest.approx((0.98, 1.0))
    assert all(a[1] == pytest.approx(b[0]) and a[0] < a[1] for a, b in zip(spans, spans[1:]))


def test_align_and_diarize_progress_land_in_their_spans(harness, monkeypatch):
    seen = {}

    def align(result, lang, dev, audio, **k):
        k["progress"](50)
        seen["align"] = _progress(harness)[-1]
        return result

    def diarize(audio, model_dir, **k):
        k["progress"](0.5)
        seen["diarize"] = _progress(harness)[-1]
        return "DF"

    monkeypatch.setattr(runner.pipeline, "align", align)
    monkeypatch.setattr(runner.pipeline, "diarize", diarize)
    cfg = _cfg(harness, language="fr", align=True, diarize=True)
    runner.run(cfg)
    spans = runner._Overall(cfg, "cpu").spans
    for name in ("align", "diarize"):
        lo, hi = spans[name]
        assert lo < seen[name] < hi


def test_detect_stage_reports_language_and_probability(harness):
    runner.run(_cfg(harness, language="auto"))
    assert _stage(harness, "detect", "start")
    assert _stage(harness, "detect", "done")["detail"] == "ur · 0.87"


def test_detect_stage_with_explicit_language_has_prob_one_and_no_detection(harness):
    runner.run(_cfg(harness, language="fr"))
    assert harness.detect_calls == []
    assert _stage(harness, "detect", "start")
    assert _stage(harness, "detect", "done")["detail"] == "fr · 1.00"


def test_transcribe_stage_reports_chunks_and_speech_seconds(harness):
    harness.segments = [{"start": 0.0, "end": 2.0}, {"start": 3.0, "end": 6.4}]
    runner.run(_cfg(harness))
    assert _stage(harness, "transcribe", "start")
    assert _stage(harness, "transcribe", "done")["detail"] == "2 chunks · 5 s speech"


def test_align_stage_emitted_when_alignment_runs(harness):
    runner.run(_cfg(harness, language="fr", align=True))
    assert _stage(harness, "align", "start")["detail"] == "default model for fr"
    assert _stage(harness, "align", "done")


@pytest.mark.parametrize("over", [
    {"align": False},
    {"align": True, "task": "translate"},
])
def test_align_stage_absent_when_alignment_does_not_run(harness, over):
    runner.run(_cfg(harness, **over))
    assert [e for e in _stages(harness) if e["name"] == "align"] == []


def test_align_stage_absent_when_language_has_no_alignment_model(harness, monkeypatch):
    monkeypatch.setattr(runner.pipeline, "align_supported", lambda lang: lang != "ps")
    runner.run(_cfg(harness, language="ps", align=True))
    assert [e for e in _stages(harness) if e["name"] == "align"] == []


def test_export_stage_lists_written_files_and_skips_failures(harness):
    a = str(harness.tmp_path / "out" / "a.srt")
    b = str(harness.tmp_path / "out" / "a.txt")
    c = str(harness.tmp_path / "out" / "a.vtt")
    harness.export_rows = [("srt", a, None), ("txt", b, None), ("vtt", c, "boom")]
    runner.run(_cfg(harness, formats=["srt", "txt", "vtt"]))
    assert _stage(harness, "export", "start")
    done = _stage(harness, "export", "done")
    assert [x.strip() for x in done["detail"].split(",")] == ["a.srt", "a.txt"]
    assert done["files"] == [a, b]


# ── v1.4 T3: alignment overrides ─────────────────────────────────────────────

def _logs(h):
    return [e["m"] for e in h.events if e.get("t") == "log"]


def test_align_defaults_forward_nearest_no_chars_and_no_model(harness):
    runner.run(_cfg(harness, language="fr", align=True))
    (_, _, _, kw), = harness.align_calls
    assert kw["interpolate_method"] == "nearest"
    assert kw["return_char_alignments"] is False
    assert not kw.get("align_model")


def test_align_options_reach_pipeline_align(harness):
    runner.run(_cfg(harness, language="fr", align=True, align_model="org/my-wav2vec2",
                    interpolate_method="linear", char_alignments=True))
    (_, _, _, kw), = harness.align_calls
    assert kw["align_model"] == "org/my-wav2vec2"
    assert kw["interpolate_method"] == "linear"
    assert kw["return_char_alignments"] is True


def test_align_model_runs_alignment_for_unsupported_language(harness, monkeypatch):
    monkeypatch.setattr(runner.pipeline, "align_supported", lambda lang: lang != "ps")
    runner.run(_cfg(harness, language="ps", align=True, align_model="org/ps-model"))
    assert len(harness.align_calls) == 1
    assert _stage(harness, "align", "done")


def test_align_model_is_logged_and_is_stage_detail(harness):
    runner.run(_cfg(harness, language="fr", align=True, align_model="org/my-wav2vec2"))
    assert _stage(harness, "align", "start")["detail"] == "org/my-wav2vec2"
    assert any("org/my-wav2vec2" in m for m in _logs(harness))


def test_unsupported_language_without_align_model_still_skipped(harness, monkeypatch):
    monkeypatch.setattr(runner.pipeline, "align_supported", lambda lang: lang != "ps")
    runner.run(_cfg(harness, language="ps", align=True))
    assert harness.align_calls == []


def test_translate_still_skips_alignment_with_align_model(harness):
    runner.run(_cfg(harness, align=True, task="translate", align_model="org/my-wav2vec2"))
    assert harness.align_calls == []
    assert [e for e in _stages(harness) if e["name"] == "align"] == []


def test_segment_resolution_defaults_to_sentence_in_export_opts(harness):
    runner.run(_cfg(harness))
    (args, _), = harness.export_calls
    assert args[-1]["segment_resolution"] == "sentence"


def test_segment_resolution_forwarded_in_export_opts(harness):
    runner.run(_cfg(harness, segment_resolution="chunk"))
    (args, _), = harness.export_calls
    assert args[-1]["segment_resolution"] == "chunk"


# ── v1.4 T4: speaker diarization ─────────────────────────────────────────────



def _two_speakers(h):
    h.assigned_segments = [
        {"start": 0.0, "end": 1.0, "text": "a", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": "b", "speaker": "SPEAKER_01"},
        {"start": 2.0, "end": 3.0, "text": "c", "speaker": "SPEAKER_00"},
    ]


def _warns(h):
    return [e["m"] for e in h.events if e.get("t") == "warn"]


def _diarize_cfg(h, **over):
    base = dict(language="fr", align=True, diarize=True)
    base.update(over)
    return _cfg(h, **base)


def test_diarize_off_makes_no_diarize_call_and_no_stage_events(harness):
    runner.run(_cfg(harness, language="fr", align=True))
    assert harness.diarize_calls == []
    assert harness.assign_calls == []
    assert [e for e in _stages(harness) if e["name"] == "diarize"] == []


def test_diarize_receives_audio_model_dir_and_speaker_range(harness):
    runner.run(_diarize_cfg(harness, min_speakers=2, max_speakers=4))
    (args, kw), = harness.diarize_calls
    assert args == (harness.audio, str(harness.model_dir))
    assert kw["min_speakers"] == 2
    assert kw["max_speakers"] == 4
    assert kw["return_embeddings"] is False
    assert "hf_token" not in kw and "model" not in kw


def test_diarize_speaker_counts_default_to_none(harness):
    runner.run(_diarize_cfg(harness))
    (_, kw), = harness.diarize_calls
    assert kw["min_speakers"] is None
    assert kw["max_speakers"] is None


def test_diarize_runs_after_alignment_and_before_export(harness):
    runner.run(_diarize_cfg(harness))
    order = [(e["name"], e["state"]) for e in _stages(harness)
             if e["name"] in ("align", "diarize", "export")]
    assert order == [("align", "start"), ("align", "done"),
                     ("diarize", "start"), ("diarize", "done"),
                     ("export", "start"), ("export", "done")]


def test_assign_speakers_gets_diarize_output_and_aligned_result(harness):
    runner.run(_diarize_cfg(harness))
    (df, result, emb), = harness.assign_calls
    assert df == "DIARIZE_DF"
    assert "segments" in result
    assert emb is None


def test_assigned_result_is_what_gets_exported(harness):
    _two_speakers(harness)
    runner.run(_diarize_cfg(harness))
    (args, _), = harness.export_calls
    assert args[0]["segments"] == harness.assigned_segments


def test_diarize_stage_detail_is_auto_without_speaker_range(harness):
    runner.run(_diarize_cfg(harness))
    assert _stage(harness, "diarize", "start")["detail"] == "auto"


def test_diarize_stage_detail_shows_range_when_both_given(harness):
    runner.run(_diarize_cfg(harness, min_speakers=2, max_speakers=4))
    assert _stage(harness, "diarize", "start")["detail"] == "2–4 speakers"


def test_diarize_stage_detail_is_auto_when_only_one_bound_given(harness):
    runner.run(_diarize_cfg(harness, min_speakers=2))
    assert _stage(harness, "diarize", "start")["detail"] == "auto"


def test_diarize_done_detail_counts_distinct_speakers(harness):
    _two_speakers(harness)
    runner.run(_diarize_cfg(harness))
    assert _stage(harness, "diarize", "done")["detail"] == "2 speakers"


def test_speaker_embeddings_true_requests_and_passes_embeddings(harness):
    runner.run(_diarize_cfg(harness, speaker_embeddings=True))
    (_, kw), = harness.diarize_calls
    assert kw["return_embeddings"] is True
    (df, _, emb), = harness.assign_calls
    assert df == "DIARIZE_DF"
    assert emb == harness.diarize_embeddings
    (args, _), = harness.export_calls
    assert args[-1]["speaker_embeddings"] == harness.diarize_embeddings


def test_speaker_embeddings_off_gives_export_no_embeddings(harness):
    runner.run(_diarize_cfg(harness))
    (args, _), = harness.export_calls
    assert not args[-1].get("speaker_embeddings")


def test_speaker_names_and_cue_flag_reach_export_opts(harness):
    names = {"SPEAKER_00": "Amina"}
    runner.run(_diarize_cfg(harness, speaker_names=names, speaker_names_in_cues=False))
    (args, _), = harness.export_calls
    assert args[-1]["speaker_names"] == names
    assert args[-1]["speaker_names_in_cues"] is False


def test_speaker_names_in_cues_defaults_to_true(harness):
    runner.run(_diarize_cfg(harness))
    (args, _), = harness.export_calls
    assert args[-1]["speaker_names_in_cues"] is True
    assert not args[-1].get("speaker_names")


def test_ready_models_are_not_downloaded(harness):
    runner.run(_diarize_cfg(harness))
    assert harness.ensure_calls == []
    assert len(harness.diarize_calls) == 1


def test_missing_models_download_with_progress_then_diarize(harness):
    harness.speakers_ready = False
    harness.ensure_progress = (0.25, 1.0)
    runner.run(_diarize_cfg(harness))
    assert harness.ensure_calls == [(str(harness.model_dir), False)]
    statuses = [e["m"] for e in harness.of("status")]
    assert "Downloading speaker models... 25%" in statuses
    assert "Downloading speaker models... 100%" in statuses
    assert len(harness.diarize_calls) == 1
    assert _warns(harness) == []


def test_download_failure_warns_and_run_continues(harness):
    harness.speakers_ready = False
    harness.ensure_error = OSError("no network\nsecond line")
    runner.run(_diarize_cfg(harness))
    assert "Speaker identification failed: no network" in _warns(harness)
    assert harness.diarize_calls == []
    assert len(harness.export_calls) == 1
    assert harness.of("done")


def test_offline_with_missing_models_warns_with_the_friendly_text(harness):
    harness.speakers_ready = False
    harness.ensure_error = RuntimeError("The speaker models are not downloaded yet. Turn off Offline only once "
                                        "to download them (about 34 MB).")
    runner.run(_diarize_cfg(harness, offline_only=True))
    assert harness.ensure_calls == [(str(harness.model_dir), True)]
    assert any("not downloaded yet" in m for m in _warns(harness))
    assert harness.diarize_calls == []
    assert harness.of("done")


def test_diarize_failure_warns_with_first_error_line_and_continues(harness):
    harness.diarize_error = RuntimeError("bad model\nsecond line of traceback")
    runner.run(_diarize_cfg(harness))
    assert "Speaker identification failed: bad model" in _warns(harness)
    assert not any("second line" in m for m in _warns(harness))
    assert harness.assign_calls == []
    assert len(harness.export_calls) == 1
    assert harness.of("done")


# ── T6: final transcript event ───────────────────────────────────────────────

def test_one_transcript_event_with_the_exported_segments(harness):
    harness.segments = [
        {"start": 0.0, "end": 1.0, "text": " hi", "speaker": "SPEAKER_00",
         "words": [{"word": "hi", "start": 0.0, "end": 0.5, "score": 0.31, "speaker": "SPEAKER_00"}]},
    ]
    runner.run(_cfg(harness, formats=["json"]))
    events = harness.of("transcript")
    assert len(events) == 1
    (args, _), = harness.export_calls
    assert events[0]["segments"] == args[0]["segments"]
    # low-score words are passed through untouched; the UI decides what is doubtful
    assert events[0]["segments"][0]["words"][0]["score"] == 0.31


def test_transcript_event_comes_before_done(harness):
    runner.run(_cfg(harness))
    kinds = [e.get("t") for e in harness.events]
    assert kinds.index("transcript") < kinds.index("done")


# ── final review: English-only models and failed detection ───────────────────

def test_english_only_model_on_auto_skips_detection(harness):
    runner.run(_cfg(harness, model="small.en", language="auto"))
    assert harness.detect_calls == []
    assert harness.of("language") == [{"t": "language", "code": "en", "prob": 1.0}]
    assert harness.transcribe_calls[0][0] == "en"


def test_detection_failure_falls_back_to_transcribe_detection(harness, monkeypatch):
    def boom(model, audio):
        raise RuntimeError("detect_language not supported")
    monkeypatch.setattr(runner.pipeline, "detect_language", boom)
    runner.run(_cfg(harness, language="auto"))
    assert harness.transcribe_calls[0][0] is None          # whisperx detects it
    assert harness.of("language") == []
    assert harness.of("done")


# ── H2: unload before align, caches, visible alignment failure ───────────────

def _marks(h):
    return [e["name"] for e in h.events if e.get("t") == "_mark"]


def test_model_unloaded_after_transcribe_and_before_align(harness):
    runner.run(_cfg(harness, language="fr", align=True))
    m = _marks(harness)
    assert m.index("transcribe") < m.index("unload") < m.index("align")


def test_cache_env_points_under_model_dir(harness):
    import os
    runner.run(_cfg(harness))
    assert os.environ["NLTK_DATA"] == str(harness.model_dir / "nltk_data")
    assert os.environ["TORCH_HOME"] == str(harness.model_dir / "torch")


def test_missing_punkt_offline_skips_alignment_with_warning(harness):
    harness.punkt = False
    runner.run(_cfg(harness, language="fr", align=True, offline_only=True))
    assert harness.punkt_calls == [(str(harness.model_dir), True)]
    assert harness.align_calls == []
    assert harness.of("warn")


def test_alignment_failure_warns_and_marks_stage_failed(harness):
    harness.align_error = RuntimeError("boom\nTraceback...")
    runner.run(_cfg(harness, language="fr", align=True))
    warns = [e["m"] for e in harness.of("warn")]
    assert "Word timestamps failed: boom; exporting segment timing only." in warns
    assert _stage(harness, "align", "done")["detail"] == "failed"
    assert harness.of("done")


# ── honest results (H3) ──────────────────────────────────────────────────────

@pytest.mark.parametrize("model", ["large-v3-turbo", "small.en", "distil-large-v3"])
def test_translate_with_unsupported_model_fails_before_loading(harness, model):
    with pytest.raises(runner.UserError, match=f"'{model}' can only transcribe"):
        runner.run(_cfg(harness, task="translate", model=model))
    assert harness.load_kwargs is None


def test_translate_with_multilingual_model_runs(harness):
    runner.run(_cfg(harness, task="translate", model="large-v3"))
    assert harness.of("done")


def test_low_confidence_language_warns_and_continues(harness):
    harness.detected = ("bo", 0.17)
    runner.run(_cfg(harness, language="auto"))
    assert "Language guess is uncertain: bo 17%. Pick the language for a better result." \
        in [e["m"] for e in harness.of("warn")]
    assert harness.of("done")


def test_confident_language_does_not_warn(harness):
    runner.run(_cfg(harness, language="auto"))
    assert not harness.of("warn")


def _oom_transcribe(monkeypatch, fail_above):
    batches = []

    def transcribe(model, audio, batch, language, **k):
        batches.append(batch)
        if batch > fail_above:
            raise RuntimeError("CUDA failed with error out of memory")
        return {"segments": [], "language": "en"}, "en"

    monkeypatch.setattr(runner.pipeline, "transcribe", transcribe)
    return batches


def test_oom_halves_batch_and_retries(harness, monkeypatch):
    batches = _oom_transcribe(monkeypatch, fail_above=4)
    runner.run(_cfg(harness, batch=16))
    assert batches == [16, 8, 4]
    warns = [e["m"] for e in harness.of("warn")]
    assert "GPU memory is full \u2014 retrying with batch 8" in warns
    assert "GPU memory is full \u2014 retrying with batch 4" in warns
    assert harness.of("done")


def test_oom_at_batch_one_is_reraised(harness, monkeypatch):
    batches = _oom_transcribe(monkeypatch, fail_above=0)
    with pytest.raises(RuntimeError, match="out of memory"):
        runner.run(_cfg(harness, batch=2))
    assert batches == [2, 1]


def test_pyannote_vad_oom_wording_also_backs_off(harness, monkeypatch):
    batches = []

    def transcribe(model, audio, batch, language, **k):
        batches.append(batch)
        if batch > 8:
            raise MemoryError("batch_size ( 32) is probably too large. Try a smaller value")
        return {"segments": [], "language": "en"}, "en"

    monkeypatch.setattr(runner.pipeline, "transcribe", transcribe)
    runner.run(_cfg(harness, batch=16))
    assert batches == [16, 8]


def test_follow_on_error_right_after_oom_is_retried_once_at_same_batch(harness, monkeypatch):
    batches = []
    errors = iter([RuntimeError("CUDA failed with error out of memory"),
                   RuntimeError("parallel_for failed: invalid device ordinal"), None])

    def transcribe(model, audio, batch, language, **k):
        batches.append(batch)
        err = next(errors)
        if err:
            raise err
        return {"segments": [], "language": "en"}, "en"

    monkeypatch.setattr(runner.pipeline, "transcribe", transcribe)
    runner.run(_cfg(harness, batch=16))
    assert batches == [16, 8, 8]


def test_other_errors_are_not_retried(harness, monkeypatch):
    calls = []

    def transcribe(*a, **k):
        calls.append(a[2])
        raise RuntimeError("boom")

    monkeypatch.setattr(runner.pipeline, "transcribe", transcribe)
    with pytest.raises(RuntimeError, match="boom"):
        runner.run(_cfg(harness, batch=16))
    assert calls == [16]


def test_default_batch_drops_to_8_on_small_gpu(harness, monkeypatch):
    seen = []
    monkeypatch.setattr(runner.pipeline, "detect_device",
                        lambda req: ("cuda", "cuda", "cuda"))
    monkeypatch.setattr(runner.pipeline, "initial_batch",
                        lambda b, d, i=0: 8 if (b, d) == (16, "cuda") else b)
    monkeypatch.setattr(runner.pipeline, "transcribe",
                        lambda m, a, batch, lang, **k: (seen.append(batch) or
                                                         ({"segments": []}, "en")))
    runner.run(_cfg(harness, batch=16, device="cuda"))
    assert seen == [8]
    assert any("batch 8" in e["m"] for e in harness.of("log"))


def test_initial_batch_only_changes_default_on_small_cuda(monkeypatch):
    import types
    from core import pipeline
    props = types.SimpleNamespace(total_memory=8 * 1024 ** 3)
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(
        get_device_properties=lambda i: props))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert pipeline.initial_batch(16, "cuda") == 8
    assert pipeline.initial_batch(4, "cuda") == 4
    assert pipeline.initial_batch(16, "cpu") == 16
    props.total_memory = 24 * 1024 ** 3
    assert pipeline.initial_batch(16, "cuda") == 16


GiB = 1024 ** 3


@pytest.mark.parametrize("free,total,want", [
    (6, 8, 8), (2, 8, 4), (1, 8, 2), (20, 24, 16)])
def test_initial_batch_uses_free_memory(free, total, want):
    from core import pipeline
    assert pipeline.initial_batch(16, "cuda", memory=(free * GiB, total * GiB)) == want


def test_initial_batch_keeps_configured_batch_and_cpu():
    from core import pipeline
    assert pipeline.initial_batch(32, "cuda", memory=(1 * GiB, 8 * GiB)) == 32
    assert pipeline.initial_batch(16, "cpu", memory=(1 * GiB, 8 * GiB)) == 16


def test_initial_batch_without_mem_get_info_uses_total_only(monkeypatch):
    import types
    from core import pipeline
    props = types.SimpleNamespace(total_memory=8 * GiB)
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(
        get_device_properties=lambda i: props))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert pipeline.initial_batch(16, "cuda", memory=None) == 8
    props.total_memory = 24 * GiB
    assert pipeline.initial_batch(16, "cuda", memory=None) == 16


def test_initial_batch_reads_free_memory_from_torch(monkeypatch):
    import types
    from core import pipeline
    seen = []
    props = types.SimpleNamespace(total_memory=8 * GiB)

    def mem_get_info(i):
        seen.append(i)
        return (1 * GiB, 8 * GiB)
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(
        get_device_properties=lambda i: props, mem_get_info=mem_get_info))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert pipeline.initial_batch(16, "cuda", 1, memory=None) == 2
    assert seen == [1]


@pytest.mark.parametrize("model,ok", [
    ("large-v3", True), ("medium", True), ("small", True),
    ("large-v3-turbo", False), ("small.en", False), ("distil-large-v3", False)])
def test_supports_translate(model, ok):
    from core import pipeline
    assert pipeline.supports_translate(model) is ok


def test_main_error_event_is_first_line_and_traceback_goes_to_log(harness, monkeypatch):
    import io
    import json

    def boom(cfg):
        raise RuntimeError("first line\nsecond line")

    monkeypatch.setattr(runner, "run", boom)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"x": 1}) + "\n"))
    with pytest.raises(SystemExit) as ex:
        runner.main()
    assert ex.value.code == 1
    assert harness.of("error")[0]["m"] == "first line"
    assert any("Traceback" in e["m"] and "second line" in e["m"] for e in harness.of("log"))


# ── unreadable audio gets a plain sentence (R2) ──────────────────────────────

_BANNER = ("ffmpeg version 2025-09-01-git-abc Copyright (c) 2000-2025 the FFmpeg developers\n"
           "  built with gcc 14\n  configuration: --enable-gpl\n")


def _run_main(harness, monkeypatch, cfg):
    import io
    import json
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(cfg) + "\n"))
    with pytest.raises(SystemExit) as ex:
        runner.main()
    assert ex.value.code == 1


def test_missing_audio_file_raises_plain_user_error_before_ffmpeg(harness, monkeypatch):
    calls = []
    monkeypatch.setattr(runner.pipeline, "load_audio", lambda path: calls.append(path))
    cfg = _cfg(harness, audio=str(harness.tmp_path / "gone.wav"))
    with pytest.raises(runner.UserError) as ex:
        runner.run(cfg)
    assert str(ex.value) == "Could not read this file: it no longer exists."
    assert calls == []


@pytest.mark.parametrize("reason", [
    "Invalid data found when processing input",
    "Unsupported codec for this stream"])
def test_failed_load_audio_reports_last_line_as_reason(harness, monkeypatch, reason):
    def bad(path):
        raise RuntimeError("Failed to load audio: " + _BANNER + reason + "\n\n")

    monkeypatch.setattr(runner.pipeline, "load_audio", bad)
    with pytest.raises(runner.UserError) as ex:
        runner.run(_cfg(harness))
    assert str(ex.value) == f"Could not read this file: {reason}"


def test_failed_load_audio_strips_path_prefix_from_reason(harness, monkeypatch):
    def bad(path):
        raise RuntimeError("Failed to load audio: " + _BANNER
                           + f"{path}: Invalid data found when processing input\n")

    monkeypatch.setattr(runner.pipeline, "load_audio", bad)
    with pytest.raises(runner.UserError) as ex:
        runner.run(_cfg(harness))
    assert str(ex.value) == "Could not read this file: Invalid data found when processing input"


def test_other_load_audio_errors_propagate_unchanged(harness, monkeypatch):
    err = ValueError("something else broke")

    def bad(path):
        raise err

    monkeypatch.setattr(runner.pipeline, "load_audio", bad)
    with pytest.raises(ValueError) as ex:
        runner.run(_cfg(harness))
    assert ex.value is err


def test_error_event_is_the_single_plain_sentence(harness, monkeypatch):
    def bad(path):
        raise RuntimeError("Failed to load audio: " + _BANNER
                           + f"{path}: Invalid data found when processing input\n")

    monkeypatch.setattr(runner.pipeline, "load_audio", bad)
    _run_main(harness, monkeypatch, _cfg(harness))
    assert harness.of("error")[0]["m"] == \
        "Could not read this file: Invalid data found when processing input"
    assert not any("Traceback" in e["m"] for e in harness.of("log"))


def test_missing_file_error_event_is_the_single_plain_sentence(harness, monkeypatch):
    _run_main(harness, monkeypatch, _cfg(harness, audio=str(harness.tmp_path / "gone.wav")))
    assert harness.of("error")[0]["m"] == "Could not read this file: it no longer exists."
