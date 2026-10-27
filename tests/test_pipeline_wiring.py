"""
Wiring tests for core/pipeline.py: kwargs passed to whisperx / huggingface_hub
plus the pure model-repo and alignment-language helpers.

whisperx and huggingface_hub are replaced by fakes; no network, no models.
"""
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import pipeline
from core.pipeline import (ALIGN_LANGS, align_supported, MODEL_INFO,
                           model_repo, model_cache_dir)


# ── helpers ───────────────────────────────────────────────────────────────────

def _install_fake_whisperx(monkeypatch):
    calls = []
    fake = types.ModuleType("whisperx")

    def load_model(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    fake.load_model = load_model
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    return calls


class _FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append((audio, kw))
        return self.result


# ── alignment languages ───────────────────────────────────────────────────────

@pytest.mark.parametrize("lang", ["en", "ka", "auto", None])
def test_align_supported_true(lang):
    assert align_supported(lang) is True


def test_align_supported_false_for_pashto():
    assert align_supported("ps") is False


def test_ps_not_in_align_langs():
    assert "ps" not in ALIGN_LANGS


# ── model repo / cache dir ────────────────────────────────────────────────────

def test_model_repo_turbo():
    assert model_repo("large-v3-turbo") == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"


def test_model_repo_distil():
    assert model_repo("distil-large-v3") == "Systran/faster-distil-whisper-large-v3"


def test_model_repo_unknown_falls_back_to_systran():
    assert model_repo("made-up") == "Systran/faster-whisper-made-up"


def test_model_cache_dir_uses_repo_name():
    assert model_cache_dir("large-v3-turbo", "M") == os.path.join(
        "M", "hub", "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo")


def test_model_info_contains_all_models_with_valid_values():
    for name in ["tiny", "base", "small", "medium", "large-v2", "large-v3",
                 "large-v3-turbo", "distil-large-v3"]:
        assert name in MODEL_INFO
    for name, value in MODEL_INFO.items():
        assert isinstance(value, tuple) and len(value) == 2, name
        assert isinstance(value[0], str), name
        assert isinstance(value[1], int) and value[1] > 0, name


# ── load_model wiring ─────────────────────────────────────────────────────────

def test_load_model_passes_all_options_to_whisperx(monkeypatch):
    calls = _install_fake_whisperx(monkeypatch)
    pipeline.load_model("small", "cpu", "int8", "en", 3,
                        vad_onset=0.4, vad_offset=0.3, chunk_size=20,
                        task="translate", initial_prompt="Hello")
    assert len(calls) == 1
    args, kw = calls[0]
    assert args == ("small", "cpu")
    assert kw == {
        "compute_type": "int8",
        "language": "en",
        "task": "translate",
        "vad_options": {"vad_onset": 0.4, "vad_offset": 0.3, "chunk_size": 20},
        "asr_options": {"beam_size": 3, "initial_prompt": "Hello", "hotwords": None,
                        "suppress_numerals": False},
        "vad_method": "pyannote",
        "device_index": 0,
        "threads": 4,
        "local_files_only": False,
    }


def test_load_model_defaults(monkeypatch):
    calls = _install_fake_whisperx(monkeypatch)
    pipeline.load_model("small", "cpu", "int8", "en", 5)
    args, kw = calls[0]
    assert args == ("small", "cpu")
    assert kw["task"] == "transcribe"
    assert kw["vad_options"] == {"vad_onset": 0.5, "vad_offset": 0.363, "chunk_size": 30}
    assert kw["asr_options"] == {"beam_size": 5, "initial_prompt": None, "hotwords": None,
                                 "suppress_numerals": False}
    assert set(kw) == {"compute_type", "language", "task", "vad_options", "asr_options",
                       "vad_method", "device_index", "threads", "local_files_only"}
    assert kw["vad_method"] == "pyannote"
    assert kw["device_index"] == 0
    assert kw["threads"] == 4
    assert kw["local_files_only"] is False


def test_load_model_empty_initial_prompt_becomes_none(monkeypatch):
    calls = _install_fake_whisperx(monkeypatch)
    pipeline.load_model("small", "cpu", "int8", "en", 5, initial_prompt="")
    assert calls[0][1]["asr_options"]["initial_prompt"] is None


def test_load_model_forwards_hotwords_and_suppress_numerals(monkeypatch):
    calls = _install_fake_whisperx(monkeypatch)
    pipeline.load_model("small", "cpu", "int8", "en", 5,
                        hotwords="Kubernetes, WhisperX", suppress_numerals=True)
    ao = calls[0][1]["asr_options"]
    assert ao["hotwords"] == "Kubernetes, WhisperX"
    assert ao["suppress_numerals"] is True


def test_load_model_empty_hotwords_becomes_none(monkeypatch):
    calls = _install_fake_whisperx(monkeypatch)
    pipeline.load_model("small", "cpu", "int8", "en", 5, hotwords="")
    assert calls[0][1]["asr_options"]["hotwords"] is None


def test_load_model_suppress_numerals_is_coerced_to_bool(monkeypatch):
    calls = _install_fake_whisperx(monkeypatch)
    pipeline.load_model("small", "cpu", "int8", "en", 5, suppress_numerals=1)
    assert calls[0][1]["asr_options"]["suppress_numerals"] is True


_FULL_DECODE = {
    "temperatures": [0.0, 0.2, 0.4],
    "best_of": 3,
    "patience": 1.5,
    "length_penalty": 0.9,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "compression_ratio_threshold": 2.2,
    "log_prob_threshold": -0.8,
    "no_speech_threshold": 0.5,
    "condition_on_previous_text": False,
    "prompt_reset_on_temperature": 0.7,
    "hallucination_silence_threshold": 2.0,
    "max_new_tokens": 128,
}


def test_load_model_copies_every_decode_key_into_asr_options(monkeypatch):
    calls = _install_fake_whisperx(monkeypatch)
    pipeline.load_model("small", "cpu", "int8", "en", 4, hotwords="hw",
                        suppress_numerals=True, decode=dict(_FULL_DECODE))
    ao = calls[0][1]["asr_options"]
    expected = {"beam_size": 4, "initial_prompt": None, "hotwords": "hw",
                "suppress_numerals": True}
    expected.update(_FULL_DECODE)
    assert ao == expected


def test_load_model_partial_decode_leaves_other_keys_absent(monkeypatch):
    calls = _install_fake_whisperx(monkeypatch)
    pipeline.load_model("small", "cpu", "int8", "en", 5,
                        decode={"best_of": 2, "no_speech_threshold": 0.3})
    ao = calls[0][1]["asr_options"]
    assert ao["best_of"] == 2
    assert ao["no_speech_threshold"] == 0.3
    for key in _FULL_DECODE:
        if key not in ("best_of", "no_speech_threshold"):
            assert key not in ao, key


@pytest.mark.parametrize("decode", [None, {}])
def test_load_model_without_decode_has_no_decode_keys(monkeypatch, decode):
    calls = _install_fake_whisperx(monkeypatch)
    pipeline.load_model("small", "cpu", "int8", "en", 5, decode=decode)
    ao = calls[0][1]["asr_options"]
    assert set(ao) == {"beam_size", "initial_prompt", "hotwords", "suppress_numerals"}
    assert "temperatures" not in ao


def test_load_model_forwards_vad_method_device_index_threads_offline(monkeypatch):
    calls = _install_fake_whisperx(monkeypatch)
    pipeline.load_model("small", "cuda", "float16", "en", 5, vad_method="silero",
                        device_index=1, threads=8, local_files_only=True)
    kw = calls[0][1]
    assert kw["vad_method"] == "silero"
    assert kw["device_index"] == 1
    assert kw["threads"] == 8
    assert kw["local_files_only"] is True


# ── detect_language ───────────────────────────────────────────────────────────

class _FakeAudio:
    """Tiny stand-in for a numpy array: .shape and slicing only."""
    def __init__(self, n):
        self.shape = (n,)

    def __getitem__(self, sl):
        return _FakeAudio(len(range(*sl.indices(self.shape[0]))))


def _install_fake_whisperx_audio(monkeypatch, n_samples=480000):
    mel_calls = []
    fake = types.ModuleType("whisperx")
    fake_audio = types.ModuleType("whisperx.audio")
    fake_audio.N_SAMPLES = n_samples

    def log_mel_spectrogram(audio, n_mels=80, padding=0, **kw):
        mel_calls.append({"audio": audio, "n_mels": n_mels, "padding": padding})
        return "MEL"

    fake_audio.log_mel_spectrogram = log_mel_spectrogram
    fake.audio = fake_audio
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    monkeypatch.setitem(sys.modules, "whisperx.audio", fake_audio)
    return mel_calls


class _FakeCT2:
    def __init__(self, results):
        self.results = results
        self.seen = []

    def detect_language(self, enc):
        self.seen.append(enc)
        return self.results


class _FakeFW:
    def __init__(self, ct2, feat_kwargs):
        self.model = ct2
        self.feat_kwargs = feat_kwargs
        self.encoded = []

    def encode(self, segment):
        self.encoded.append(segment)
        return "ENC"


class _FakePipeline:
    def __init__(self, fw):
        self.model = fw


def test_detect_language_returns_code_and_probability(monkeypatch):
    _install_fake_whisperx_audio(monkeypatch)
    ct2 = _FakeCT2([[("<|ur|>", 0.87)]])
    model = _FakePipeline(_FakeFW(ct2, {"feature_size": 80}))
    code, prob = pipeline.detect_language(model, _FakeAudio(480000))
    assert code == "ur"
    assert prob == pytest.approx(0.87)
    assert isinstance(prob, float)
    assert ct2.seen == ["ENC"]
    assert model.model.encoded == ["MEL"]


def test_detect_language_pads_short_audio_and_uses_model_mels(monkeypatch):
    mel_calls = _install_fake_whisperx_audio(monkeypatch)
    model = _FakePipeline(_FakeFW(_FakeCT2([[("<|en|>", 0.5)]]), {"feature_size": 128}))
    pipeline.detect_language(model, _FakeAudio(100000))
    assert mel_calls[0]["n_mels"] == 128
    assert mel_calls[0]["padding"] == 480000 - 100000


def test_detect_language_truncates_long_audio_without_padding(monkeypatch):
    mel_calls = _install_fake_whisperx_audio(monkeypatch)
    model = _FakePipeline(_FakeFW(_FakeCT2([[("<|fr|>", 0.9)]]), {}))
    pipeline.detect_language(model, _FakeAudio(700000))
    assert mel_calls[0]["audio"].shape == (480000,)
    assert mel_calls[0]["padding"] == 0
    assert mel_calls[0]["n_mels"] == 80


# ── transcribe wiring ─────────────────────────────────────────────────────────

def test_transcribe_passes_chunk_size_task_and_language():
    result = {"segments": [], "language": "fr"}
    model = _FakeModel(result)
    out = pipeline.transcribe(model, "AUDIO", 4, "fr", chunk_size=15, task="translate")
    assert len(model.calls) == 1
    audio, kw = model.calls[0]
    assert audio == "AUDIO"
    assert kw == {"batch_size": 4, "chunk_size": 15, "task": "translate", "language": "fr"}
    assert out == (result, "fr")


def test_transcribe_without_language_omits_kwarg_and_uses_detected():
    result = {"segments": [], "language": "de"}
    model = _FakeModel(result)
    out = pipeline.transcribe(model, "AUDIO", 4, None)
    _, kw = model.calls[0]
    assert "language" not in kw
    assert kw == {"batch_size": 4, "chunk_size": 30, "task": "transcribe"}
    assert out == (result, "de")


# ── download_model wiring ─────────────────────────────────────────────────────

def test_download_model_uses_model_repo_and_hub_cache(monkeypatch, tmp_path):
    recorded = {}
    fake = types.ModuleType("huggingface_hub")

    def snapshot_download(repo, **kw):
        recorded["repo"] = repo
        recorded["kw"] = kw
        return str(tmp_path)

    class HfApi:
        def model_info(self, *a, **k):
            raise RuntimeError("no network")

    fake.snapshot_download = snapshot_download
    fake.HfApi = HfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    pipeline.download_model("large-v3-turbo", str(tmp_path))

    assert recorded["repo"] == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert recorded["kw"]["cache_dir"] == os.path.join(str(tmp_path), "hub")


# ── v1.4 T5: distil models, Audacity export, cached-model inventory ───────────

@pytest.mark.parametrize("name,repo", [
    ("distil-large-v3.5", "distil-whisper/distil-large-v3.5-ct2"),
    ("distil-medium.en", "Systran/faster-distil-whisper-medium.en"),
    ("distil-small.en", "Systran/faster-distil-whisper-small.en"),
])
def test_distil_models_in_model_info_with_repo_and_size(name, repo):
    assert name in MODEL_INFO
    assert MODEL_INFO[name][0] == repo
    assert isinstance(MODEL_INFO[name][1], int) and MODEL_INFO[name][1] > 0
    assert model_repo(name) == repo


def test_export_formats_include_aud():
    assert "aud" in pipeline.EXPORT_FORMATS


def test_export_aud_routes_to_audacity_writer(monkeypatch, tmp_path):
    recorded = []
    fake_utils = types.ModuleType("whisperx.utils")

    def get_writer(fmt, out_dir):
        recorded.append((fmt, out_dir))
        return lambda result, audio_path, opts: None

    fake_utils.get_writer = get_writer
    fake_wx = types.ModuleType("whisperx")
    fake_wx.utils = fake_utils
    monkeypatch.setitem(sys.modules, "whisperx", fake_wx)
    monkeypatch.setitem(sys.modules, "whisperx.utils", fake_utils)

    out_dir = str(tmp_path / "out")
    outcomes = pipeline.export({"segments": []}, ["aud"], out_dir,
                               str(tmp_path / "talk.wav"), {})

    assert recorded == [("aud", out_dir)]
    assert outcomes == [("aud", os.path.join(out_dir, "talk.aud"), None)]


def _make_repo(model_dir, repo_dir_name, files):
    """files: {relative path inside the repo folder: size in bytes}"""
    root = Path(model_dir) / "hub" / repo_dir_name
    for rel, size in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)
    return root


def test_cached_models_lists_whisper_model_by_model_info_name(tmp_path):
    root = _make_repo(tmp_path, "models--Systran--faster-whisper-small", {
        "blobs/abc123": 1000,
        "snapshots/rev1/model.bin": 500,
        "snapshots/rev1/config.json": 24,
    })
    found = pipeline.cached_models(str(tmp_path))
    assert len(found) == 1
    entry = found[0]
    assert entry["name"] == "small"
    assert entry["kind"] == "whisper"
    assert entry["size_bytes"] == 1524
    assert os.path.normcase(os.path.abspath(entry["path"])) == \
        os.path.normcase(os.path.abspath(str(root)))


def test_cached_models_classifies_align_diarize_vad_and_ignores_other_folders(tmp_path):
    _make_repo(tmp_path, "models--jonatasgrosman--wav2vec2-large-xlsr-53-english",
               {"blobs/a": 10})
    _make_repo(tmp_path, "models--pyannote--speaker-diarization-3.1", {"blobs/b": 20})
    _make_repo(tmp_path, "models--pyannote--segmentation", {"blobs/c": 30})
    (tmp_path / "hub" / ".locks").mkdir(parents=True)
    (tmp_path / "hub" / ".locks" / "junk").write_bytes(b"x" * 5)

    found = pipeline.cached_models(str(tmp_path))
    kinds = {e["name"]: e["kind"] for e in found}
    assert kinds == {
        "jonatasgrosman/wav2vec2-large-xlsr-53-english": "align",
        "pyannote/speaker-diarization-3.1": "diarize",
        "pyannote/segmentation": "vad",
    }
    sizes = {e["name"]: e["size_bytes"] for e in found}
    assert sizes["pyannote/segmentation"] == 30


def test_cached_models_sorted_by_kind_then_name(tmp_path):
    _make_repo(tmp_path, "models--pyannote--speaker-diarization-3.1", {"blobs/b": 1})
    _make_repo(tmp_path, "models--Systran--faster-whisper-tiny", {"blobs/a": 1})
    _make_repo(tmp_path, "models--Systran--faster-whisper-base", {"blobs/a": 1})
    found = pipeline.cached_models(str(tmp_path))
    assert [(e["kind"], e["name"]) for e in found] == sorted(
        (e["kind"], e["name"]) for e in found)
    assert {e["name"] for e in found} == {
        "base", "tiny", "pyannote/speaker-diarization-3.1"}


def test_cached_models_missing_hub_returns_empty_list(tmp_path):
    assert pipeline.cached_models(str(tmp_path)) == []


def test_delete_cached_model_removes_folder(tmp_path):
    root = _make_repo(tmp_path, "models--Systran--faster-whisper-small",
                      {"blobs/a": 10, "snapshots/r/model.bin": 10})
    keep = _make_repo(tmp_path, "models--Systran--faster-whisper-tiny", {"blobs/a": 10})
    pipeline.delete_cached_model(str(tmp_path), "small")
    assert not root.exists()
    assert keep.exists()
    assert [e["name"] for e in pipeline.cached_models(str(tmp_path))] == ["tiny"]


def test_delete_cached_model_unknown_name_raises_key_error(tmp_path):
    _make_repo(tmp_path, "models--Systran--faster-whisper-small", {"blobs/a": 10})
    with pytest.raises(KeyError):
        pipeline.delete_cached_model(str(tmp_path), "large-v3")


# ── v1.4 T2: progress callback and waveform bins ─────────────────────────────

def test_transcribe_passes_progress_callback_to_model():
    model = _FakeModel({"segments": [], "language": "en"})
    cb = lambda pct: None
    pipeline.transcribe(model, "AUDIO", 4, "en", progress_cb=cb)
    assert model.calls[0][1]["progress_callback"] is cb


def test_transcribe_without_progress_cb_omits_progress_callback_kwarg():
    model = _FakeModel({"segments": [], "language": "en"})
    pipeline.transcribe(model, "AUDIO", 4, "en")
    assert "progress_callback" not in model.calls[0][1]
    model2 = _FakeModel({"segments": [], "language": "en"})
    pipeline.transcribe(model2, "AUDIO", 4, "en", progress_cb=None)
    assert "progress_callback" not in model2.calls[0][1]


def test_waveform_bins_returns_peak_abs_per_slice_normalised_to_one():
    audio = [0.0] * 400
    audio[5] = 0.1
    audio[150] = -0.5
    audio[250] = 0.25
    out = pipeline.waveform_bins(audio, n=4)
    assert out == pytest.approx([0.2, 1.0, 0.5, 0.0])


def test_waveform_bins_default_length_is_200():
    out = pipeline.waveform_bins([0.1, -0.2, 0.3, 0.4] * 500)
    assert len(out) == 200
    assert all(isinstance(x, float) for x in out)
    assert max(out) == pytest.approx(1.0)
    assert min(out) >= 0.0


def test_waveform_bins_respects_custom_n():
    assert len(pipeline.waveform_bins([0.5] * 1000, n=50)) == 50


def test_waveform_bins_all_zero_audio_gives_all_zero_bins():
    out = pipeline.waveform_bins([0.0] * 1000, n=10)
    assert out == [0.0] * 10


def test_waveform_bins_short_audio_is_padded_with_zeros_to_n():
    out = pipeline.waveform_bins([0.5, -1.0, 0.25], n=10)
    assert len(out) == 10
    assert all(isinstance(x, float) for x in out)
    assert sorted(x for x in out if x) == pytest.approx([0.25, 0.5, 1.0])
    assert out.count(0.0) == 7


def test_waveform_bins_empty_audio_gives_n_zeros():
    assert pipeline.waveform_bins([], n=8) == [0.0] * 8


# ── v1.4 T3: alignment overrides ─────────────────────────────────────────────

def _install_fake_align(monkeypatch, fail_on=()):
    """Fake whisperx with load_align_model/align recording kwargs.
    Devices in fail_on make load_align_model raise."""
    rec = {"load": [], "align": []}
    fake = types.ModuleType("whisperx")

    def load_align_model(*args, **kwargs):
        rec["load"].append((args, kwargs))
        if kwargs.get("device") in fail_on:
            raise RuntimeError("device unavailable")
        return "MODEL_A", {"meta": 1}

    def align(*args, **kwargs):
        rec["align"].append((args, kwargs))
        return {"segments": ["aligned"]}

    fake.load_align_model = load_align_model
    fake.align = align
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    return rec


def test_align_passes_model_name_when_align_model_set(monkeypatch):
    rec = _install_fake_align(monkeypatch)
    pipeline.align({"segments": ["s"]}, "fr", "cpu", "AUDIO", align_model="org/w2v2")
    (_, kw), = rec["load"]
    assert kw["model_name"] == "org/w2v2"
    assert kw["language_code"] == "fr"
    assert kw["device"] == "cpu"


@pytest.mark.parametrize("model", [None, ""])
def test_align_omits_model_name_when_align_model_falsy(monkeypatch, model):
    rec = _install_fake_align(monkeypatch)
    pipeline.align({"segments": ["s"]}, "fr", "cpu", "AUDIO", align_model=model)
    (_, kw), = rec["load"]
    assert "model_name" not in kw


def test_align_omits_model_name_by_default(monkeypatch):
    rec = _install_fake_align(monkeypatch)
    pipeline.align({"segments": ["s"]}, "fr", "cpu", "AUDIO")
    (_, kw), = rec["load"]
    assert "model_name" not in kw


def test_align_forwards_interpolate_method_and_char_alignments(monkeypatch):
    rec = _install_fake_align(monkeypatch)
    pipeline.align({"segments": ["s"]}, "fr", "cpu", "AUDIO",
                   interpolate_method="linear", return_char_alignments=True)
    (args, kw), = rec["align"]
    assert args == (["s"], "MODEL_A", {"meta": 1}, "AUDIO", "cpu")
    assert kw == {"interpolate_method": "linear", "return_char_alignments": True}


def test_align_defaults_are_nearest_and_no_char_alignments(monkeypatch):
    rec = _install_fake_align(monkeypatch)
    pipeline.align({"segments": ["s"]}, "fr", "cpu", "AUDIO")
    (_, kw), = rec["align"]
    assert kw == {"interpolate_method": "nearest", "return_char_alignments": False}


def test_align_cpu_fallback_keeps_same_options(monkeypatch):
    rec = _install_fake_align(monkeypatch, fail_on=("cuda",))
    out = pipeline.align({"segments": ["s"]}, "fr", "cuda", "AUDIO",
                         align_model="org/w2v2", interpolate_method="ignore",
                         return_char_alignments=True)
    assert [kw["device"] for _, kw in rec["load"]] == ["cuda", "cpu"]
    assert rec["load"][1][1]["model_name"] == "org/w2v2"
    (args, kw), = rec["align"]
    assert args[-1] == "cpu"
    assert kw == {"interpolate_method": "ignore", "return_char_alignments": True}
    assert out["language"] == "fr"


def test_align_preserves_language_key(monkeypatch):
    _install_fake_align(monkeypatch)
    out = pipeline.align({"segments": ["s"]}, "fr", "cpu", "AUDIO", align_model="x/y")
    assert out["language"] == "fr"


def _fake_writer_capture(monkeypatch):
    seen = []
    fake_utils = types.ModuleType("whisperx.utils")
    fake_utils.get_writer = lambda fmt, out_dir: (
        lambda result, audio_path, opts: seen.append(dict(opts)))
    fake_wx = types.ModuleType("whisperx")
    fake_wx.utils = fake_utils
    monkeypatch.setitem(sys.modules, "whisperx", fake_wx)
    monkeypatch.setitem(sys.modules, "whisperx.utils", fake_utils)
    return seen


def test_export_passes_segment_resolution_to_writer_options(monkeypatch, tmp_path):
    seen = _fake_writer_capture(monkeypatch)
    pipeline.export({"segments": []}, ["srt"], str(tmp_path / "o"),
                    str(tmp_path / "a.wav"), {"segment_resolution": "chunk"})
    assert seen[0]["segment_resolution"] == "chunk"


def test_export_segment_resolution_defaults_to_sentence(monkeypatch, tmp_path):
    seen = _fake_writer_capture(monkeypatch)
    pipeline.export({"segments": []}, ["srt"], str(tmp_path / "o"),
                    str(tmp_path / "a.wav"), {})
    assert seen[0]["segment_resolution"] == "sentence"
    assert set(seen[0]) >= {"max_line_width", "max_line_count", "highlight_words"}


# ── v1.4 T4: diarization wrappers and speaker names at export ────────────────

def _install_fake_diarize(monkeypatch):
    """Fake whisperx (assign_word_speakers only). Returns a dict that records every call."""
    rec = {"assign": []}

    def assign_word_speakers(diarize_df, result, speaker_embeddings=None):
        rec["assign"].append((diarize_df, result, speaker_embeddings))
        return {"assigned": True}

    fake_wx = types.ModuleType("whisperx")
    fake_wx.assign_word_speakers = assign_word_speakers
    monkeypatch.setitem(sys.modules, "whisperx", fake_wx)
    return rec


def _install_fake_sherpa(monkeypatch, found=2):
    """Fake sherpa_onnx. `found` = how many speakers it finds when the cluster count is not fixed."""
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    rec = {"runs": [], "found": found}

    def cfgclass(name):
        def init(self, **kw):
            self.__dict__.update(kw)
        return type(name, (), {"__init__": init})

    class Seg:
        def __init__(self, start, end, speaker):
            self.start, self.end, self.speaker = start, end, speaker

    class Result:
        def __init__(self, n):
            self.segs = [Seg(float(i), i + 1.0, i % n) for i in range(n * 2)]

        def sort_by_start_time(self):
            return self.segs

    class Diarization:
        def __init__(self, cfg):
            self.cfg = cfg

        def process(self, audio):
            n = self.cfg.clustering.num_clusters
            rec["runs"].append(n)
            return Result(n if n > 0 else rec["found"])

    class Stream:
        def accept_waveform(self, rate, clip):
            pass

        def input_finished(self):
            pass

    class Extractor:
        def __init__(self, cfg):
            pass

        def create_stream(self):
            return Stream()

        def compute(self, stream):
            return [0.5, 0.25]

    fake = types.ModuleType("sherpa_onnx")
    for name in ("OfflineSpeakerSegmentationPyannoteModelConfig", "OfflineSpeakerSegmentationModelConfig",
                 "SpeakerEmbeddingExtractorConfig", "OfflineSpeakerDiarizationConfig", "FastClusteringConfig"):
        setattr(fake, name, cfgclass(name))
    fake.OfflineSpeakerDiarization = Diarization
    fake.SpeakerEmbeddingExtractor = Extractor
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)
    return rec


def _audio():
    import numpy as np
    return np.zeros(16000 * 40, dtype="float32")


def test_diarize_returns_speaker_xx_labels(monkeypatch, tmp_path):
    _install_fake_sherpa(monkeypatch, found=2)
    df = pipeline.diarize(_audio(), str(tmp_path))
    assert list(df.columns) == ["start", "end", "speaker"]
    assert set(df["speaker"]) == {"SPEAKER_00", "SPEAKER_01"}


def test_diarize_min_equal_max_fixes_the_cluster_count(monkeypatch, tmp_path):
    rec = _install_fake_sherpa(monkeypatch, found=2)
    df = pipeline.diarize(_audio(), str(tmp_path), min_speakers=3, max_speakers=3)
    assert rec["runs"] == [3]
    assert df["speaker"].nunique() == 3


def test_diarize_auto_makes_one_pass_with_no_fixed_count(monkeypatch, tmp_path):
    rec = _install_fake_sherpa(monkeypatch, found=2)
    pipeline.diarize(_audio(), str(tmp_path))
    assert rec["runs"] == [-1]


def test_diarize_range_clamps_too_few_speakers_up_to_the_minimum(monkeypatch, tmp_path):
    rec = _install_fake_sherpa(monkeypatch, found=1)
    df = pipeline.diarize(_audio(), str(tmp_path), min_speakers=2, max_speakers=4)
    assert rec["runs"] == [-1, 2]
    assert df["speaker"].nunique() == 2


def test_diarize_range_clamps_too_many_speakers_down_to_the_maximum(monkeypatch, tmp_path):
    rec = _install_fake_sherpa(monkeypatch, found=6)
    df = pipeline.diarize(_audio(), str(tmp_path), min_speakers=2, max_speakers=4)
    assert rec["runs"] == [-1, 4]
    assert df["speaker"].nunique() == 4


def test_diarize_inside_the_range_makes_one_pass(monkeypatch, tmp_path):
    rec = _install_fake_sherpa(monkeypatch, found=3)
    pipeline.diarize(_audio(), str(tmp_path), min_speakers=2, max_speakers=4)
    assert rec["runs"] == [-1]


def test_diarize_returns_embeddings_dict_when_asked(monkeypatch, tmp_path):
    _install_fake_sherpa(monkeypatch, found=2)
    df, emb = pipeline.diarize(_audio(), str(tmp_path), return_embeddings=True)
    assert set(emb) == set(df["speaker"]) == {"SPEAKER_00", "SPEAKER_01"}
    assert emb["SPEAKER_00"] == [0.5, 0.25]


def test_diarize_progress_reaches_sherpa_callback(monkeypatch, tmp_path):
    rec = _install_fake_sherpa(monkeypatch, found=2)
    base = sys.modules["sherpa_onnx"].OfflineSpeakerDiarization

    class WithCallback(base):
        def process(self, audio, callback=None):
            rec["rets"] = [callback(done, 4) for done in (1, 4)]
            return super().process(audio)

    monkeypatch.setattr(sys.modules["sherpa_onnx"], "OfflineSpeakerDiarization", WithCallback)
    got = []
    pipeline.diarize(_audio(), str(tmp_path), progress=got.append)
    assert got == [0.25, 1.0]
    assert rec["rets"] == [0, 0]          # 0 = keep going


def test_align_progress_reaches_whisperx_callback(monkeypatch):
    rec = _install_fake_align(monkeypatch)
    cb = lambda p: None
    pipeline.align({"segments": ["s"]}, "fr", "cpu", "AUDIO", progress=cb)
    (_, kw), = rec["align"]
    assert kw["progress_callback"] is cb


def test_speaker_models_ready_follows_the_files_on_disk(tmp_path):
    assert not pipeline.speaker_models_ready(str(tmp_path))
    for key in pipeline.SPEAKER_MODELS:
        path = pipeline.speaker_model_path(str(tmp_path), key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(b"x")
    assert pipeline.speaker_models_ready(str(tmp_path))


def test_ensure_speaker_models_offline_and_missing_raises_friendly_text(tmp_path):
    with pytest.raises(RuntimeError, match="not downloaded yet"):
        pipeline.ensure_speaker_models(str(tmp_path), offline=True)


def test_ensure_speaker_models_downloads_missing_files_with_progress(monkeypatch, tmp_path):
    import io
    import urllib.request
    urls = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    def urlopen(url, timeout=0):
        urls.append(url)
        return Resp(b"x" * 100)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(pipeline, "SPEAKER_MODELS", {
        "embedding": ("emb.onnx", "https://example.invalid/emb.onnx", 100)})
    seen = []
    pipeline.ensure_speaker_models(str(tmp_path), offline=False, progress=seen.append)
    assert urls == ["https://example.invalid/emb.onnx"]
    assert os.path.getsize(tmp_path / "speakers" / "emb.onnx") == 100
    assert seen and seen[-1] == 1.0
    pipeline.ensure_speaker_models(str(tmp_path), offline=False)     # nothing missing: no second download
    assert len(urls) == 1


def test_assign_speakers_calls_assign_word_speakers_and_returns_its_result(monkeypatch):
    rec = _install_fake_diarize(monkeypatch)
    result = {"segments": ["s"]}
    out = pipeline.assign_speakers("DF", result)
    assert out == {"assigned": True}
    (df, res, emb), = rec["assign"]
    assert df == "DF"
    assert res is result
    assert emb is None


def test_assign_speakers_forwards_embeddings(monkeypatch):
    rec = _install_fake_diarize(monkeypatch)
    emb = {"SPEAKER_00": [0.1, 0.2]}
    pipeline.assign_speakers("DF", {"segments": []}, emb)
    assert rec["assign"][0][2] == emb


def _speaker_result():
    return {"language": "en", "segments": [
        {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00",
         "words": [{"word": "hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]},
        {"start": 1.0, "end": 2.0, "text": "there", "speaker": "SPEAKER_01",
         "words": [{"word": "there", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"}]},
        {"start": 2.0, "end": 3.0, "text": "plain",
         "words": [{"word": "plain", "start": 2.0, "end": 3.0}]},
    ]}


def _capture_writers(monkeypatch):
    """Fake get_writer recording the result each format's writer receives."""
    got = {}
    fake_utils = types.ModuleType("whisperx.utils")

    def get_writer(fmt, out_dir):
        def write(result, audio_path, opts):
            got[fmt] = result
        return write

    fake_utils.get_writer = get_writer
    fake_wx = types.ModuleType("whisperx")
    fake_wx.utils = fake_utils
    monkeypatch.setitem(sys.modules, "whisperx", fake_wx)
    monkeypatch.setitem(sys.modules, "whisperx.utils", fake_utils)
    return got


def _speakers_of(result):
    return ([s.get("speaker") for s in result["segments"]],
            [w.get("speaker") for s in result["segments"] for w in s.get("words", [])])


_SPEAKER_FORMATS = ["srt", "vtt", "txt", "json", "aud"]


@pytest.mark.parametrize("fmt", _SPEAKER_FORMATS)
def test_export_renames_speakers_for_writer(monkeypatch, tmp_path, fmt):
    got = _capture_writers(monkeypatch)
    pipeline.export(_speaker_result(), [fmt], str(tmp_path / "o"),
                    str(tmp_path / "a.wav"),
                    {"speaker_names": {"SPEAKER_00": "Amina"}})
    seg_sp, word_sp = _speakers_of(got[fmt])
    assert seg_sp == ["Amina", "SPEAKER_01", None]
    assert word_sp == ["Amina", "SPEAKER_01", None]


def test_export_renames_multiple_speakers(monkeypatch, tmp_path):
    got = _capture_writers(monkeypatch)
    pipeline.export(_speaker_result(), ["txt"], str(tmp_path / "o"),
                    str(tmp_path / "a.wav"),
                    {"speaker_names": {"SPEAKER_00": "Amina", "SPEAKER_01": "Omar"}})
    assert _speakers_of(got["txt"])[0] == ["Amina", "Omar", None]


def test_export_blank_speaker_names_are_ignored(monkeypatch, tmp_path):
    got = _capture_writers(monkeypatch)
    pipeline.export(_speaker_result(), ["srt"], str(tmp_path / "o"),
                    str(tmp_path / "a.wav"),
                    {"speaker_names": {"SPEAKER_00": "  ", "SPEAKER_01": ""}})
    assert _speakers_of(got["srt"])[0] == ["SPEAKER_00", "SPEAKER_01", None]


def test_export_does_not_mutate_callers_result(monkeypatch, tmp_path):
    _capture_writers(monkeypatch)
    result = _speaker_result()
    pipeline.export(result, ["srt", "txt", "word_json"], str(tmp_path / "o"),
                    str(tmp_path / "a.wav"),
                    {"speaker_names": {"SPEAKER_00": "Amina"},
                     "speaker_names_in_cues": False})
    assert result == _speaker_result()


def test_export_without_speaker_names_passes_result_content_unchanged(monkeypatch, tmp_path):
    got = _capture_writers(monkeypatch)
    pipeline.export(_speaker_result(), ["srt", "json"], str(tmp_path / "o"),
                    str(tmp_path / "a.wav"), {})
    assert got["srt"] == _speaker_result()
    assert got["json"] == _speaker_result()


def test_export_word_json_words_carry_renamed_speaker(monkeypatch, tmp_path):
    _capture_writers(monkeypatch)
    out_dir = str(tmp_path / "o")
    outcomes = pipeline.export(_speaker_result(), ["word_json"], out_dir,
                               str(tmp_path / "a.wav"),
                               {"speaker_names": {"SPEAKER_00": "Amina"}})
    path = os.path.join(out_dir, "a_words.json")
    assert outcomes == [("word_json", path, None)]
    import json
    with open(path, encoding="utf-8") as fh:
        words = json.load(fh)
    assert [w.get("speaker") for w in words] == ["Amina", "SPEAKER_01", None]


@pytest.mark.parametrize("fmt", ["srt", "vtt"])
def test_export_cues_without_names_drops_segment_speaker_for_cue_formats(
        monkeypatch, tmp_path, fmt):
    got = _capture_writers(monkeypatch)
    pipeline.export(_speaker_result(), [fmt], str(tmp_path / "o"),
                    str(tmp_path / "a.wav"),
                    {"speaker_names": {"SPEAKER_00": "Amina"},
                     "speaker_names_in_cues": False})
    assert all("speaker" not in s for s in got[fmt]["segments"])
    assert [s["text"] for s in got[fmt]["segments"]] == ["hello", "there", "plain"]


@pytest.mark.parametrize("fmt", ["txt", "json", "aud"])
def test_export_cues_flag_does_not_affect_other_formats(monkeypatch, tmp_path, fmt):
    got = _capture_writers(monkeypatch)
    pipeline.export(_speaker_result(), [fmt], str(tmp_path / "o"),
                    str(tmp_path / "a.wav"),
                    {"speaker_names": {"SPEAKER_00": "Amina"},
                     "speaker_names_in_cues": False})
    assert _speakers_of(got[fmt])[0] == ["Amina", "SPEAKER_01", None]


@pytest.mark.parametrize("fmt", ["srt", "vtt"])
def test_export_cues_default_keeps_speaker_in_cue_formats(monkeypatch, tmp_path, fmt):
    got = _capture_writers(monkeypatch)
    pipeline.export(_speaker_result(), [fmt], str(tmp_path / "o"),
                    str(tmp_path / "a.wav"),
                    {"speaker_names": {"SPEAKER_00": "Amina"}})
    assert _speakers_of(got[fmt])[0] == ["Amina", "SPEAKER_01", None]


def test_export_writes_speaker_embeddings_file(monkeypatch, tmp_path):
    _capture_writers(monkeypatch)
    out_dir = str(tmp_path / "o")
    emb = {"SPEAKER_00": [0.1, 0.2], "SPEAKER_01": [0.3, 0.4]}
    outcomes = pipeline.export(_speaker_result(), ["srt"], out_dir,
                               str(tmp_path / "talk.wav"), {"speaker_embeddings": emb})
    path = os.path.join(out_dir, "talk_speakers.json")
    assert ("speakers", path, None) in outcomes
    import json
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh) == emb


@pytest.mark.parametrize("emb", [None, {}])
def test_export_without_embeddings_writes_no_speakers_file(monkeypatch, tmp_path, emb):
    _capture_writers(monkeypatch)
    out_dir = str(tmp_path / "o")
    outcomes = pipeline.export(_speaker_result(), ["srt"], out_dir,
                               str(tmp_path / "talk.wav"), {"speaker_embeddings": emb})
    assert not any(o[0] == "speakers" for o in outcomes)
    assert not os.path.exists(os.path.join(out_dir, "talk_speakers.json"))


# ── H2: ensure_punkt / unload_model ──────────────────────────────────────────

def test_ensure_punkt_true_when_present_and_never_downloads(tmp_path, monkeypatch):
    from core import pipeline
    (tmp_path / "nltk_data" / "tokenizers" / "punkt_tab").mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "nltk", None)     # import would fail
    assert pipeline.ensure_punkt(str(tmp_path), False) is True
    assert pipeline.ensure_punkt(str(tmp_path), True) is True


def test_ensure_punkt_offline_missing_is_false(tmp_path):
    from core import pipeline
    assert pipeline.ensure_punkt(str(tmp_path), True) is False
    assert not (tmp_path / "nltk_data").exists()


def test_ensure_punkt_downloads_into_model_dir(tmp_path, monkeypatch):
    import types
    from core import pipeline
    calls = []

    def download(name, download_dir=None, quiet=False):
        calls.append((name, download_dir))
        os.makedirs(os.path.join(download_dir, "tokenizers", "punkt_tab"))
        return True

    monkeypatch.setitem(sys.modules, "nltk", types.SimpleNamespace(download=download))
    assert pipeline.ensure_punkt(str(tmp_path), False) is True
    assert calls == [("punkt_tab", str(tmp_path / "nltk_data"))]


def test_ensure_punkt_download_failure_is_false(tmp_path, monkeypatch):
    import types
    from core import pipeline

    def download(*a, **k):
        raise OSError("no network")

    monkeypatch.setitem(sys.modules, "nltk", types.SimpleNamespace(download=download))
    assert pipeline.ensure_punkt(str(tmp_path), False) is False


def test_unload_model_never_raises():
    from core import pipeline
    assert pipeline.unload_model(object()) is None
    assert pipeline.unload_model() is None
