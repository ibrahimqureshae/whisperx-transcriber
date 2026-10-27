"""
CLI tests: transcribe.main() with core.pipeline faked and sys.argv patched.
Asserts what pipeline.load_model receives.  Nothing touches real models.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import transcribe
from core import pipeline


@pytest.fixture
def cli(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(pipeline, "detect_device", lambda req: ("cpu", "cpu", "cpu"))
    monkeypatch.setattr(pipeline, "resolve_compute", lambda t, c: (c, False))

    def load_model(*a, **k):
        seen["args"], seen["kw"] = a, k
        return object()

    monkeypatch.setattr(pipeline, "load_model", load_model)
    monkeypatch.setattr(pipeline, "load_audio", lambda p: "AUDIO")
    monkeypatch.setattr(pipeline, "transcribe",
                        lambda *a, **k: ({"segments": [], "language": "en"}, "en"))
    monkeypatch.setattr(pipeline, "align_supported", lambda l: False)

    def align(*a, **k):
        seen["align"] = (a, k)
        return a[0]

    def export(*a, **k):
        seen["export"] = (a, k)
        return []

    monkeypatch.setattr(pipeline, "align", align)
    monkeypatch.setattr(pipeline, "export", export)

    def run(*flags):
        argv = ["transcribe.py", str(tmp_path / "a.wav"),
                "--model-dir", str(tmp_path), "--output-dir", str(tmp_path)]
        monkeypatch.setattr(sys, "argv", argv + list(flags))
        transcribe.main()
        return seen

    return run


def _decode(seen):
    return seen["kw"].get("decode") or {}


def test_no_new_flags_gives_defaults(cli):
    seen = cli()
    kw = seen["kw"]
    assert kw.get("hotwords") in (None, "")
    assert kw.get("suppress_numerals") is False
    assert kw.get("vad_method") == "pyannote"
    assert kw.get("device_index") == 0
    assert kw.get("threads") == 4
    assert kw.get("local_files_only") is False
    assert _decode(seen) == {}


def test_hotwords_flag(cli):
    assert cli("--hotwords", "Kubernetes, WhisperX")["kw"]["hotwords"] == "Kubernetes, WhisperX"


def test_suppress_numerals_flag(cli):
    assert cli("--suppress-numerals")["kw"]["suppress_numerals"] is True


@pytest.mark.parametrize("method", ["silero", "pyannote"])
def test_vad_method_flag(cli, method):
    assert cli("--vad-method", method)["kw"]["vad_method"] == method


def test_vad_method_rejects_unknown_choice(cli):
    with pytest.raises(SystemExit):
        cli("--vad-method", "bogus")


def test_device_index_flag_is_int(cli):
    assert cli("--device-index", "1")["kw"]["device_index"] == 1


def test_threads_flag_is_int(cli):
    assert cli("--threads", "8")["kw"]["threads"] == 8


def test_offline_flag_sets_local_files_only(cli):
    assert cli("--offline")["kw"]["local_files_only"] is True


def test_single_temperature_becomes_list_of_one_float(cli):
    assert _decode(cli("--temperature", "0.2"))["temperatures"] == [0.2]


def test_temperature_comma_list_becomes_float_list(cli):
    assert _decode(cli("--temperature", "0,0.2,0.4"))["temperatures"] == [0.0, 0.2, 0.4]


def test_beam_size_flag_sets_beam(cli):
    seen = cli("--beam-size", "2")
    beam = seen["kw"].get("beam", seen["args"][4] if len(seen["args"]) > 4 else None)
    assert beam == 2


def test_threshold_flags_map_to_decode_keys(cli):
    decode = _decode(cli("--no-speech-threshold", "0.4", "--logprob-threshold", "-0.7",
                         "--compression-ratio-threshold", "2.0"))
    assert decode["no_speech_threshold"] == 0.4
    assert decode["log_prob_threshold"] == -0.7
    assert decode["compression_ratio_threshold"] == 2.0


@pytest.mark.parametrize("text,value", [("true", True), ("false", False)])
def test_condition_on_previous_text_flag_is_bool(cli, text, value):
    got = _decode(cli("--condition-on-previous-text", text))["condition_on_previous_text"]
    assert got is value


def test_decode_contains_only_flags_given(cli):
    decode = _decode(cli("--no-speech-threshold", "0.4"))
    assert set(decode) == {"no_speech_threshold"}


# ── v1.4 T3: alignment overrides ─────────────────────────────────────────────

@pytest.fixture
def cli_align(cli, monkeypatch):
    """Like cli, but the language is alignable so pipeline.align is reached."""
    monkeypatch.setattr(pipeline, "align_supported", lambda l: True)
    return cli


def test_align_flags_default(cli_align):
    seen = cli_align()
    kw = seen["align"][1]
    assert kw.get("align_model") in (None, "")
    assert kw.get("interpolate_method", "nearest") == "nearest"
    assert kw.get("return_char_alignments", False) is False
    assert seen["export"][0][-1]["segment_resolution"] == "sentence"


def test_align_model_flag_reaches_align(cli_align):
    seen = cli_align("--align-model", "org/my-wav2vec2")
    assert seen["align"][1]["align_model"] == "org/my-wav2vec2"


@pytest.mark.parametrize("method", ["nearest", "linear", "ignore"])
def test_interpolate_method_flag_reaches_align(cli_align, method):
    assert cli_align("--interpolate-method", method)["align"][1]["interpolate_method"] == method


def test_interpolate_method_rejects_unknown_choice(cli_align):
    with pytest.raises(SystemExit):
        cli_align("--interpolate-method", "cubic")


def test_char_alignments_flag_reaches_align(cli_align):
    assert cli_align("--char-alignments")["align"][1]["return_char_alignments"] is True


@pytest.mark.parametrize("res", ["sentence", "chunk"])
def test_segment_resolution_flag_reaches_export_opts(cli_align, res):
    seen = cli_align("--segment-resolution", res)
    assert seen["export"][0][-1]["segment_resolution"] == res


def test_segment_resolution_rejects_unknown_choice(cli_align):
    with pytest.raises(SystemExit):
        cli_align("--segment-resolution", "word")


# ── v1.4 T4: speaker diarization ─────────────────────────────────────────────


@pytest.fixture
def cli_diar(cli_align, monkeypatch):
    """cli_align plus fakes for pipeline.diarize / ensure_speaker_models / assign_speakers."""
    calls = {"diarize": [], "assign": [], "ensure": []}

    def diarize(audio, model_dir, **k):
        calls["diarize"].append(((audio, model_dir), k))
        if k.get("return_embeddings"):
            return "DF", {"SPEAKER_00": [0.1]}
        return "DF"

    def assign_speakers(df, result, speaker_embeddings=None):
        calls["assign"].append((df, speaker_embeddings))
        return result

    monkeypatch.setattr(pipeline, "ensure_speaker_models",
                        lambda model_dir, offline, progress=None: calls["ensure"].append((model_dir, offline)))
    monkeypatch.setattr(pipeline, "diarize", diarize)
    monkeypatch.setattr(pipeline, "assign_speakers", assign_speakers)

    def run(*flags):
        seen = cli_align(*flags)
        seen["calls"] = calls
        return seen

    return run


def test_no_diarize_flag_makes_no_diarize_call(cli_diar):
    seen = cli_diar()
    assert seen["calls"]["diarize"] == []
    assert seen["calls"]["assign"] == []


def test_diarize_flag_downloads_models_and_calls_diarize_without_a_token(cli_diar):
    seen = cli_diar("--diarize")
    (args, kw), = seen["calls"]["diarize"]
    assert args[0] == "AUDIO"
    assert len(seen["calls"]["ensure"]) == 1 and seen["calls"]["ensure"][0][1] is False
    assert "hf_token" not in kw
    assert kw["min_speakers"] is None
    assert kw["max_speakers"] is None
    assert kw["return_embeddings"] is False
    assert seen["calls"]["assign"] == [("DF", None)]


def test_diarize_min_and_max_speakers_are_ints(cli_diar):
    seen = cli_diar("--diarize",
                    "--min-speakers", "2", "--max-speakers", "4")
    (_, kw), = seen["calls"]["diarize"]
    assert kw["min_speakers"] == 2
    assert kw["max_speakers"] == 4


@pytest.mark.parametrize("flag", ["--min-speakers", "--max-speakers"])
def test_speaker_count_flags_reject_non_integers(cli_diar, flag):
    with pytest.raises(SystemExit):
        cli_diar("--diarize", flag, "two")


def test_hf_token_flag_is_gone(cli_diar):
    with pytest.raises(SystemExit):
        cli_diar("--diarize", "--hf-token", "x")


def test_diarize_failure_warns_and_continues(cli_diar, capsys, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no network")
    monkeypatch.setattr(pipeline, "ensure_speaker_models", boom)
    seen = cli_diar("--diarize")
    assert "no network" in capsys.readouterr().out
    assert seen["calls"]["diarize"] == []
    assert "export" in seen


def test_speaker_embeddings_flag_requests_and_exports_embeddings(cli_diar):
    seen = cli_diar("--diarize", "--speaker-embeddings")
    (_, kw), = seen["calls"]["diarize"]
    assert kw["return_embeddings"] is True
    assert seen["calls"]["assign"] == [("DF", {"SPEAKER_00": [0.1]})]
    assert seen["export"][0][-1]["speaker_embeddings"] == {"SPEAKER_00": [0.1]}


def test_without_speaker_embeddings_flag_export_has_no_embeddings(cli_diar):
    seen = cli_diar("--diarize")
    assert not seen["export"][0][-1].get("speaker_embeddings")


def test_translate_with_unsupported_model_exits_2(cli, capsys):
    with pytest.raises(SystemExit) as ex:
        cli("--task", "translate", "--model", "large-v3-turbo")
    assert ex.value.code == 2
    assert "'large-v3-turbo' can only transcribe" in capsys.readouterr().err


def test_translate_with_large_v3_is_accepted(cli):
    assert cli("--task", "translate", "--model", "large-v3")["kw"]["task"] == "translate"
