"""Tests for core/settings.py (persisted GUI settings). No GUI, no network."""
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import settings
from core.settings import DEFAULTS, load, save, settings_path


def test_load_missing_file_returns_copy_of_defaults(tmp_path):
    snapshot = copy.deepcopy(DEFAULTS)
    result = load(str(tmp_path / "settings.json"))
    assert result == DEFAULTS
    assert result is not DEFAULTS
    result["model"] = "tiny"
    result["formats"].append("vtt")
    assert DEFAULTS == snapshot


def test_load_invalid_json_returns_defaults(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert load(str(p)) == DEFAULTS


def test_load_drops_unknown_keys(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"model": "small", "bogus": 1}), encoding="utf-8")
    result = load(str(p))
    assert result["model"] == "small"
    assert "bogus" not in result


def test_load_replaces_wrong_typed_values_with_default(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(
        json.dumps({"formats": "srt", "align": "yes", "batch": 16}),
        encoding="utf-8",
    )
    result = load(str(p))
    assert result["formats"] == ["srt"]
    assert result["align"] is True
    assert result["batch"] == "16"


def test_load_rejects_bool_for_str_key(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"model": True}), encoding="utf-8")
    assert load(str(p))["model"] == DEFAULTS["model"]


def test_load_rejects_int_for_bool_key(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"highlight": 1, "align": 0}), encoding="utf-8")
    result = load(str(p))
    assert result["highlight"] is DEFAULTS["highlight"]
    assert result["align"] is DEFAULTS["align"]


def _non_default_values():
    d = {}
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            d[k] = not v
        elif isinstance(v, list):
            d[k] = ["vtt", "txt"]
        elif isinstance(v, dict):
            d[k] = {sub: "changed-" + sub for sub in v}
        else:
            d[k] = "changed-" + k
    return d


def test_save_then_load_round_trips_every_key(tmp_path):
    path = str(tmp_path / "settings.json")
    d = _non_default_values()
    assert all(d[k] != DEFAULTS[k] for k in DEFAULTS)
    save(path, d)
    assert load(path) == d


def test_save_writes_only_defaults_keys_as_utf8_json(tmp_path):
    path = tmp_path / "settings.json"
    save(str(path), {**DEFAULTS, "extra": 1, "initial_prompt": "naïve café"})
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert "extra" not in on_disk
    assert set(on_disk) == set(DEFAULTS)
    assert on_disk["initial_prompt"] == "naïve café"
    assert load(str(path))["initial_prompt"] == "naïve café"


def test_save_to_unwritable_path_does_not_raise(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x", encoding="utf-8")
    save(str(f / "settings.json"), dict(DEFAULTS))


def test_settings_path_joins_settings_json():
    assert settings_path("C:/x") == os.path.join("C:/x", "settings.json")


# ── v1.4 schema ─────────────────────────────────────────

V13_KEYS = {
    "file", "out_dir", "language", "model", "device", "compute", "batch", "beam",
    "chunk", "formats", "align", "open_folder", "highlight", "vad_onset",
    "vad_offset", "max_width", "max_count", "model_dir", "theme", "task",
    "initial_prompt",
}

V14_NEW_DEFAULTS = {
    "hotwords": "",
    "suppress_numerals": False,
    "vad_method": "pyannote",
    "device_index": "0",
    "threads": "4",
    "offline_only": False,
    "decode": {
        "temperatures": "", "best_of": "", "patience": "", "length_penalty": "",
        "repetition_penalty": "", "no_repeat_ngram_size": "",
        "compression_ratio_threshold": "", "log_prob_threshold": "",
        "no_speech_threshold": "", "condition_on_previous_text": "",
        "prompt_reset_on_temperature": "", "hallucination_silence_threshold": "",
        "max_new_tokens": "",
    },
    "align_model": "",
    "interpolate_method": "nearest",
    "char_alignments": False,
    "diarize": False,
    "min_speakers": "",
    "max_speakers": "",
    "speaker_embeddings": False,
    "segment_resolution": "sentence",
    "speaker_names_in_cues": True,
    "queue_autostart": True,
    "queue_on_failure": "continue",
    "open_folder_per_batch": True,
    "last_dir": "",
    "ui_scale": "auto",          # H20: settings-file key, no control
}

V13_SETTINGS = {
    "file": "C:/media/talk.mp4",
    "out_dir": "C:/out",
    "language": "English",
    "model": "small",
    "device": "cpu",
    "compute": "int8",
    "batch": "8",
    "beam": "3",
    "chunk": "20",
    "formats": ["vtt", "txt"],
    "align": False,
    "open_folder": False,
    "highlight": True,
    "vad_onset": "0.400",
    "vad_offset": "0.300",
    "max_width": "42",
    "max_count": "2",
    "model_dir": "D:/models",
    "theme": "light",
    "task": "translate",
    "initial_prompt": "Meeting notes",
}


def _write_json(tmp_path, obj):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def test_defaults_have_exact_v14_schema():
    assert set(DEFAULTS) == V13_KEYS | set(V14_NEW_DEFAULTS)
    for k, v in V14_NEW_DEFAULTS.items():
        assert DEFAULTS[k] == v, k


def test_v13_settings_file_loads_with_values_kept_and_new_keys_at_default(tmp_path):
    result = load(_write_json(tmp_path, V13_SETTINGS))
    for k, v in V13_SETTINGS.items():
        assert result[k] == v, k
    for k, v in V14_NEW_DEFAULTS.items():
        assert result[k] == v, k


def test_load_decode_partial_fills_missing_subkeys_with_defaults(tmp_path):
    result = load(_write_json(tmp_path, {"decode": {"best_of": "3"}}))
    assert result["decode"] == {**V14_NEW_DEFAULTS["decode"], "best_of": "3"}


def test_load_decode_drops_unknown_subkeys(tmp_path):
    result = load(_write_json(tmp_path, {"decode": {"patience": "1.2", "bogus": "x"}}))
    assert "bogus" not in result["decode"]
    assert result["decode"]["patience"] == "1.2"


def test_load_decode_wrong_typed_subvalue_falls_back_to_default(tmp_path):
    result = load(_write_json(
        tmp_path, {"decode": {"best_of": 5, "patience": True, "max_new_tokens": "64"}}
    ))
    assert result["decode"]["best_of"] == ""
    assert result["decode"]["patience"] == ""
    assert result["decode"]["max_new_tokens"] == "64"


def test_load_non_dict_decode_falls_back_entirely(tmp_path):
    for bad in ["fast", ["best_of"], 3]:
        result = load(_write_json(tmp_path, {"decode": bad}))
        assert result["decode"] == V14_NEW_DEFAULTS["decode"], bad


def test_save_then_load_keeps_decode_subkeys(tmp_path):
    path = str(tmp_path / "settings.json")
    decode = {k: "" for k in V14_NEW_DEFAULTS["decode"]}
    decode.update(best_of="5", temperatures="0.0,0.2", condition_on_previous_text="false")
    save(path, {**copy.deepcopy(DEFAULTS), "decode": decode})
    assert load(path)["decode"] == decode


def test_old_token_and_diarize_model_are_gone_after_the_next_save(tmp_path):
    import json
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"hf_token": "hf_OLD", "diarize_model": "x/y", "beam": "3"}), encoding="utf-8")
    loaded = load(str(p))
    assert "hf_token" not in loaded and "diarize_model" not in loaded
    save(str(p), loaded)
    text = p.read_text(encoding="utf-8")
    assert "hf_OLD" not in text and "diarize_model" not in text and "hf_token" not in text
    assert json.loads(text)["beam"] == "3"
