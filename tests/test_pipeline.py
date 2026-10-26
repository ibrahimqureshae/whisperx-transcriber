"""
Golden-file tests for core/pipeline.py.

Slow tests (requiring the tiny whisper model) are marked @pytest.mark.slow
and skipped in fast CI runs via: pytest -m "not slow"
"""
import os
import json
import re
import sys
import pytest
from pathlib import Path

# Make sure the project root is on the path when running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline import format_srt_timestamp

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_WAV = FIXTURES / "sample.wav"
REFERENCE = "the quick brown fox jumps over the lazy dog"


# ── SRT timestamp formatter ────────────────────────────────────────────────────

def test_format_srt_timestamp_zero():
    assert format_srt_timestamp(0.0) == "00:00:00,000"


def test_format_srt_timestamp_subsecond():
    assert format_srt_timestamp(1.5) == "00:00:01,500"
    assert format_srt_timestamp(4.2) == "00:00:04,200"


def test_format_srt_timestamp_minutes():
    assert format_srt_timestamp(90.0) == "00:01:30,000"
    assert format_srt_timestamp(61.75) == "00:01:01,750"


def test_format_srt_timestamp_hours():
    # 1h 1m 1.5s = 3661.5s
    assert format_srt_timestamp(3661.5) == "01:01:01,500"


def test_format_srt_timestamp_rounding():
    # Ensure rounding ms=999.5 → 1000 is handled (no "HH:MM:SS,1000")
    result = format_srt_timestamp(0.9995)
    assert "," in result
    ms_part = result.split(",")[1]
    assert len(ms_part) == 3, f"ms part should be 3 digits, got {ms_part!r}"


# ── Export format tests (synthetic result, no model needed) ──────────────────

def _make_result():
    """Minimal whisperx-style result dict for export tests."""
    return {
        "segments": [{
            "start": 1.5,
            "end": 4.2,
            "text": " the quick brown fox",
            "words": [
                {"word": "the",   "start": 1.5, "end": 1.8, "score": 0.9},
                {"word": "quick", "start": 1.9, "end": 2.2, "score": 0.9},
                {"word": "brown", "start": 2.3, "end": 2.6, "score": 0.9},
                {"word": "fox",   "start": 2.7, "end": 4.2, "score": 0.9},
            ],
        }],
        "language": "en",
    }


def test_export_srt(tmp_path):
    from core.pipeline import export
    result = _make_result()
    outcomes = export(result, ["srt"], str(tmp_path), str(SAMPLE_WAV), {})
    assert len(outcomes) == 1
    fmt, path, err = outcomes[0]
    assert fmt == "srt"
    assert err is None, f"SRT export failed: {err}"
    content = Path(path).read_text(encoding="utf-8")
    assert len(content) > 0
    assert "-->" in content, "SRT must contain timestamp separator"
    # SRT timestamps look like 00:00:01,500
    assert re.search(r"\d{2}:\d{2}:\d{2},\d{3}", content), "SRT timestamp format not found"


def test_export_vtt(tmp_path):
    from core.pipeline import export
    result = _make_result()
    outcomes = export(result, ["vtt"], str(tmp_path), str(SAMPLE_WAV), {})
    fmt, path, err = outcomes[0]
    assert err is None, f"VTT export failed: {err}"
    content = Path(path).read_text(encoding="utf-8")
    assert len(content) > 0
    assert "WEBVTT" in content, "VTT must start with WEBVTT header"
    assert "-->" in content


def test_export_txt(tmp_path):
    from core.pipeline import export
    result = _make_result()
    outcomes = export(result, ["txt"], str(tmp_path), str(SAMPLE_WAV), {})
    fmt, path, err = outcomes[0]
    assert err is None, f"TXT export failed: {err}"
    content = Path(path).read_text(encoding="utf-8")
    assert len(content) > 0
    assert "fox" in content.lower()


def test_export_tsv(tmp_path):
    from core.pipeline import export
    result = _make_result()
    outcomes = export(result, ["tsv"], str(tmp_path), str(SAMPLE_WAV), {})
    fmt, path, err = outcomes[0]
    assert err is None, f"TSV export failed: {err}"
    content = Path(path).read_text(encoding="utf-8")
    assert len(content) > 0
    # TSV has tab-separated columns
    assert "\t" in content


def test_export_word_json(tmp_path):
    from core.pipeline import export
    result = _make_result()
    outcomes = export(result, ["word_json"], str(tmp_path), str(SAMPLE_WAV), {})
    fmt, path, err = outcomes[0]
    assert err is None, f"word_json export failed: {err}"
    words = json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(words, list)
    assert len(words) == 4  # the, quick, brown, fox
    assert words[0]["word"] == "the"


def test_export_multiple_formats(tmp_path):
    from core.pipeline import export
    result = _make_result()
    formats = ["srt", "vtt", "txt", "word_json"]
    outcomes = export(result, formats, str(tmp_path), str(SAMPLE_WAV), {})
    assert len(outcomes) == 4
    for fmt, path, err in outcomes:
        assert err is None, f"Export {fmt} failed: {err}"
        assert path is not None
        assert os.path.getsize(path) > 0, f"{fmt} file is empty"


# ── WER test — requires tiny model download (~75 MB, CPU inference) ───────────

def _normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for WER comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


@pytest.mark.slow
def test_transcribe_wer():
    """
    Transcribe sample.wav with the tiny English model and check WER < 0.15.
    Text is normalized (lowercase, punctuation stripped) before comparison.
    Skipped in fast CI; run with: pytest -m slow
    """
    from jiwer import wer
    from core.pipeline import detect_device, resolve_compute, load_model, load_audio, transcribe

    assert SAMPLE_WAV.exists(), f"Fixture missing: {SAMPLE_WAV}"

    _, t_dev, _ = detect_device("cpu")
    compute, _ = resolve_compute(t_dev, "int8")
    model = load_model("tiny", t_dev, compute, "en", 5)
    audio = load_audio(str(SAMPLE_WAV))
    result, _ = transcribe(model, audio, 16, "en")

    hypothesis = " ".join(seg["text"].strip() for seg in result["segments"]).strip()
    error_rate = wer(_normalize_text(REFERENCE), _normalize_text(hypothesis))
    assert error_rate < 0.15, (
        f"WER too high: {error_rate:.3f}\n"
        f"  Reference:  {REFERENCE!r}\n"
        f"  Hypothesis: {hypothesis!r}"
    )
