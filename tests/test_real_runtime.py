r"""
Real-runtime smoke tests: the actual runner process, real whisperx, real models.
The faked suite hid two release blockers (a stdlib name clash and a native crash),
so core changes must also pass these.

    G:\WhisperX\qa-runtime\Scripts\python.exe -m pytest tests -m real -q

Env: WHISPERX_QA_MODELS (model cache; default G:\WhisperX\qa-models),
     WHISPERX_QA_MEDIA (folder with English_test.wav / Urdu_test.m4a; default tests/fixtures).
Skipped when whisperx/torch are not importable (the UI-only venv).
"""
import copy
import os
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.real
whisperx = pytest.importorskip("whisperx")

from core.settings import DEFAULTS
from ui import state as st

MODELS = os.environ.get("WHISPERX_QA_MODELS", r"G:\WhisperX\qa-models")
MEDIA = Path(os.environ.get("WHISPERX_QA_MEDIA", str(ROOT / "tests" / "fixtures")))


def _run(tmp_path, audio, **over):
    os.environ.pop("WHISPERX_RUNNER", None)            # the real runner, launched like the app does
    s = copy.deepcopy(DEFAULTS)
    s.update(model_dir=MODELS, out_dir=str(tmp_path), **over)
    cfg = st.build_cfg(s, str(audio))
    events, done = [], threading.Event()
    rc = []
    proc = st.RunnerProcess(cfg, events.append, lambda code: (rc.append(code), done.set()))
    proc.start()
    assert done.wait(900), "runner did not finish within 15 min"
    return rc[0], events


def _err(events):
    return [e.get("m") for e in events if e.get("t") in ("error", "warn")]


def test_english_small_en_end_to_end(tmp_path):
    rc, events = _run(tmp_path, MEDIA / "English_test.wav", model="small.en",
                      language="Auto-detect", formats=["srt", "word_json"])
    assert rc == 0, _err(events)
    srt = (tmp_path / "English_test.srt").read_text(encoding="utf-8")
    assert "quick brown fox" in srt.lower()


def test_urdu_hf_aligner_gives_word_timings(tmp_path):
    rc, events = _run(tmp_path, MEDIA / "Urdu_test.m4a", model="large-v3-turbo",
                      language="Auto-detect", formats=["srt", "word_json"], batch="8")
    assert rc == 0, _err(events)
    lang = [e for e in events if e.get("t") == "language"]
    assert lang and lang[0]["code"] == "ur"
    words = (tmp_path / "Urdu_test_words.json").read_text(encoding="utf-8")
    assert words.strip() not in ("", "[]")


def test_translate_with_turbo_is_refused(tmp_path):
    rc, events = _run(tmp_path, MEDIA / "Urdu_test.m4a", model="large-v3-turbo",
                      task="translate", formats=["srt"])
    assert rc == 1
    errs = [e["m"] for e in events if e.get("t") == "error"]
    assert errs == ["Translate needs a multilingual model that supports translation "
                    "(e.g. large-v3, medium). 'large-v3-turbo' can only transcribe."]


def test_long_file_default_batch_finishes_on_small_gpu(tmp_path):
    if not (MEDIA / "long_10min.wav").is_file():
        pytest.skip(f"long_10min.wav not in {MEDIA}; point WHISPERX_QA_MEDIA at a folder with it")
    rc, events = _run(tmp_path, MEDIA / "long_10min.wav", model="large-v3-turbo",
                      language="Auto-detect", formats=["srt"])
    assert rc == 0, _err(events)
    notes = [e.get("m", "") for e in events if e.get("t") in ("warn", "log")]
    assert any("retrying with batch" in m or "starting at batch" in m for m in notes)


def test_renamed_speaker_reaches_srt_txt_and_json(tmp_path):
    from core import pipeline
    result = {"language": "en", "segments": [
        {"start": 0.0, "end": 2.0, "text": " Hello there.", "speaker": "SPEAKER_00",
         "words": [{"word": "Hello", "start": 0.0, "end": 0.8, "score": 0.9, "speaker": "SPEAKER_00"},
                   {"word": "there.", "start": 0.9, "end": 2.0, "score": 0.9, "speaker": "SPEAKER_00"}]},
        {"start": 2.5, "end": 4.0, "text": " Hi.", "speaker": "SPEAKER_01",
         "words": [{"word": "Hi.", "start": 2.5, "end": 4.0, "score": 0.9, "speaker": "SPEAKER_01"}]}]}
    out = pipeline.export(result, ["srt", "txt", "json"], str(tmp_path), str(tmp_path / "talk.wav"),
                          {"speaker_names": {"SPEAKER_00": "Amina"}})
    assert [e for _, _, e in out] == [None, None, None], out
    for ext in ("srt", "txt", "json"):
        text = (tmp_path / f"talk.{ext}").read_text(encoding="utf-8")
        assert "Amina" in text and "SPEAKER_00" not in text, ext
        assert "SPEAKER_01" in text, ext
