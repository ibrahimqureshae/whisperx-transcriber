"""
Stand-in for core/runner.py that replays a plausible event stream, so the GUI
can be driven and screenshotted without torch or models.

    set WHISPERX_RUNNER=tests\\fake_runner.py   (ui.state.runner_command uses it)

Env: FAKE_RUNNER_SECONDS  total run time (default 12)
     FAKE_RUNNER_FAIL=1   fail during transcription
     FAKE_RUNNER_HOLD=1   stop at 40% and wait forever (for screenshots / Cancel)
"""
import json
import math
import os
import sys
import time


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    cfg = json.loads(sys.stdin.readline() or "{}")
    total = float(os.environ.get("FAKE_RUNNER_SECONDS", "12"))
    tick = total / 40
    name = os.path.splitext(os.path.basename(cfg.get("audio", "audio.wav")))[0]
    out_dir = cfg.get("out_dir") or "."

    def stage(n, state, detail="", **extra):
        emit({"t": "stage", "name": n, "state": state, "detail": detail, **extra})

    # overall progress spans like core/runner.py: load 4 %, transcribe, align, diarize, export 2 %
    mid = ["transcribe", "align"] + (["diarize"] if cfg.get("diarize") else [])   # the fake always aligns
    spans, lo = {"load": (0.0, 0.04)}, 0.04
    for k in mid:
        spans[k] = (lo, lo + 0.94 / len(mid)); lo = spans[k][1]
    spans["export"] = (lo, 1.0)

    def progress(n, frac):
        a, b = spans[n]
        emit({"t": "progress", "v": round(a + (b - a) * frac, 4)})

    emit({"t": "log", "m": "Device: CUDA   Compute: float16"})
    bins = [round(abs(math.sin(i / 7.0)) * (0.35 + 0.65 * abs(math.sin(i / 23.0))), 3)
            for i in range(200)]
    emit({"t": "wave", "bins": bins, "seconds": 2292.0})
    stage("load", "start", cfg.get("model", "large-v2"))
    progress("load", 0)
    emit({"t": "status", "m": "Loading model..."})
    emit({"t": "pmode", "mode": "indeterminate"})
    time.sleep(tick * 3)
    stage("load", "done", f"{cfg.get('model', 'large-v2')} · 2.1 s")
    progress("load", 1)

    stage("detect", "start")
    time.sleep(tick)
    emit({"t": "language", "code": "ur", "prob": 0.87})
    stage("detect", "done", "ur · 0.87")

    stage("vad", "start")
    time.sleep(tick)
    stage("vad", "done", "2292 s audio")

    stage("transcribe", "start", "14 chunks")
    emit({"t": "pmode", "mode": "determinate"})
    for i in range(1, 21):
        pct = i * 5
        if os.environ.get("FAKE_RUNNER_FAIL") and pct >= 50:
            emit({"t": "error", "m": "CUDA out of memory. Tried to allocate 2.00 GiB\nTraceback ..."})
            sys.exit(1)
        if os.environ.get("FAKE_RUNNER_HOLD") and pct >= 40:
            while True:
                time.sleep(1)
        progress("transcribe", pct / 100)
        emit({"t": "status", "m": f"Transcribing… {pct}%"})
        time.sleep(tick)
    stage("transcribe", "done", "14 chunks · 402 s speech")

    stage("align", "start", "jonatasgrosman/wav2vec2-large-xlsr-53-urdu")
    for i in range(1, 4):
        time.sleep(tick)
        progress("align", i / 3)
    stage("align", "done", "jonatasgrosman/wav2vec2-large-xlsr-53-urdu")

    segments = [
        {"start": 0.0, "end": 3.2, "text": " السلام علیکم، آج ہم بات کریں گے", "speaker": "SPEAKER_00",
         "words": [{"word": "السلام", "start": 0.0, "end": 0.6, "score": 0.91, "speaker": "SPEAKER_00"},
                   {"word": "علیکم،", "start": 0.6, "end": 1.1, "score": 0.42, "speaker": "SPEAKER_00"}]},
        {"start": 3.4, "end": 6.0, "text": " Thank you for having me.", "speaker": "SPEAKER_01",
         "words": [{"word": "Thank", "start": 3.4, "end": 3.7, "score": 0.95, "speaker": "SPEAKER_01"},
                   {"word": "you", "start": 3.7, "end": 3.9, "score": 0.97, "speaker": "SPEAKER_01"}]},
    ]
    if cfg.get("diarize"):
        stage("diarize", "start")
        for i in range(1, 4):
            time.sleep(tick)
            progress("diarize", i / 3)
        stage("diarize", "done", "2 speakers")

    stage("export", "start")
    files = [os.path.join(out_dir, f"{name}.{f if f != 'word_json' else 'words.json'}")
             for f in (cfg.get("formats") or ["srt"])]
    time.sleep(tick)
    stage("export", "done", ", ".join(os.path.basename(f) for f in files), files=files)
    progress("export", 1)
    emit({"t": "transcript", "segments": segments})
    emit({"t": "log", "m": "Done!"})
    emit({"t": "done", "path": files[-1] if files else None})


if __name__ == "__main__":
    main()
