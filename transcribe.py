"""
WhisperX Transcriber — headless CLI

Usage:
    python transcribe.py <audio_or_video_file> [options]

Options:
    --model      Model name (default: large-v2)
                 Choices: tiny, base, small, medium, large-v2, large-v3
    --language   Language code (default: auto-detect)
                 Examples: en, ar, fr, de, es, zh, ja, ur, fa
    --device     cuda | cpu (default: auto-detect)
    --output     Output format: word_json, srt, vtt, txt, tsv, json (default: word_json)
    --model-dir  Where to cache downloaded models (default: ./Models)

Examples:
    python transcribe.py audio.mp3
    python transcribe.py audio.mp3 --language ar --model large-v2
    python transcribe.py audio.mp3 --device cpu --output srt
    python transcribe.py audio.mp3 --model-dir D:\\models
"""
import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="WhisperX headless transcription CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio", help="Path to audio or video file")
    parser.add_argument("--model",     default="large-v2",
                        help="Whisper model (default: large-v2)")
    parser.add_argument("--language",  default=None,
                        help="Language code, e.g. en, ar, fr (default: auto-detect)")
    parser.add_argument("--device",    default="auto",
                        help="cuda | cpu | auto (default: auto)")
    parser.add_argument("--output",    default="word_json",
                        choices=["word_json", "srt", "vtt", "txt", "tsv", "json"],
                        help="Output format (default: word_json)")
    parser.add_argument("--model-dir", default=None,
                        help="Model cache directory (default: ./Models next to this script)")
    args = parser.parse_args()

    # Model cache directory
    if args.model_dir:
        model_dir = os.path.abspath(args.model_dir)
    else:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
    os.makedirs(model_dir, exist_ok=True)
    os.environ["HF_HOME"] = model_dir

    from core import pipeline

    # Device
    _, t_dev, a_dev = pipeline.detect_device(args.device)
    default_compute = "float16" if t_dev == "cuda" else "int8"
    compute, adjusted = pipeline.resolve_compute(t_dev, default_compute)
    print(f"Device: {t_dev.upper()}   Compute: {compute}")

    # Model
    print(f"Loading model '{args.model}'...")
    model = pipeline.load_model(args.model, t_dev, compute, args.language, beam=5)

    # Audio
    print("Loading audio...")
    audio = pipeline.load_audio(args.audio)

    # Transcribe
    print("Transcribing...")
    result, detected_lang = pipeline.transcribe(model, audio, 16, args.language)
    print(f"  Language: {detected_lang}   Segments: {len(result['segments'])}")

    # Align
    print("Aligning word timestamps...")
    try:
        result = pipeline.align(result, detected_lang, a_dev, audio)
    except Exception as e:
        print(f"  Warning: alignment failed ({e}) — continuing without word timestamps")
        result.setdefault("language", detected_lang)

    # Export
    out_dir = os.path.dirname(os.path.abspath(args.audio))
    results = pipeline.export(result, [args.output], out_dir, args.audio, {})
    for fmt, path, err in results:
        if err:
            print(f"  {fmt} error: {err}")
        elif fmt == "word_json":
            import json
            with open(path, encoding="utf-8") as fh:
                words = json.load(fh)
            print(f"Done!  {len(words)} words → {path}")
        else:
            print(f"Done!  → {path}")


if __name__ == "__main__":
    main()
