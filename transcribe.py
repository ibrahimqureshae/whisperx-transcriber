"""
WhisperX Transcriber — headless CLI

Usage:
    python transcribe.py <audio_or_video_file> [options]

Options:
    --model      Model name (default: large-v2)
                 Choices: tiny, base, small, medium, large-v2, large-v3,
                 large-v3-turbo, distil-large-v3
    --language   Language code (default: auto-detect)
                 Examples: en, ar, fr, de, es, zh, ja, ur, fa
    --device     cuda | cpu (default: auto-detect)
    --task       transcribe | translate (to English) (default: transcribe)
    --output     Output format: word_json, srt, vtt, txt, tsv, json, aud (default: word_json)
                 May be repeated, e.g. --output srt --output txt
    --output-dir Where to write outputs (default: the audio file's folder)
    --initial-prompt  Text to bias the transcription (names, jargon, style)
    --model-dir  Where to cache downloaded models (default: ./Models)
    --hotwords   Words to favour while decoding (names, jargon)
    --suppress-numerals  Spell numbers out instead of using digits
    --vad-method silero | pyannote (default: pyannote)
    --device-index  GPU index (default: 0)
    --threads    CPU threads (default: 4)
    --offline    Never download; use only cached models
    --temperature  Sampling temperature, or a comma list (e.g. 0,0.2,0.4)
    --beam-size  Beam size (default: 5)
    --no-speech-threshold, --logprob-threshold, --compression-ratio-threshold
                 Decoding fallback thresholds
    --condition-on-previous-text  true | false
    --align-model  wav2vec2 model id for alignment (also enables
                 alignment for languages without a built-in model)
    --interpolate-method  nearest | linear | ignore (default: nearest)
    --char-alignments  Include per-character timings in the alignment result
    --segment-resolution  sentence | chunk (default: sentence)
    --diarize    Identify speakers (downloads ~34 MB of models once)
    --min-speakers, --max-speakers  Bounds on the number of speakers (integers)
    --speaker-embeddings  Also write <name>_speakers.json with speaker embeddings

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
    parser.add_argument("--task",      default="transcribe",
                        choices=["transcribe", "translate"],
                        help="transcribe or translate to English (default: transcribe)")
    parser.add_argument("--output",    action="append",
                        choices=["word_json", "srt", "vtt", "txt", "tsv", "json", "aud"],
                        help="Output format; repeatable (default: word_json)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: the audio file's folder)")
    parser.add_argument("--initial-prompt", default="",
                        help="Initial prompt to bias the transcription")
    parser.add_argument("--model-dir", default=None,
                        help="Model cache directory (default: ./Models next to this script)")
    parser.add_argument("--hotwords", default=None,
                        help="Words to favour while decoding")
    parser.add_argument("--suppress-numerals", action="store_true",
                        help="Spell numbers out instead of using digits")
    parser.add_argument("--vad-method", default="pyannote",
                        choices=["silero", "pyannote"],
                        help="Voice activity detection method (default: pyannote)")
    parser.add_argument("--device-index", type=int, default=0,
                        help="GPU index (default: 0)")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU threads (default: 4)")
    parser.add_argument("--offline", action="store_true",
                        help="Never download models; use only cached ones")
    parser.add_argument("--temperature", default=None,
                        help="Sampling temperature, or comma list e.g. 0,0.2,0.4")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam size (default: 5)")
    parser.add_argument("--no-speech-threshold", type=float, default=None)
    parser.add_argument("--logprob-threshold", type=float, default=None)
    parser.add_argument("--compression-ratio-threshold", type=float, default=None)
    parser.add_argument("--condition-on-previous-text", choices=["true", "false"],
                        default=None)
    parser.add_argument("--align-model", default=None,
                        help="wav2vec2 model id for alignment")
    parser.add_argument("--interpolate-method", default="nearest",
                        choices=["nearest", "linear", "ignore"],
                        help="How to time words without alignment (default: nearest)")
    parser.add_argument("--char-alignments", action="store_true",
                        help="Include per-character timings")
    parser.add_argument("--segment-resolution", default="sentence",
                        choices=["sentence", "chunk"],
                        help="Segment resolution (default: sentence)")
    parser.add_argument("--diarize", action="store_true",
                        help="Identify speakers (downloads ~34 MB of models once)")
    parser.add_argument("--min-speakers", type=int, default=None,
                        help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, default=None,
                        help="Maximum number of speakers")
    parser.add_argument("--speaker-embeddings", action="store_true",
                        help="Also write <name>_speakers.json with speaker embeddings")
    args = parser.parse_args()

    # Decoding options: only the flags that were given
    decode = {}
    if args.temperature is not None:
        decode["temperatures"] = [float(t) for t in args.temperature.split(",")]
    if args.no_speech_threshold is not None:
        decode["no_speech_threshold"] = args.no_speech_threshold
    if args.logprob_threshold is not None:
        decode["log_prob_threshold"] = args.logprob_threshold
    if args.compression_ratio_threshold is not None:
        decode["compression_ratio_threshold"] = args.compression_ratio_threshold
    if args.condition_on_previous_text is not None:
        decode["condition_on_previous_text"] = args.condition_on_previous_text == "true"

    from core import pipeline
    if args.task == "translate" and not pipeline.supports_translate(args.model):
        parser.error("Translate needs a multilingual model that supports translation "
                     f"(e.g. large-v3, medium). '{args.model}' can only transcribe.")

    # Model cache directory
    if args.model_dir:
        model_dir = os.path.abspath(args.model_dir)
    else:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
    os.makedirs(model_dir, exist_ok=True)
    os.environ["HF_HOME"] = model_dir

    # Device
    _, t_dev, a_dev = pipeline.detect_device(args.device)
    default_compute = "float16" if t_dev == "cuda" else "int8"
    compute, adjusted = pipeline.resolve_compute(t_dev, default_compute)
    print(f"Device: {t_dev.upper()}   Compute: {compute}")

    # Model
    print(f"Loading model '{args.model}'...")
    model = pipeline.load_model(args.model, t_dev, compute, args.language, beam=args.beam_size,
                                 task=args.task, initial_prompt=args.initial_prompt,
                                 hotwords=args.hotwords,
                                 suppress_numerals=args.suppress_numerals,
                                 vad_method=args.vad_method,
                                 device_index=args.device_index, threads=args.threads,
                                 local_files_only=args.offline, decode=decode)

    # Audio
    print("Loading audio...")
    audio = pipeline.load_audio(args.audio)

    language = args.language
    if language is None:
        try:
            language, prob = pipeline.detect_language(model, audio)
            print(f"  Detected language: {language} ({prob:.2f})")
            if prob < 0.5:
                print(f"  Warning: language guess is uncertain: {language} {prob:.0%}. "
                      "Use --language to pick it.")
        except Exception as e:
            print(f"  Language detection unavailable ({e}) — transcribe will detect it")

    # Transcribe
    print("Transcribing...")
    result, detected_lang = pipeline.transcribe_with_backoff(
        model, audio, pipeline.initial_batch(16, t_dev, args.device_index), language,
        notify=lambda m: print(f"  Warning: {m}"), task=args.task)
    print(f"  Language: {detected_lang}   Segments: {len(result['segments'])}")

    # Align
    if args.task == "translate":
        print("  Note: word timestamps are not available for translated output; "
              "exporting segment timing only.")
        result.setdefault("language", detected_lang)
    elif not args.align_model and not pipeline.align_supported(detected_lang):
        print(f"  Warning: word timestamps are not available for {detected_lang}; "
              "exporting segment timing only.")
        result.setdefault("language", detected_lang)
    else:
        print("Aligning word timestamps...")
        try:
            result = pipeline.align(
                result, detected_lang, a_dev, audio, align_model=args.align_model,
                interpolate_method=args.interpolate_method,
                return_char_alignments=args.char_alignments)
        except Exception as e:
            print(f"  Warning: alignment failed ({e}) — continuing without word timestamps")
            result.setdefault("language", detected_lang)

    # Diarize
    export_opts = {"segment_resolution": args.segment_resolution}
    if args.diarize:
        print("Identifying speakers...")
        try:
            pipeline.ensure_speaker_models(model_dir, offline=False)
            out = pipeline.diarize(
                audio, model_dir, min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                return_embeddings=args.speaker_embeddings)
            segments_df, embeddings = out if args.speaker_embeddings else (out, None)
            result = pipeline.assign_speakers(segments_df, result, embeddings)
            if embeddings:
                export_opts["speaker_embeddings"] = embeddings
        except Exception as e:
            msg = (str(e).splitlines() or [""])[0]
            print(f"  Warning: speaker identification failed ({msg}) "
                  "— continuing without speakers")

    # Export
    out_dir = os.path.abspath(args.output_dir) if args.output_dir \
        else os.path.dirname(os.path.abspath(args.audio))
    os.makedirs(out_dir, exist_ok=True)
    results = pipeline.export(result, args.output or ["word_json"], out_dir,
                              args.audio, export_opts)
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
