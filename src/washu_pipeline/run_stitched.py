"""Pipeline for stitched WashU CDR interview recordings.

Input: a folder of .mp3 files, each containing partner + subject interviews
stitched together (partner first, subject second, no known split point).

Pipeline:
1. Transcribe full audio (whisper.cpp, full context)
2. Detect split point (Ollama + Gemma 4 LLM)
3. Diarize each half separately (pyannote, num_speakers=2)
4. Align whisper segments to speaker turns
5. De-identify both halves (joint session)

Usage:
    uv run washu-run-stitched --input "/path/to/mp3s/" [--output output/] [--tape <name>]
"""

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import librosa
import numpy as np
import torch

from .transcribe import (
    SAMPLE_RATE,
    align_segments_to_speakers,
    diarize_array,
    format_as_text,
    get_device,
    get_model_path,
    load_diarization_pipeline,
    load_whisper_model,
    merge_consecutive_segments,
    remap_speakers,
    transcribe_full_audio,
)
from .ollama_split import detect_split_point, OLLAMA_MODEL
from .deid import load_classifier, DeidMapper, deid_session
from .viewer import generate_html

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


def _partition_segments(
    whisper_segments: list[dict],
    split_time: float,
) -> tuple[list[dict], list[dict]]:
    """Partition whisper segments at the split point.

    Segments straddling the boundary are assigned to the side with more duration.
    Subject segments have their timestamps offset to start from 0.
    """
    partner = []
    subject = []

    for seg in whisper_segments:
        if seg["end"] <= split_time:
            partner.append(seg)
        elif seg["start"] >= split_time:
            subject.append({
                **seg,
                "start": round(seg["start"] - split_time, 3),
                "end": round(seg["end"] - split_time, 3),
            })
        else:
            partner_portion = split_time - seg["start"]
            subject_portion = seg["end"] - split_time
            if partner_portion >= subject_portion:
                partner.append(seg)
            else:
                subject.append({
                    **seg,
                    "start": round(seg["start"] - split_time, 3),
                    "end": round(seg["end"] - split_time, 3),
                })

    return partner, subject


def _build_transcript_dict(
    utterances: list[dict],
    model_size: str,
    audio_duration: float,
    language: str,
    half_label: str,
) -> dict:
    """Build the standard transcript output dict."""
    full_text = "\n".join(f"{u['speaker']}: {u['text']}" for u in utterances)
    return {
        "utterances": utterances,
        "text": full_text,
        "metadata": {
            "model": f"whisper.cpp/ggml-{model_size}",
            "diarization_model": "pyannote/speaker-diarization-3.1",
            "num_speakers": 2,
            "device": "Metal (whisper.cpp), mps (pyannote)",
            "audio_duration_seconds": round(audio_duration, 2),
            "language": language,
            "source": half_label,
        },
    }


def process_stitched(
    audio_path: Path,
    output_dir: Path,
    classifier,
    model_size: str = "large-v3",
    language: str = "en",
    ollama_model: str = OLLAMA_MODEL,
) -> dict:
    """Process a single stitched recording through the full pipeline."""
    tape_name = audio_path.stem
    tape_output = output_dir / tape_name
    tape_output.mkdir(parents=True, exist_ok=True)
    timings = {"tape": tape_name}
    tape_start = time.time()

    logger.info(f"{'=' * 60}")
    logger.info(f"Processing: {tape_name}")
    logger.info(f"{'=' * 60}")

    # Step 1: Transcribe full audio
    logger.info("Step 1: Transcribing full audio...")
    t0 = time.time()

    model_path = get_model_path(model_size)
    whisper_model = load_whisper_model(model_path)
    whisper_segments = transcribe_full_audio(audio_path, whisper_model, language)
    del whisper_model
    timings["transcribe_s"] = round(time.time() - t0, 1)

    # Load audio for duration and later diarization
    audio = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)[0]
    audio_duration = len(audio) / SAMPLE_RATE
    logger.info(f"Audio duration: {audio_duration:.1f}s ({audio_duration/60:.1f}min)")

    # Save raw transcript
    raw_transcript = {
        "segments": whisper_segments,
        "metadata": {
            "model": f"whisper.cpp/ggml-{model_size}",
            "audio_duration_seconds": round(audio_duration, 2),
            "num_segments": len(whisper_segments),
        },
    }
    with open(tape_output / "full_transcript.json", "w", encoding="utf-8") as f:
        json.dump(raw_transcript, f, indent=2, ensure_ascii=False)

    # Step 2: Detect split point with LLM
    logger.info("Step 2: Detecting split point with LLM...")
    t0 = time.time()
    split_info = detect_split_point(whisper_segments, audio_duration, ollama_model)
    timings["split_detect_s"] = round(time.time() - t0, 1)

    with open(tape_output / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    split_time = split_info["split_timestamp_seconds"]
    logger.info(f"Split point: {split_time:.1f}s ({split_time/60:.1f}min)")

    # Partition whisper segments
    partner_segments, subject_segments = _partition_segments(whisper_segments, split_time)
    logger.info(
        f"Partner: {len(partner_segments)} segments, "
        f"Subject: {len(subject_segments)} segments"
    )

    # Step 3: Diarize each half
    logger.info("Step 3: Diarizing each half...")
    t0 = time.time()

    device = get_device()
    hf_token = __import__("os").environ.get("HF_TOKEN")
    diarization_pipeline = load_diarization_pipeline(device, hf_token)

    split_sample = int(split_time * SAMPLE_RATE)
    partner_audio = audio[:split_sample]
    subject_audio = audio[split_sample:]

    logger.info("  Diarizing partner half...")
    partner_diar = diarize_array(partner_audio, diarization_pipeline, num_speakers=2)

    logger.info("  Diarizing subject half...")
    subject_diar = diarize_array(subject_audio, diarization_pipeline, num_speakers=2)

    del diarization_pipeline, audio, partner_audio, subject_audio
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    timings["diarize_s"] = round(time.time() - t0, 1)

    # Step 4: Align whisper segments to speaker turns
    logger.info("Step 4: Aligning segments to speakers...")
    partner_utterances = align_segments_to_speakers(partner_segments, partner_diar)
    subject_utterances = align_segments_to_speakers(subject_segments, subject_diar)

    partner_utterances = merge_consecutive_segments(partner_utterances)
    subject_utterances = merge_consecutive_segments(subject_utterances)

    partner_utterances = remap_speakers(partner_utterances)
    subject_utterances = remap_speakers(subject_utterances)

    # Build transcript dicts
    partner_duration = split_time
    subject_duration = audio_duration - split_time

    partner_transcript = _build_transcript_dict(
        partner_utterances, model_size, partner_duration, language, "partner"
    )
    subject_transcript = _build_transcript_dict(
        subject_utterances, model_size, subject_duration, language, "subject"
    )

    # Save transcripts
    with open(tape_output / "partner_transcript.json", "w", encoding="utf-8") as f:
        json.dump(partner_transcript, f, indent=2, ensure_ascii=False)
    with open(tape_output / "partner_transcript.txt", "w", encoding="utf-8") as f:
        f.write(format_as_text(partner_transcript))

    with open(tape_output / "subject_transcript.json", "w", encoding="utf-8") as f:
        json.dump(subject_transcript, f, indent=2, ensure_ascii=False)
    with open(tape_output / "subject_transcript.txt", "w", encoding="utf-8") as f:
        f.write(format_as_text(subject_transcript))

    # Step 5: De-identify
    logger.info("Step 5: De-identifying transcripts (joint session)...")
    t0 = time.time()
    mapper = DeidMapper()
    partner_deid, subject_deid = deid_session(
        [partner_transcript, subject_transcript], classifier, mapper
    )
    timings["deid_s"] = round(time.time() - t0, 1)

    deid_dir = tape_output / "deid"
    deid_dir.mkdir(parents=True, exist_ok=True)

    with open(deid_dir / "partner_deid.json", "w", encoding="utf-8") as f:
        json.dump(partner_deid, f, indent=2, ensure_ascii=False)
    with open(deid_dir / "partner_deid.txt", "w", encoding="utf-8") as f:
        f.write(partner_deid["text"])

    with open(deid_dir / "subject_deid.json", "w", encoding="utf-8") as f:
        json.dump(subject_deid, f, indent=2, ensure_ascii=False)
    with open(deid_dir / "subject_deid.txt", "w", encoding="utf-8") as f:
        f.write(subject_deid["text"])

    with open(deid_dir / "deid_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapper.get_mapping(), f, indent=2, ensure_ascii=False)

    # Step 6: Generate HTML viewer
    logger.info("Step 6: Generating HTML viewer...")
    t0 = time.time()
    html = generate_html(tape_output)
    (tape_output / "view.html").write_text(html)
    timings["viewer_s"] = round(time.time() - t0, 1)

    timings["total_s"] = round(time.time() - tape_start, 1)
    logger.info(
        f"Done: {tape_name} — "
        f"transcribe={timings['transcribe_s']}s, "
        f"split_detect={timings['split_detect_s']}s, "
        f"diarize={timings['diarize_s']}s, "
        f"deid={timings['deid_s']}s, "
        f"total={timings['total_s']}s"
    )
    return timings


def cli():
    parser = argparse.ArgumentParser(
        description="Transcribe and de-identify stitched WashU CDR recordings"
    )
    parser.add_argument(
        "--input", required=True,
        help="Folder containing .mp3 files (stitched partner+subject recordings)",
    )
    parser.add_argument(
        "--output", default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--tape", default=None,
        help="Process only this tape (filename without .mp3 extension)",
    )
    parser.add_argument(
        "--model-size", default="large-v3",
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                 "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"],
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--language", default="en",
        help="Language code for transcription (default: en)",
    )
    parser.add_argument(
        "--ollama-model", default=OLLAMA_MODEL,
        help=f"Ollama model for split detection (default: {OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip tapes that already have a view.html (fully processed)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    if args.tape:
        mp3_files = [input_dir / f"{args.tape}.mp3"]
        if not mp3_files[0].exists():
            raise FileNotFoundError(f"Audio file not found: {mp3_files[0]}")
    else:
        mp3_files = sorted(input_dir.glob("*.mp3"))
        if not mp3_files:
            raise ValueError(f"No .mp3 files found in {input_dir}")

    logger.info("Loading de-identification model...")
    t0 = time.time()
    classifier = load_classifier()
    logger.info(f"Deid model loaded in {round(time.time() - t0, 1)}s")

    all_timings = []
    run_start = time.time()

    for audio_path in mp3_files:
        if args.skip_existing:
            existing = output_dir / audio_path.stem / "view.html"
            if existing.exists():
                logger.info(f"Skipping (already complete): {audio_path.stem}")
                continue

        try:
            timings = process_stitched(
                audio_path, output_dir,
                classifier=classifier,
                model_size=args.model_size,
                language=args.language,
                ollama_model=args.ollama_model,
            )
            all_timings.append(timings)
        except Exception as e:
            logger.error(f"Failed on {audio_path.stem}: {e}")
            raise

    total_run = round(time.time() - run_start, 1)

    # Print summary
    logger.info(f"\n{'=' * 80}")
    logger.info(f"DONE — {len(all_timings)} tapes in {total_run:.0f}s ({total_run/60:.1f}min)")
    logger.info(f"{'=' * 80}")
    logger.info(f"{'Tape':<45} {'Tx':>5} {'Split':>6} {'Diar':>5} {'Deid':>5} {'Total':>6}")
    logger.info(f"{'-' * 45} {'-' * 5} {'-' * 6} {'-' * 5} {'-' * 5} {'-' * 6}")
    for t in all_timings:
        logger.info(
            f"{t['tape']:<45} {t['transcribe_s']:>4.0f}s "
            f"{t['split_detect_s']:>5.0f}s {t['diarize_s']:>4.0f}s "
            f"{t['deid_s']:>4.0f}s {t['total_s']:>5.0f}s"
        )


if __name__ == "__main__":
    cli()
