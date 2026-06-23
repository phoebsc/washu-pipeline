"""End-to-end pipeline: split audio → transcribe → de-identify.

Processes WashU CDR interview recordings through:
1. Split at timestamp into study partner / subject halves
2. Transcribe each half with whisper.cpp + pyannote diarization
3. De-identify both transcripts together (shared entity codes)

Usage:
    uv run washu-run \
        --audio-dir "/path/to/Extracted Audio Files (MP3 256kbps)" \
        --timestamps "/path/to/timestamp_info.txt" \
        --output output/ \
        [--tape Tape_11_Interview_(Source)_1]
"""

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import torch

from .split_audio import parse_timestamp_file, split_audio
from .transcribe import transcribe, format_as_text
from .deid import load_classifier, DeidMapper, deid_session
from .viewer import generate_html

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


def process_tape(
    audio_path: Path,
    split_ms: int,
    output_dir: Path,
    classifier,
    model_size: str = "large-v3",
    language: str = "en",
) -> dict:
    """Process a single tape through the full pipeline. Returns timing dict."""
    tape_name = audio_path.stem
    tape_output = output_dir / tape_name
    tape_output.mkdir(parents=True, exist_ok=True)
    timings = {"tape": tape_name}

    tape_start = time.time()

    # Step 1: Split audio
    logger.info(f"{'='*60}")
    logger.info(f"Processing: {tape_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Step 1: Splitting at {split_ms/1000:.0f}s...")
    t0 = time.time()
    split_dir = tape_output / "split"
    partner_path, subject_path = split_audio(audio_path, split_ms, split_dir)
    timings["split_s"] = round(time.time() - t0, 1)

    # Step 2a: Transcribe partner
    logger.info("Step 2a: Transcribing partner interview...")
    t0 = time.time()
    partner_transcript = transcribe(
        partner_path,
        num_speakers=2,
        model_size=model_size,
        language=language,
    )
    timings["transcribe_partner_s"] = round(time.time() - t0, 1)

    partner_json_path = tape_output / "partner_transcript.json"
    with open(partner_json_path, "w", encoding="utf-8") as f:
        json.dump(partner_transcript, f, indent=2, ensure_ascii=False)

    partner_txt_path = tape_output / "partner_transcript.txt"
    with open(partner_txt_path, "w", encoding="utf-8") as f:
        f.write(format_as_text(partner_transcript))

    # Step 2b: Transcribe subject
    logger.info("Step 2b: Transcribing subject interview...")
    t0 = time.time()
    subject_transcript = transcribe(
        subject_path,
        num_speakers=2,
        model_size=model_size,
        language=language,
    )
    timings["transcribe_subject_s"] = round(time.time() - t0, 1)

    subject_json_path = tape_output / "subject_transcript.json"
    with open(subject_json_path, "w", encoding="utf-8") as f:
        json.dump(subject_transcript, f, indent=2, ensure_ascii=False)

    subject_txt_path = tape_output / "subject_transcript.txt"
    with open(subject_txt_path, "w", encoding="utf-8") as f:
        f.write(format_as_text(subject_transcript))

    # Free GPU memory from transcription before running deid
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Step 3: De-identify both together (cross-transcript propagation)
    logger.info("Step 3: De-identifying transcripts (joint session)...")
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

    # Step 4: Generate HTML viewer
    logger.info("Step 4: Generating HTML viewer...")
    t0 = time.time()
    html = generate_html(tape_output)
    view_path = tape_output / "view.html"
    view_path.write_text(html)
    timings["viewer_s"] = round(time.time() - t0, 1)

    timings["total_s"] = round(time.time() - tape_start, 1)
    logger.info(
        f"Done: {tape_name} — "
        f"split={timings['split_s']}s, "
        f"transcribe_partner={timings['transcribe_partner_s']}s, "
        f"transcribe_subject={timings['transcribe_subject_s']}s, "
        f"deid={timings['deid_s']}s, "
        f"total={timings['total_s']}s"
    )
    return timings


def cli():
    parser = argparse.ArgumentParser(
        description="Full pipeline: split → transcribe → de-identify WashU CDR interviews"
    )
    parser.add_argument(
        "--audio-dir", required=True,
        help="Directory containing .mp3 interview files",
    )
    parser.add_argument(
        "--timestamps", required=True,
        help="Path to timestamp_info.txt with split points",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--tape", default=None,
        help="Process only this tape (e.g., 'Tape_11_Interview_(Source)_1')",
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
        "--skip-existing", action="store_true",
        help="Skip tapes that already have a view.html (fully processed)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    audio_dir = Path(args.audio_dir)
    timestamps = parse_timestamp_file(Path(args.timestamps))
    output_dir = Path(args.output)

    if args.tape:
        if args.tape not in timestamps:
            raise ValueError(f"Tape '{args.tape}' not found in timestamp file")
        timestamps = {args.tape: timestamps[args.tape]}

    logger.info("Loading de-identification model...")
    t0 = time.time()
    classifier = load_classifier()
    model_load_time = round(time.time() - t0, 1)
    logger.info(f"Model loaded in {model_load_time}s")

    all_timings = []
    run_start = time.time()

    for tape_name, split_ms in timestamps.items():
        if args.skip_existing:
            existing = output_dir / tape_name / "view.html"
            if existing.exists():
                logger.info(f"Skipping (already complete): {tape_name}")
                continue

        audio_path = audio_dir / f"{tape_name}.mp3"
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}, skipping")
            continue
        timings = process_tape(
            audio_path, split_ms, output_dir,
            classifier=classifier,
            model_size=args.model_size,
            language=args.language,
        )
        all_timings.append(timings)

    total_run = round(time.time() - run_start, 1)

    # Write timing log
    log_path = output_dir / "pipeline_timing.json"
    log_data = {
        "model_size": args.model_size,
        "model_load_s": model_load_time,
        "total_run_s": total_run,
        "tapes": all_timings,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info(f"PIPELINE COMPLETE — {len(all_timings)} tapes in {total_run:.0f}s ({total_run/60:.1f}min)")
    logger.info(f"{'='*80}")
    logger.info(f"{'Tape':<40} {'Split':>6} {'Tx-P':>6} {'Tx-S':>6} {'Deid':>6} {'Total':>7}")
    logger.info(f"{'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for t in all_timings:
        logger.info(
            f"{t['tape']:<40} {t['split_s']:>5.0f}s {t['transcribe_partner_s']:>5.0f}s "
            f"{t['transcribe_subject_s']:>5.0f}s {t['deid_s']:>5.0f}s {t['total_s']:>6.0f}s"
        )
    logger.info(f"\nTiming log saved: {log_path}")


if __name__ == "__main__":
    cli()
