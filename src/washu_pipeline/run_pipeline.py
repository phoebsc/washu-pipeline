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
import json
import logging
from pathlib import Path

from .split_audio import parse_timestamp_file, split_audio
from .transcribe import transcribe, format_as_text
from .deid import load_classifier, DeidMapper, deid_transcript

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
):
    """Process a single tape through the full pipeline."""
    tape_name = audio_path.stem
    tape_output = output_dir / tape_name
    tape_output.mkdir(parents=True, exist_ok=True)

    # Step 1: Split audio
    logger.info(f"{'='*60}")
    logger.info(f"Processing: {tape_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Step 1: Splitting at {split_ms/1000:.0f}s...")
    split_dir = tape_output / "split"
    partner_path, subject_path = split_audio(audio_path, split_ms, split_dir)

    # Step 2: Transcribe each half
    logger.info("Step 2a: Transcribing partner interview...")
    partner_transcript = transcribe(
        partner_path,
        num_speakers=2,
        model_size=model_size,
        language=language,
    )

    partner_json_path = tape_output / "partner_transcript.json"
    with open(partner_json_path, "w", encoding="utf-8") as f:
        json.dump(partner_transcript, f, indent=2, ensure_ascii=False)

    partner_txt_path = tape_output / "partner_transcript.txt"
    with open(partner_txt_path, "w", encoding="utf-8") as f:
        f.write(format_as_text(partner_transcript))

    logger.info("Step 2b: Transcribing subject interview...")
    subject_transcript = transcribe(
        subject_path,
        num_speakers=2,
        model_size=model_size,
        language=language,
    )

    subject_json_path = tape_output / "subject_transcript.json"
    with open(subject_json_path, "w", encoding="utf-8") as f:
        json.dump(subject_transcript, f, indent=2, ensure_ascii=False)

    subject_txt_path = tape_output / "subject_transcript.txt"
    with open(subject_txt_path, "w", encoding="utf-8") as f:
        f.write(format_as_text(subject_transcript))

    # Step 3: De-identify both together (shared mapper for consistent codes)
    logger.info("Step 3: De-identifying transcripts...")
    mapper = DeidMapper()

    partner_deid = deid_transcript(partner_transcript, classifier, mapper)
    subject_deid = deid_transcript(subject_transcript, classifier, mapper)

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

    logger.info(f"Done: {tape_name} → {tape_output}")
    return tape_output


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
    classifier = load_classifier()

    for tape_name, split_ms in timestamps.items():
        audio_path = audio_dir / f"{tape_name}.mp3"
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}, skipping")
            continue
        process_tape(
            audio_path, split_ms, output_dir,
            classifier=classifier,
            model_size=args.model_size,
            language=args.language,
        )

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    cli()
