"""Split interview audio files at a timestamp into study partner and subject halves.

Reads a timestamp file where each entry is:
    <filename_without_extension>
    MM:SS

The first half (0 → timestamp) is the study partner interview.
The second half (timestamp → end) is the subject interview.
"""

import argparse
import logging
from pathlib import Path

from pydub import AudioSegment

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


def parse_timestamp_file(timestamp_path: Path) -> dict[str, int]:
    """Parse timestamp file into {filename_stem: split_time_ms}."""
    lines = timestamp_path.read_text().strip().splitlines()
    entries = {}
    i = 0
    while i < len(lines) - 1:
        name = lines[i].strip()
        time_str = lines[i + 1].strip()
        parts = time_str.split(":")
        minutes, seconds = int(parts[0]), int(parts[1])
        entries[name] = (minutes * 60 + seconds) * 1000
        i += 2
    return entries


def split_audio(audio_path: Path, split_ms: int, output_dir: Path) -> tuple[Path, Path]:
    """Split audio file at split_ms into partner and subject halves."""
    audio = AudioSegment.from_mp3(str(audio_path))

    partner_audio = audio[:split_ms]
    subject_audio = audio[split_ms:]

    stem = audio_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    partner_path = output_dir / f"{stem}_partner.mp3"
    subject_path = output_dir / f"{stem}_subject.mp3"

    partner_audio.export(str(partner_path), format="mp3")
    logger.info(f"Partner audio: {partner_path} ({len(partner_audio) / 1000:.1f}s)")

    subject_audio.export(str(subject_path), format="mp3")
    logger.info(f"Subject audio: {subject_path} ({len(subject_audio) / 1000:.1f}s)")

    return partner_path, subject_path


def cli():
    parser = argparse.ArgumentParser(
        description="Split interview audio at timestamp into partner/subject halves"
    )
    parser.add_argument(
        "--audio-dir", required=True,
        help="Directory containing .mp3 files",
    )
    parser.add_argument(
        "--timestamps", required=True,
        help="Path to timestamp_info.txt",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for split audio files",
    )
    parser.add_argument(
        "--tape", default=None,
        help="Process only this tape (e.g., 'Tape_11_Interview_(Source)_1')",
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    timestamps = parse_timestamp_file(Path(args.timestamps))
    output_dir = Path(args.output)

    if args.tape:
        if args.tape not in timestamps:
            raise ValueError(f"Tape '{args.tape}' not found in timestamp file")
        timestamps = {args.tape: timestamps[args.tape]}

    for name, split_ms in timestamps.items():
        audio_path = audio_dir / f"{name}.mp3"
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}, skipping")
            continue
        tape_output = output_dir / name
        split_audio(audio_path, split_ms, tape_output)

    logger.info("Done splitting audio files")


if __name__ == "__main__":
    cli()
