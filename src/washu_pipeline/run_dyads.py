"""Pipeline for pre-split dyad recordings.

Each dyad subfolder contains exactly two files:
  partner.mp3      — study partner interview
  participant.mp3  — subject interview

The subject interview may also be named subject.mp3 for compatibility with
legacy split outputs.

Transcribes both, de-identifies them as a joint session, and writes:
  - All intermediate outputs flat into <dyad_dir>/
  - Four shareable files into <output_dir>/<dyad_id>/

Usage:
    uv run washu-run-dyads --input /path/to/folder [--output output/] [--dyad <id>]
"""

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import torch

from .transcribe import transcribe, format_as_text
from .deid import load_classifier, DeidMapper, deid_session
from .output_names import deid_output_name
from .viewer import generate_html_flat

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


def _subject_audio_path(dyad_dir: Path) -> Path:
    """Return the subject audio path, accepting current and legacy filenames."""
    for filename in ("participant.mp3", "subject.mp3"):
        path = dyad_dir / filename
        if path.exists():
            return path
    return dyad_dir / "participant.mp3"


def _shareable_outputs_exist(output_dir: Path, dyad_id: str) -> bool:
    """Return True when all shareable output files for a dyad are present."""
    dyad_output = output_dir / deid_output_name(dyad_id)
    required = (
        "partner_deid.txt",
        "subject_deid.txt",
        "partner_deid_clean.json",
        "subject_deid_clean.json",
    )
    return all((dyad_output / filename).exists() for filename in required)


def _clean_deid(deid_data: dict) -> dict:
    """Return a copy of a deid transcript with 'entities' removed from each utterance."""
    clean_utterances = []
    for utt in deid_data.get("utterances", []):
        clean_utt = {k: v for k, v in utt.items() if k != "entities"}
        clean_utterances.append(clean_utt)
    return {
        "utterances": clean_utterances,
        "text": deid_data.get("text", ""),
        "metadata": deid_data.get("metadata", {}),
    }


def process_dyad(
    dyad_dir: Path,
    output_dir: Path,
    classifier,
    model_size: str = "large-v3",
    language: str = "en",
) -> dict:
    """Process a single dyad through transcription and de-identification.

    Returns a timing dict.
    """
    dyad_id = dyad_dir.name
    partner_audio = dyad_dir / "partner.mp3"
    participant_audio = _subject_audio_path(dyad_dir)

    for path in (partner_audio, participant_audio):
        if not path.exists():
            raise FileNotFoundError(f"Expected audio file not found: {path}")

    timings = {"dyad": dyad_id}
    dyad_start = time.time()

    logger.info(f"{'='*60}")
    logger.info(f"Processing dyad: {dyad_id}")
    logger.info(f"{'='*60}")

    # Step 1: Transcribe partner
    logger.info("Step 1a: Transcribing partner...")
    t0 = time.time()
    partner_transcript = transcribe(
        partner_audio,
        num_speakers=2,
        model_size=model_size,
        language=language,
    )
    timings["transcribe_partner_s"] = round(time.time() - t0, 1)

    with open(dyad_dir / "partner_transcript.json", "w", encoding="utf-8") as f:
        json.dump(partner_transcript, f, indent=2, ensure_ascii=False)
    with open(dyad_dir / "partner_transcript.txt", "w", encoding="utf-8") as f:
        f.write(format_as_text(partner_transcript))

    # Step 1b: Transcribe participant
    logger.info("Step 1b: Transcribing participant...")
    t0 = time.time()
    subject_transcript = transcribe(
        participant_audio,
        num_speakers=2,
        model_size=model_size,
        language=language,
    )
    timings["transcribe_subject_s"] = round(time.time() - t0, 1)

    with open(dyad_dir / "subject_transcript.json", "w", encoding="utf-8") as f:
        json.dump(subject_transcript, f, indent=2, ensure_ascii=False)
    with open(dyad_dir / "subject_transcript.txt", "w", encoding="utf-8") as f:
        f.write(format_as_text(subject_transcript))

    # Free GPU memory before deid
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Step 2: De-identify both together
    logger.info("Step 2: De-identifying (joint session)...")
    t0 = time.time()
    mapper = DeidMapper()
    partner_deid, subject_deid = deid_session(
        [partner_transcript, subject_transcript], classifier, mapper
    )
    timings["deid_s"] = round(time.time() - t0, 1)

    # Write full deid files into dyad dir (flat)
    with open(dyad_dir / "partner_deid.json", "w", encoding="utf-8") as f:
        json.dump(partner_deid, f, indent=2, ensure_ascii=False)
    with open(dyad_dir / "subject_deid.json", "w", encoding="utf-8") as f:
        json.dump(subject_deid, f, indent=2, ensure_ascii=False)
    with open(dyad_dir / "deid_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapper.get_mapping(), f, indent=2, ensure_ascii=False)

    # Step 3: Generate HTML viewer
    logger.info("Step 3: Generating HTML viewer...")
    t0 = time.time()
    html = generate_html_flat(dyad_dir)
    (dyad_dir / "view.html").write_text(html)
    timings["viewer_s"] = round(time.time() - t0, 1)

    # Step 4: Write the 4 shareable files to output/<dyad_id-without-date>/
    dyad_output = output_dir / deid_output_name(dyad_id)
    dyad_output.mkdir(parents=True, exist_ok=True)

    with open(dyad_output / "partner_deid.txt", "w", encoding="utf-8") as f:
        f.write(partner_deid["text"])
    with open(dyad_output / "subject_deid.txt", "w", encoding="utf-8") as f:
        f.write(subject_deid["text"])
    with open(dyad_output / "partner_deid_clean.json", "w", encoding="utf-8") as f:
        json.dump(_clean_deid(partner_deid), f, indent=2, ensure_ascii=False)
    with open(dyad_output / "subject_deid_clean.json", "w", encoding="utf-8") as f:
        json.dump(_clean_deid(subject_deid), f, indent=2, ensure_ascii=False)

    timings["total_s"] = round(time.time() - dyad_start, 1)
    logger.info(
        f"Done: {dyad_id} — "
        f"tx_partner={timings['transcribe_partner_s']}s, "
        f"tx_subject={timings['transcribe_subject_s']}s, "
        f"deid={timings['deid_s']}s, "
        f"total={timings['total_s']}s"
    )
    return timings


def cli():
    parser = argparse.ArgumentParser(
        description="Transcribe and de-identify pre-split dyad recordings"
    )
    parser.add_argument(
        "--input", required=True,
        help="Folder containing dyad subfolders, each with partner.mp3 and participant.mp3 or subject.mp3",
    )
    parser.add_argument(
        "--output", default="output",
        help="Directory for the 4 shareable output files per dyad (default: output/)",
    )
    parser.add_argument(
        "--dyad", default=None,
        help="Process only this dyad subfolder name",
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
        help="Skip dyads whose shareable files already exist in the output directory",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    if args.dyad:
        dyad_dirs = [input_dir / args.dyad]
        if not dyad_dirs[0].is_dir():
            raise FileNotFoundError(f"Dyad folder not found: {dyad_dirs[0]}")
    else:
        dyad_dirs = sorted(
            d for d in input_dir.iterdir()
            if d.is_dir() and (d / "partner.mp3").exists() and _subject_audio_path(d).exists()
        )
        if not dyad_dirs:
            raise ValueError(f"No valid dyad folders found in {input_dir}")

    logger.info("Loading de-identification model...")
    t0 = time.time()
    classifier = load_classifier()
    logger.info(f"Model loaded in {round(time.time() - t0, 1)}s")

    all_timings = []
    run_start = time.time()

    for dyad_dir in dyad_dirs:
        if args.skip_existing and _shareable_outputs_exist(output_dir, dyad_dir.name):
            logger.info(f"Skipping (already complete): {dyad_dir.name}")
            continue
        try:
            timings = process_dyad(
                dyad_dir, output_dir,
                classifier=classifier,
                model_size=args.model_size,
                language=args.language,
            )
            all_timings.append(timings)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {dyad_dir.name}: {e}")

    total_run = round(time.time() - run_start, 1)

    logger.info(f"\n{'='*80}")
    logger.info(f"DONE — {len(all_timings)} dyads in {total_run:.0f}s ({total_run/60:.1f}min)")
    logger.info(f"{'='*80}")
    logger.info(f"{'Dyad':<40} {'Tx-P':>6} {'Tx-S':>6} {'Deid':>6} {'Total':>7}")
    logger.info(f"{'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for t in all_timings:
        logger.info(
            f"{t['dyad']:<40} {t['transcribe_partner_s']:>5.0f}s "
            f"{t['transcribe_subject_s']:>5.0f}s {t['deid_s']:>5.0f}s {t['total_s']:>6.0f}s"
        )


if __name__ == "__main__":
    cli()
