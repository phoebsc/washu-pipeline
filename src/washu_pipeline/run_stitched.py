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
import re
import shutil
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
    remap_speakers,
    transcribe_full_audio,
    transcribe_vad_chunks,
)
from .ollama_split import detect_split_point, OLLAMA_MODEL
from .ollama_score import score_transcript
from .deid import load_classifier, DeidMapper, deid_session
from .viewer import generate_html

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


def _extract_date_from_filename(filename: str) -> str | None:
    """Extract a trailing date (M-D-YY or M-D-YYYY) from a filename stem.

    Expects the date as the last dash-separated numeric components, e.g.:
      Tape_11_Interview_(Source)_1_12-21-24  -> 12-21-24
      SomeFile_9-19-2023                     -> 9-19-2023
    """
    match = re.search(r"(\d{1,2}-\d{1,2}-\d{2,4})$", filename)
    if match:
        return match.group(1)
    return None


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
            "merge_consecutive_segments": False,
        },
    }


def process_stitched(
    audio_path: Path,
    output_dir: Path,
    classifier,
    model_size: str = "large-v3",
    language: str = "en",
    ollama_model: str = OLLAMA_MODEL,
    vad_chunked_transcription: bool = False,
    stop_after_transcription: bool = False,
    deid_output_dir: Path | None = None,
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

    # Step 1: Transcribe full audio (skip if cached)
    cached_transcript = tape_output / "full_transcript.json"
    expected_model = f"whisper.cpp/ggml-{model_size}"
    expected_mode = "vad_chunks" if vad_chunked_transcription else "full_audio"
    use_cached_transcript = False
    regenerated_transcript = False
    if cached_transcript.exists():
        logger.info("Step 1: Loading cached transcription from full_transcript.json")
        with open(cached_transcript, encoding="utf-8") as f:
            raw_transcript = json.load(f)

        metadata = raw_transcript.get("metadata", {})
        cached_model = metadata.get("model")
        cached_mode = metadata.get("transcription_mode", "full_audio")
        use_cached_transcript = (
            cached_model == expected_model
            and cached_mode == expected_mode
        )
        if use_cached_transcript:
            whisper_segments = raw_transcript["segments"]
            audio_duration = metadata["audio_duration_seconds"]
            timings["transcribe_s"] = 0.0
            logger.info(
                f"Loaded {len(whisper_segments)} segments, "
                f"duration: {audio_duration:.1f}s ({audio_duration/60:.1f}min)"
            )
        else:
            logger.info(
                "Cached transcription does not match requested model/mode "
                f"({cached_model}, {cached_mode}); regenerating"
            )

    if not use_cached_transcript:
        regenerated_transcript = True
        logger.info("Step 1: Transcribing full audio...")
        t0 = time.time()

        model_path = get_model_path(model_size)
        whisper_model = load_whisper_model(model_path)
        if vad_chunked_transcription:
            whisper_segments = transcribe_vad_chunks(audio_path, whisper_model, language)
        else:
            whisper_segments = transcribe_full_audio(audio_path, whisper_model, language)
        del whisper_model
        timings["transcribe_s"] = round(time.time() - t0, 1)

        # Load audio for duration
        audio = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)[0]
        audio_duration = len(audio) / SAMPLE_RATE
        logger.info(f"Audio duration: {audio_duration:.1f}s ({audio_duration/60:.1f}min)")

        # Save raw transcript
        raw_transcript = {
            "segments": whisper_segments,
            "metadata": {
                "model": expected_model,
                "transcription_mode": expected_mode,
                "audio_duration_seconds": round(audio_duration, 2),
                "num_segments": len(whisper_segments),
            },
        }
        with open(cached_transcript, "w", encoding="utf-8") as f:
            json.dump(raw_transcript, f, indent=2, ensure_ascii=False)

    if stop_after_transcription:
        logger.info("Stopping after Step 1 transcription as requested")
        timings["total_s"] = round(time.time() - tape_start, 1)
        return timings

    # Step 2: Detect split point with LLM
    split_info_path = tape_output / "split_info.json"
    use_cached_split = False
    if split_info_path.exists() and not regenerated_transcript:
        logger.info("Step 2: Loading cached split point from split_info.json")
        with open(split_info_path, encoding="utf-8") as f:
            split_info = json.load(f)
        timings["split_detect_s"] = 0.0
        use_cached_split = True
    else:
        logger.info("Step 2: Detecting split point with LLM...")
        t0 = time.time()
        split_info = detect_split_point(whisper_segments, audio_duration, ollama_model)
        timings["split_detect_s"] = round(time.time() - t0, 1)

        with open(split_info_path, "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)

    split_time = split_info["split_timestamp_seconds"]
    logger.info(f"Split point: {split_time:.1f}s ({split_time/60:.1f}min)")

    # Partition whisper segments
    partner_segments, subject_segments = _partition_segments(whisper_segments, split_time)
    logger.info(
        f"Partner: {len(partner_segments)} segments, "
        f"Subject: {len(subject_segments)} segments"
    )

    partner_json_path = tape_output / "partner_transcript.json"
    subject_json_path = tape_output / "subject_transcript.json"
    use_cached_transcripts = False
    if use_cached_split and partner_json_path.exists() and subject_json_path.exists():
        with open(partner_json_path, encoding="utf-8") as f:
            partner_transcript = json.load(f)
        with open(subject_json_path, encoding="utf-8") as f:
            subject_transcript = json.load(f)

        partner_no_merge = (
            partner_transcript.get("metadata", {}).get("merge_consecutive_segments") is False
        )
        subject_no_merge = (
            subject_transcript.get("metadata", {}).get("merge_consecutive_segments") is False
        )
        use_cached_transcripts = partner_no_merge and subject_no_merge
        if use_cached_transcripts:
            logger.info("Step 3/4: Loading cached no-merge partner/subject transcripts")
            timings["diarize_s"] = 0.0
        else:
            logger.info(
                "Cached partner/subject transcripts were generated with merged "
                "utterances; regenerating no-merge transcripts"
            )
    else:
        partner_transcript = subject_transcript = None

    if not use_cached_transcripts:
        # Step 3: Diarize each half
        logger.info("Step 3: Diarizing each half...")
        t0 = time.time()

        audio = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)[0]
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
        interview_date = _extract_date_from_filename(tape_name)
        date_header = f"[date] {interview_date} [date]\n" if interview_date else ""

        with open(partner_json_path, "w", encoding="utf-8") as f:
            json.dump(partner_transcript, f, indent=2, ensure_ascii=False)
        with open(tape_output / "partner_transcript.txt", "w", encoding="utf-8") as f:
            f.write(date_header + format_as_text(partner_transcript))

        with open(subject_json_path, "w", encoding="utf-8") as f:
            json.dump(subject_transcript, f, indent=2, ensure_ascii=False)
        with open(tape_output / "subject_transcript.txt", "w", encoding="utf-8") as f:
            f.write(date_header + format_as_text(subject_transcript))

    # Step 5: Score transcript (Ollama)
    scores_path = tape_output / "scores.json"
    if scores_path.exists() and use_cached_transcripts:
        logger.info("Step 5: Loading cached scores")
        timings["score_s"] = 0.0
    else:
        logger.info("Step 5: Scoring transcript with LLM...")
        t0 = time.time()
        scores = score_transcript(tape_output, model=ollama_model)
        timings["score_s"] = round(time.time() - t0, 1)
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

    # Copy scores.json to deid_output if configured
    if deid_output_dir is not None:
        tape_deid_output = deid_output_dir / tape_name
        tape_deid_output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scores_path, tape_deid_output / "scores.json")

    # Step 6: De-identify
    deid_dir = tape_output / "deid"
    deid_dir.mkdir(parents=True, exist_ok=True)
    partner_deid_path = deid_dir / "partner_deid.json"
    subject_deid_path = deid_dir / "subject_deid.json"
    mapping_path = deid_dir / "deid_mapping.json"
    use_cached_deid = (
        use_cached_transcripts
        and partner_deid_path.exists()
        and subject_deid_path.exists()
        and mapping_path.exists()
    )
    if use_cached_deid:
        logger.info("Step 6: Loading cached de-identified transcripts")
        timings["deid_s"] = 0.0
        # Ensure deid_output has copies even when using cache
        if deid_output_dir is not None:
            tape_deid_output = deid_output_dir / tape_name
            tape_deid_output.mkdir(parents=True, exist_ok=True)
            partner_deid_txt = deid_dir / "partner_deid.txt"
            subject_deid_txt = deid_dir / "subject_deid.txt"
            if partner_deid_txt.exists():
                shutil.copy2(partner_deid_txt, tape_deid_output / "partner_deid.txt")
            if subject_deid_txt.exists():
                shutil.copy2(subject_deid_txt, tape_deid_output / "subject_deid.txt")
    else:
        logger.info("Step 6: De-identifying transcripts (joint session)...")
        t0 = time.time()
        mapper = DeidMapper()
        partner_deid, subject_deid = deid_session(
            [partner_transcript, subject_transcript], classifier, mapper
        )
        timings["deid_s"] = round(time.time() - t0, 1)

        with open(partner_deid_path, "w", encoding="utf-8") as f:
            json.dump(partner_deid, f, indent=2, ensure_ascii=False)
        with open(deid_dir / "partner_deid.txt", "w", encoding="utf-8") as f:
            f.write(partner_deid["text"])

        with open(subject_deid_path, "w", encoding="utf-8") as f:
            json.dump(subject_deid, f, indent=2, ensure_ascii=False)
        with open(deid_dir / "subject_deid.txt", "w", encoding="utf-8") as f:
            f.write(subject_deid["text"])

        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapper.get_mapping(), f, indent=2, ensure_ascii=False)

        # Copy deid .txt files to deid_output if configured
        if deid_output_dir is not None:
            tape_deid_output = deid_output_dir / tape_name
            tape_deid_output.mkdir(parents=True, exist_ok=True)
            shutil.copy2(deid_dir / "partner_deid.txt", tape_deid_output / "partner_deid.txt")
            shutil.copy2(deid_dir / "subject_deid.txt", tape_deid_output / "subject_deid.txt")

    # Step 7: Generate HTML viewer
    view_path = tape_output / "view.html"
    if view_path.exists() and use_cached_deid:
        logger.info("Step 7: Using cached HTML viewer")
        timings["viewer_s"] = 0.0
    else:
        logger.info("Step 7: Generating HTML viewer...")
        t0 = time.time()
        html = generate_html(tape_output)
        view_path.write_text(html)
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
        "--input", default="../interview_data",
        help="Folder containing .mp3 files (default: ../interview_data)",
    )
    parser.add_argument(
        "--output", default="../output_to_be_removed",
        help="Output directory (default: ../output_to_be_removed)",
    )
    parser.add_argument(
        "--deid-output", default="deid_output",
        help="Directory for deliverable deid files: partner_deid.txt, subject_deid.txt, scores.json (default: deid_output/)",
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
    parser.add_argument(
        "--vad-chunked-transcription", action="store_true",
        help="Use speech-aware short chunks for the initial stitched transcription",
    )
    parser.add_argument(
        "--stop-after-transcription", action="store_true",
        help="Stop after writing full_transcript.json",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    deid_output_dir = Path(args.deid_output)

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

    if args.skip_existing:
        original_count = len(mp3_files)
        mp3_files = [
            path for path in mp3_files
            if not (output_dir / path.stem / "view.html").exists()
        ]
        skipped_count = original_count - len(mp3_files)
        if skipped_count:
            logger.info(f"Skipping {skipped_count} already-complete tape(s)")

    classifier = None
    if mp3_files and not args.stop_after_transcription:
        logger.info("Loading de-identification model...")
        t0 = time.time()
        classifier = load_classifier()
        logger.info(f"Deid model loaded in {round(time.time() - t0, 1)}s")

    all_timings = []
    run_start = time.time()

    for audio_path in mp3_files:
        try:
            timings = process_stitched(
                audio_path, output_dir,
                classifier=classifier,
                model_size=args.model_size,
                language=args.language,
                ollama_model=args.ollama_model,
                vad_chunked_transcription=args.vad_chunked_transcription,
                stop_after_transcription=args.stop_after_transcription,
                deid_output_dir=deid_output_dir,
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
            f"{t.get('split_detect_s', 0):>5.0f}s {t.get('diarize_s', 0):>4.0f}s "
            f"{t.get('deid_s', 0):>4.0f}s {t.get('total_s', t['transcribe_s']):>5.0f}s"
        )


if __name__ == "__main__":
    cli()
