"""Transcribe interview audio using whisper.cpp (Metal-accelerated) + pyannote diarization.

Adapted from vcdr_monitoring/transcribe_whisper_cpp.py for the WashU pipeline.
Speakers are labeled as 'interviewer' and 'participant'.

After initial model download, runs completely locally with no network calls.

Pipeline: diarize first (exclusive mode, min_duration_off=0.3), then transcribe each segment
individually with whisper.cpp.

Produces:
  <output_dir>/diarized_transcript.json   — structured JSON with speaker-labeled utterances
  <output_dir>/diarized_transcript.txt    — human-readable formatted version
"""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pywhispercpp.model import Model as WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

SAMPLE_RATE = 16000

MODELS_DIR = Path.home() / ".cache" / "whisper-cpp-models"

GGML_MODEL_URLS = {
    "tiny": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
    "tiny.en": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
    "base": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
    "base.en": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
    "small": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
    "small.en": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
    "medium": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
    "medium.en": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin",
    "large-v1": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin",
    "large-v2": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin",
    "large-v3": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
    "large-v3-turbo": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
}


def get_model_path(model_size: str) -> Path:
    """Get or download the GGML model file. Downloads once, then fully local."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_file = MODELS_DIR / f"ggml-{model_size}.bin"

    if model_file.exists():
        logger.info(f"Using cached model: {model_file}")
        return model_file

    if model_size not in GGML_MODEL_URLS:
        raise ValueError(
            f"Unknown model size '{model_size}'. "
            f"Available: {', '.join(sorted(GGML_MODEL_URLS.keys()))}"
        )

    url = GGML_MODEL_URLS[model_size]
    logger.info(f"Downloading model '{model_size}' to {model_file} (one-time download)...")

    import urllib.request
    urllib.request.urlretrieve(url, model_file)
    logger.info(f"Download complete: {model_file}")
    return model_file


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_diarization_pipeline(device: torch.device, hf_token: str | None = None) -> Pipeline:
    diarization = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    diarization.instantiate({
        "segmentation": {"min_duration_off": 0.3},
        "clustering": {"method": "centroid", "min_cluster_size": 12, "threshold": 0.7045},
    })
    diarization.to(device)
    return diarization


def load_audio(audio_path: Path) -> np.ndarray:
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return audio


def load_whisper_model(model_path: Path) -> WhisperModel:
    """Load whisper.cpp model with anti-hallucination parameters."""
    return WhisperModel(
        str(model_path),
        print_realtime=False,
        print_progress=False,
        entropy_thold=2.0,
        logprob_thold=-0.8,
        no_speech_thold=0.5,
        suppress_blank=True,
        suppress_nst=True,
        max_tokens=100,
        temperature=0.0,
        temperature_inc=0.2,
    )


def transcribe_segment(
    audio: np.ndarray,
    segment: dict,
    model: WhisperModel,
    language: str = "en",
) -> dict:
    """Transcribe a single diarization segment using whisper.cpp."""
    start_sample = int(segment["start"] * SAMPLE_RATE)
    end_sample = int(segment["end"] * SAMPLE_RATE)
    chunk = audio[start_sample:end_sample]

    if len(chunk) < SAMPLE_RATE * 0.1:
        return {**segment, "text": ""}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, chunk, SAMPLE_RATE)

    try:
        segments = model.transcribe(tmp_path, language=language)
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {**segment, "text": text}


def diarize(audio_path: Path, diarization_pipeline: Pipeline, num_speakers: int) -> list[dict]:
    logger.info(f"Running diarization on {audio_path} (num_speakers={num_speakers})")
    waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    chunk_samples = 10 * SAMPLE_RATE
    remainder = len(waveform) % chunk_samples
    if remainder > 0:
        pad_length = chunk_samples - remainder
        waveform = np.pad(waveform, (0, pad_length), mode="constant")
    waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
    audio_input = {"waveform": waveform_tensor, "sample_rate": SAMPLE_RATE}
    result = diarization_pipeline(audio_input, num_speakers=num_speakers)

    annotation = result.exclusive_speaker_diarization

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
        })

    logger.info(f"Diarization found {len(segments)} segments")
    return segments


def merge_consecutive_segments(segments: list[dict], gap_threshold: float = 0.5) -> list[dict]:
    """Merge consecutive segments from the same speaker if the gap is small."""
    if not segments:
        return []

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        if seg["speaker"] == prev["speaker"] and gap <= gap_threshold:
            prev["end"] = seg["end"]
            prev["text"] = (prev.get("text", "") + " " + seg.get("text", "")).strip()
        else:
            merged.append(seg.copy())
    return merged


def remap_speakers(utterances: list[dict], interviewer_label: str | None = None) -> list[dict]:
    """Remap pyannote speaker labels to interviewer/participant.

    If interviewer_label is None, the first speaker is assigned as interviewer.
    """
    if not utterances:
        return utterances

    if interviewer_label is None:
        interviewer_label = utterances[0]["speaker"]
        logger.info(f"Auto-detected first speaker as interviewer: {interviewer_label}")

    speakers = sorted(set(u["speaker"] for u in utterances))
    if interviewer_label not in speakers:
        logger.warning(
            f"--interviewer-label '{interviewer_label}' not found in diarization output. "
            f"Available labels: {speakers}. Keeping raw labels."
        )
        return utterances

    mapping = {}
    for s in speakers:
        if s == interviewer_label:
            mapping[s] = "interviewer"
        else:
            mapping[s] = "participant"

    return [{**u, "speaker": mapping[u["speaker"]]} for u in utterances]


def transcribe(
    audio_path: Path,
    num_speakers: int = 2,
    interviewer_label: str | None = None,
    model_size: str = "large-v3",
    model_path: Path | None = None,
    language: str = "en",
) -> dict:
    device = get_device()
    logger.info(f"Using device: {device} (pyannote diarization)")
    logger.info("Whisper.cpp uses Metal acceleration automatically on Apple Silicon")

    hf_token = os.environ.get("HF_TOKEN")

    if model_path is None:
        model_path = get_model_path(model_size)

    logger.info("Loading diarization pipeline...")
    diarization_pipeline = load_diarization_pipeline(device, hf_token)

    audio = load_audio(audio_path)
    audio_duration = len(audio) / SAMPLE_RATE
    logger.info(f"Audio loaded: {audio_duration:.1f}s")

    logger.info("Running pyannote diarization...")
    diar_segments = diarize(audio_path, diarization_pipeline, num_speakers)

    logger.info(f"Loading whisper.cpp model: {model_path.name}")
    whisper_model = load_whisper_model(model_path)

    logger.info(f"Transcribing {len(diar_segments)} diarized segments...")
    utterances = []
    for i, seg in enumerate(diar_segments):
        result = transcribe_segment(audio, seg, whisper_model, language)
        if result.get("text"):
            utterances.append(result)
        if (i + 1) % 50 == 0:
            logger.info(f"  Transcribed {i + 1}/{len(diar_segments)} segments")

    utterances.sort(key=lambda u: u["start"])
    utterances = merge_consecutive_segments(utterances)
    utterances = remap_speakers(utterances, interviewer_label)

    full_text = "\n".join(
        f"{u['speaker']}: {u['text']}" for u in utterances
    )

    return {
        "utterances": utterances,
        "text": full_text,
        "metadata": {
            "model": f"whisper.cpp/ggml-{model_size}",
            "diarization_model": "pyannote/speaker-diarization-3.1",
            "num_speakers": num_speakers,
            "device": f"Metal (whisper.cpp), {device} (pyannote)",
            "audio_duration_seconds": round(audio_duration, 2),
            "language": language,
        },
    }


def format_as_text(diarized: dict) -> str:
    lines = []
    utterances = diarized.get("utterances") or diarized.get("segments") or []
    for utt in utterances:
        speaker = utt.get("speaker", "unknown")
        text = utt.get("text", "").strip()
        start = utt.get("start", "")
        end = utt.get("end", "")
        timestamp = f"[{start:.2f}-{end:.2f}]" if isinstance(start, (int, float)) else ""
        lines.append(f"{speaker.upper()} {timestamp}: {text}")
    return "\n".join(lines)


def cli():
    parser = argparse.ArgumentParser(
        description="Transcribe interview audio with whisper.cpp (Metal) + pyannote diarization"
    )
    parser.add_argument("--audio", required=True, help="Path to the interview audio file")
    parser.add_argument("--output", required=True, help="Output directory for transcript files")
    parser.add_argument(
        "--interviewer-label",
        help="Pyannote speaker label to map to 'interviewer' (e.g., SPEAKER_00). "
        "If omitted, the first speaker detected is used.",
    )
    parser.add_argument(
        "--num-speakers", type=int, default=2,
        help="Number of speakers in the audio (default: 2)",
    )
    parser.add_argument(
        "--model-size", default="large-v3",
        choices=sorted(GGML_MODEL_URLS.keys()),
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--model-path",
        help="Path to a pre-downloaded GGML model file (overrides --model-size)",
    )
    parser.add_argument(
        "--language", default="en",
        help="Language code for transcription (default: en)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = None
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

    diarized = transcribe(
        audio_path,
        num_speakers=args.num_speakers,
        interviewer_label=args.interviewer_label,
        model_size=args.model_size,
        model_path=model_path,
        language=args.language,
    )

    json_path = output_dir / "diarized_transcript.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(diarized, f, indent=2, ensure_ascii=False)
    logger.info(f"Diarized JSON saved to: {json_path}")

    txt_path = output_dir / "diarized_transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_as_text(diarized))
    logger.info(f"Formatted transcript saved to: {txt_path}")


if __name__ == "__main__":
    cli()
