"""Detect the split point between partner and subject interviews using Ollama + Gemma 4.

Given a full-audio transcript (from whisper), identifies where the interviewer
transitions from speaking with the study partner to speaking with the subject.
Uses a local LLM via Ollama (no cloud API calls).
"""

import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "gemma4:27b"
OLLAMA_URL = "http://localhost:11434/api/generate"

SPLIT_DETECTION_PROMPT = """\
You are analyzing a transcript of a clinical interview recording.
The recording contains TWO consecutive interviews:
1. First: an interview with the STUDY PARTNER (typically a spouse or family member)
2. Second: an interview with the SUBJECT (the patient themselves)

The interviewer conducts both interviews back-to-back in a single recording.

Your task: Find the exact point where the interviewer transitions from the partner \
interview to the subject interview. This typically happens when:
- The interviewer thanks or dismisses the partner
- The interviewer greets or welcomes the subject
- There is a clear shift in who is being addressed
- The conversation resets to introductory questions after a period of substantive interview

Each line in the transcript below has the format: [start_seconds-end_seconds]: text
Speaker labels are NOT available — you must detect the transition based on CONTENT.

Respond with ONLY a JSON object in this exact format:
{"split_timestamp_seconds": <float>, "confidence": "<high|medium|low>", "reasoning": "<brief explanation>"}

The split_timestamp_seconds should be the START timestamp of the first utterance \
that belongs to the subject interview.

TRANSCRIPT:
"""


def check_ollama_available(model: str = OLLAMA_MODEL) -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            available_models = [m["name"] for m in data.get("models", [])]
            # Match with or without :latest tag
            for m in available_models:
                if m == model or m.startswith(f"{model}:") or model.startswith(f"{m.split(':')[0]}"):
                    return True
            logger.warning(
                f"Ollama is running but model '{model}' not found. "
                f"Available: {available_models}. Run: ollama pull {model}"
            )
            return False
    except (urllib.error.URLError, OSError):
        return False


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call Ollama generate API. Returns the response text."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_ctx": 32768,
        },
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    logger.info(f"Calling Ollama ({model}) for split-point detection...")
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read())
        return result["response"]


def _format_transcript_for_llm(whisper_segments: list[dict]) -> str:
    """Format whisper segments as timestamped lines for the LLM."""
    lines = []
    for seg in whisper_segments:
        lines.append(f"[{seg['start']:.1f}-{seg['end']:.1f}]: {seg['text']}")
    return "\n".join(lines)


def detect_split_point(
    whisper_segments: list[dict],
    audio_duration_seconds: float,
    model: str = OLLAMA_MODEL,
) -> dict:
    """Detect the partner/subject split point using a local LLM.

    Args:
        whisper_segments: List of {"start": float, "end": float, "text": str}
        audio_duration_seconds: Total audio duration for validation
        model: Ollama model name

    Returns:
        {"split_timestamp_seconds": float, "confidence": str, "reasoning": str}

    Raises:
        RuntimeError: If Ollama is unavailable or response is unparseable
    """
    if not check_ollama_available(model):
        raise RuntimeError(
            f"Ollama is not running or model '{model}' is not available. "
            f"Start Ollama with 'ollama serve' and pull the model with 'ollama pull {model}'"
        )

    transcript_text = _format_transcript_for_llm(whisper_segments)
    prompt = SPLIT_DETECTION_PROMPT + transcript_text

    response_text = _call_ollama(prompt, model)

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Ollama returned unparseable response. Raw output:\n{response_text[:500]}"
        )

    required_keys = {"split_timestamp_seconds", "confidence", "reasoning"}
    if not required_keys.issubset(result.keys()):
        raise RuntimeError(
            f"Ollama response missing required keys. Got: {list(result.keys())}"
        )

    split_time = float(result["split_timestamp_seconds"])

    min_bound = audio_duration_seconds * 0.10
    max_bound = audio_duration_seconds * 0.90
    if not (min_bound <= split_time <= max_bound):
        logger.warning(
            f"Split point {split_time:.1f}s is outside expected range "
            f"[{min_bound:.0f}s, {max_bound:.0f}s] for {audio_duration_seconds:.0f}s audio. "
            f"Proceeding but flagging as suspicious."
        )

    if result.get("confidence") == "low":
        logger.warning(
            f"LLM reported low confidence in split detection: {result['reasoning']}"
        )

    logger.info(
        f"Split detected at {split_time:.1f}s "
        f"(confidence: {result['confidence']}, reason: {result['reasoning']})"
    )

    return {
        "split_timestamp_seconds": split_time,
        "confidence": result["confidence"],
        "reasoning": result["reasoning"],
    }
