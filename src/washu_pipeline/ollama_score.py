"""Score subject transcripts using Ollama + Gemma 4 for clinical CDR items.

Scores three items from the subject interview:
1. Memory registration — immediate repetition of a name+address
2. Memory recall — delayed recall of the same name+address
3. Date orientation — year, month, day correctness

Runs independently of de-identification. Uses the same local Ollama instance
as split detection (Gemma 4 31B).
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

from .ollama_split import call_ollama, check_ollama_available, OLLAMA_MODEL

logger = logging.getLogger(__name__)


# --- Pydantic models for LLM output validation ---

class MemoryScores(BaseModel):
    target_phrase: str
    registration: list[int]
    recall: list[int]

    @field_validator("registration", "recall")
    @classmethod
    def validate_binary(cls, v: list[int]) -> list[int]:
        if not all(x in (0, 1) for x in v):
            raise ValueError("All values must be 0 or 1")
        return v


OrientationValue = Literal["correct", "incorrect", "not_given", "ground_truth_missing"]


class OrientationScores(BaseModel):
    year: OrientationValue
    month: OrientationValue
    day: OrientationValue


# --- Prompts ---

MEMORY_PROMPT = """\
You are analyzing a transcript of a clinical interview with a subject (participant).

Your task: Score the subject's performance on a NAME AND ADDRESS memory test.

This test has two parts:
1. REGISTRATION: The interviewer gives the subject a name and address to remember, \
then immediately asks them to repeat it. The subject may attempt to repeat it one or more times.
2. RECALL: Later in the interview, the interviewer asks the subject to recall the \
same name and address from memory.

For each part, score EACH attempt by the subject:
- 1 = the attempt is correct and complete (all components of the name and address are stated correctly)
- 0 = the attempt is incorrect or incomplete (any component missing or wrong)

Components of the target phrase typically include: a person's name, a street number, \
a street name, and a city.

Respond with ONLY this JSON object:
{json_schema}

TRANSCRIPT:
{transcript}

END OF TRANSCRIPT.
"""

ORIENTATION_PROMPT = """\
You are analyzing a transcript of a clinical interview with a subject (participant).

The actual date of this interview is: {interview_date}

Your task: Find where the interviewer asks the subject what today's date is, \
then evaluate the subject's response for each date component.

For each component (year, month, day), determine:
- "correct" — the subject stated this component and it matches the actual interview date
- "incorrect" — the subject stated this component but it does NOT match the actual interview date
- "not_given" — the subject did not provide this component at all
- "ground_truth_missing" — the interview date does not contain enough information to verify this component

The interview date is in MM-DD-YYYY format. The subject may express the date in any \
natural language form (e.g., "December" for month 12, "Monday" is NOT a day-of-month).

Respond with ONLY this JSON object:
{json_schema}

TRANSCRIPT:
{transcript}

END OF TRANSCRIPT.
"""


def _parse_date_header(first_line: str) -> str | None:
    """Extract date from the [date] M-D-YY or M-D-YYYY [date] header line."""
    match = re.match(r"\[date\]\s*(\d{1,2}-\d{1,2}-\d{2,4})\s*\[date\]", first_line.strip())
    if match:
        return match.group(1)
    return None


def score_memory(transcript_text: str, model: str = OLLAMA_MODEL) -> MemoryScores:
    """Score memory registration and recall from the subject transcript."""
    schema_example = json.dumps({
        "target_phrase": "<the name and address the interviewer asked subject to remember>",
        "registration": [0, 1],
        "recall": [0],
    }, indent=2)

    prompt = MEMORY_PROMPT.format(
        json_schema=schema_example,
        transcript=transcript_text,
    )

    response_text = call_ollama(prompt, model)

    try:
        raw = json.loads(response_text)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Ollama returned unparseable JSON for memory scoring. Raw:\n{response_text[:500]}"
        )

    return MemoryScores.model_validate(raw)


def score_orientation(
    transcript_text: str, interview_date: str, model: str = OLLAMA_MODEL
) -> OrientationScores:
    """Score date orientation from the subject transcript."""
    schema_example = json.dumps({
        "year": "correct | incorrect | not_given | ground_truth_missing",
        "month": "correct | incorrect | not_given | ground_truth_missing",
        "day": "correct | incorrect | not_given | ground_truth_missing",
    }, indent=2)

    prompt = ORIENTATION_PROMPT.format(
        interview_date=interview_date,
        json_schema=schema_example,
        transcript=transcript_text,
    )

    response_text = call_ollama(prompt, model)

    try:
        raw = json.loads(response_text)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Ollama returned unparseable JSON for orientation scoring. Raw:\n{response_text[:500]}"
        )

    return OrientationScores.model_validate(raw)


def score_transcript(
    tape_output_dir: Path, model: str = OLLAMA_MODEL
) -> dict:
    """Score a subject transcript for memory and orientation items.

    Reads subject_transcript.txt from the tape output directory.
    Returns the combined scores dict.
    """
    transcript_path = tape_output_dir / "subject_transcript.txt"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Subject transcript not found: {transcript_path}")

    text = transcript_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    interview_date = None
    transcript_body = text
    if lines and _parse_date_header(lines[0]):
        interview_date = _parse_date_header(lines[0])
        transcript_body = "\n".join(lines[1:])

    if not check_ollama_available(model):
        raise RuntimeError(
            f"Ollama is not running or model '{model}' is not available. "
            f"Start Ollama with 'ollama serve' and pull the model with 'ollama pull {model}'"
        )

    # Agent 1: Memory registration + recall
    logger.info("Scoring memory (registration + recall)...")
    memory = score_memory(transcript_body, model)

    # Agent 2: Date orientation
    if interview_date:
        logger.info(f"Scoring orientation (interview date: {interview_date})...")
        orientation = score_orientation(transcript_body, interview_date, model)
    else:
        logger.info("No [date] header found — scoring orientation with unknown ground truth")
        orientation = score_orientation(transcript_body, "UNKNOWN", model)

    return {
        "memory": memory.model_dump(),
        "orientation": orientation.model_dump(),
    }


def cli():
    parser = argparse.ArgumentParser(
        description="Score subject transcripts for CDR clinical items using local LLM"
    )
    parser.add_argument(
        "--input", required=True,
        help="Tape output directory containing subject_transcript.txt",
    )
    parser.add_argument(
        "--ollama-model", default=OLLAMA_MODEL,
        help=f"Ollama model name (default: {OLLAMA_MODEL})",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )

    tape_dir = Path(args.input)
    if not tape_dir.exists():
        raise FileNotFoundError(f"Directory not found: {tape_dir}")

    scores = score_transcript(tape_dir, model=args.ollama_model)

    output_path = tape_dir / "scores.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    logger.info(f"Scores written to {output_path}")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    cli()
