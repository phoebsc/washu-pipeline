"""Standalone step: LLM-based speaker relabeling for transcript .txt files.

Reads partner_deid.txt and subject_deid.txt from an output folder, uses Azure
OpenAI (GPT-5.5) to fix mislabeled utterances and split lines that contain
speech from two speakers. Writes corrected files back in place.

Not included in run_stitched — run independently:
    uv run washu-relabel --folder output/Tape_11_Interview_(Source)_1_12-21-1992
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

ENV_PATH = Path(__file__).resolve().parents[3] / "vCDR_eval_webapp" / ".env"

load_dotenv(ENV_PATH)


def get_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.getenv("VERSA_OPENAI_API_KEY"),
        api_version=os.getenv("VERSA_API_VERSION"),
        azure_endpoint=os.getenv("VERSA_RESOURCE_ENDPOINT"),
    )


MODEL = "gpt-5.5-2026-04-24"

SYSTEM_PROMPT = """\
You are a transcript editor for clinical dementia interviews (CDR format).

Each interview has exactly two speakers:
- "interviewer" — the clinician asking questions
- "participant" — the person being interviewed (either the subject or their study partner)

You will receive a transcript where each line is prefixed with a speaker label. Your job:
1. Fix any lines where the speaker label is WRONG based on conversational context.
2. Split any lines that contain speech from BOTH speakers into separate lines, each with the correct label.

Rules for labeling:
- The interviewer asks questions, probes for details, gives instructions (e.g. "tell me about...", "how often...", "usually, sometimes, or rarely?"), and sometimes gives brief acknowledgments after the participant answers.
- The participant answers questions, provides information about themselves or their family member, and sometimes hesitates before answering.
- Hesitation sounds ("Hmm", "Um", "Uh") belong to whoever is about to speak next — if the next substantive utterance is a participant answer, the hesitation is the participant's.
- Short acknowledgments ("Right", "Mm-hmm", "Okay", "Yeah", "Good") — assign based on who is confirming. After the participant gives an answer, a brief "Mm-hmm" or "Okay" is typically the interviewer acknowledging. After the interviewer asks something, a "Right" or "Yeah" is typically the participant confirming.
- When a line contains a clear speaker transition (e.g., a question followed by an answer on the same line), split it at the transition point. Each resulting line must contain speech from only ONE speaker.

Rules for output:
- Preserve the exact wording. Do NOT paraphrase, summarize, add, or remove words.
- Output ONLY the corrected transcript, one utterance per line, each prefixed with "interviewer: " or "participant: ".
- Do NOT add line numbers, blank lines, headers, or any other formatting.
- If a line is already correctly labeled and contains only one speaker, output it unchanged.
"""


def relabel_transcript(client: AzureOpenAI, transcript_text: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript_text},
        ],
        max_completion_tokens=16000,
    )
    return response.choices[0].message.content.strip()


def process_folder(folder: Path, dry_run: bool = False) -> None:
    client = get_client()

    for filename in ["subject_deid.txt", "partner_deid.txt"]:
        filepath = folder / filename
        if not filepath.exists():
            print(f"  Skipping {filename} (not found)")
            continue

        original = filepath.read_text()
        if not original.strip():
            print(f"  Skipping {filename} (empty)")
            continue

        print(f"  Processing {filename} ({len(original.splitlines())} lines)...")
        corrected = relabel_transcript(client, original)

        out_name = filename.replace(".txt", "_relabeled.txt")
        out_path = folder / out_name

        if dry_run:
            print(f"\n{'='*60}")
            print(f"  {out_name} — CORRECTED OUTPUT:")
            print(f"{'='*60}")
            print(corrected)
            print(f"{'='*60}\n")
        else:
            out_path.write_text(corrected + "\n")
            print(f"  Wrote {len(corrected.splitlines())} lines to {out_path}")


def cli():
    parser = argparse.ArgumentParser(
        description="Re-label speaker utterances in transcript .txt files using LLM"
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Path to output folder containing subject_deid.txt and/or partner_deid.txt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print corrected output without overwriting files",
    )
    args = parser.parse_args()

    if not args.folder.exists():
        print(f"Error: folder not found: {args.folder}")
        raise SystemExit(1)

    print(f"Relabeling transcripts in: {args.folder}")
    process_folder(args.folder, dry_run=args.dry_run)
    print("Done.")


if __name__ == "__main__":
    cli()
