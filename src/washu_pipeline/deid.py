"""De-identification pipeline using local privacy-filter model.

Runs the openai/privacy-filter token classifier (1.5B params, local inference)
to detect PHI entities, then replaces them with unique coded identifiers that
are consistent across the entire session (partner + subject transcripts).

No network calls after initial model download.

Produces:
  <output_dir>/deid_transcript.json   — coded transcript + entity mapping
  <output_dir>/deid_transcript.txt    — human-readable coded version
  <output_dir>/deid_mapping.json      — PHI word → code lookup table
"""

import argparse
import json
import logging
import os
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from transformers import pipeline as hf_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


class DeidMapper:
    """Assign and track de-id codes for PHI entities within a session."""

    def __init__(self):
        self.word_to_code: dict[str, str] = {}
        self.counters: dict[str, int] = {}

    def get_code(self, word: str, entity_group: str) -> str:
        """Get or assign a de-id code for a PHI word.

        Same word always gets the same code within the session.
        """
        normalized = word.strip().lower()
        if normalized in self.word_to_code:
            return self.word_to_code[normalized]

        code_prefix = entity_group.replace("private_", "")

        if code_prefix not in self.counters:
            self.counters[code_prefix] = 1
        else:
            self.counters[code_prefix] += 1

        code = f"{code_prefix}_{self.counters[code_prefix]:02d}"
        self.word_to_code[normalized] = code
        # Also store the original case version for the mapping output
        self.word_to_code[f"__original__{normalized}"] = word
        return code

    def get_mapping(self) -> dict[str, str]:
        """Return the code-to-word mapping (reversed for documentation)."""
        mapping = {}
        for key, code in self.word_to_code.items():
            if key.startswith("__original__"):
                continue
            original = self.word_to_code.get(f"__original__{key}", key)
            mapping[code] = original
        return mapping


@dataclass
class TurnSpan:
    turn_idx: int
    start_in_prose: int
    end_in_prose: int


def load_classifier():
    """Load the openai/privacy-filter token classification model on CPU.

    CPU is used to avoid MPS memory contention with whisper.cpp/pyannote
    when running the full pipeline. The model is fast enough on CPU (~10s).
    """
    logger.info("Loading privacy-filter model (CPU)...")
    classifier = hf_pipeline(
        task="token-classification",
        model="openai/privacy-filter",
        token=os.getenv("HF_TOKEN"),
        aggregation_strategy="simple",
        device="cpu",
    )
    logger.info("Model loaded.")
    return classifier


def _build_prose(turns: list[dict]) -> tuple[str, list[TurnSpan]]:
    """Join turn texts into a single prose string for model inference."""
    parts = []
    turn_index = []
    pos = 0

    for turn_idx, turn in enumerate(turns):
        text = turn["text"]
        if not text:
            continue
        if pos > 0:
            parts.append(" ")
            pos += 1
        turn_index.append(TurnSpan(turn_idx, pos, pos + len(text)))
        parts.append(text)
        pos += len(text)

    return "".join(parts), turn_index


def _merge_adjacent(entities: list[dict], prose: str) -> list[dict]:
    """Merge adjacent same-type entities and re-slice word from prose."""
    if not entities:
        return []
    sorted_ents = sorted(entities, key=lambda e: e["start"])
    merged = [sorted_ents[0].copy()]
    for ent in sorted_ents[1:]:
        prev = merged[-1]
        if ent["entity_group"] == prev["entity_group"] and ent["start"] - prev["end"] <= 1:
            prev["end"] = ent["end"]
            prev["score"] = min(prev["score"], ent["score"])
        else:
            merged.append(ent.copy())
    for ent in merged:
        ent["word"] = prose[ent["start"]:ent["end"]]
    return merged


def _trim_span(prose: str, start: int, end: int) -> tuple[int, int, str]:
    """Trim leading/trailing whitespace from a span."""
    word = prose[start:end]
    stripped = word.strip()
    if not stripped:
        return start, end, word
    left_offset = word.index(stripped[0])
    return start + left_offset, start + left_offset + len(stripped), stripped


def run_model_pass(classifier, turns: list[dict]) -> dict[int, list[dict]]:
    """Run privacy-filter model on turns as a single prose string.

    Returns: dict of turn_idx -> list of entities with relative offsets.
    """
    prose, turn_index = _build_prose(turns)
    if not prose:
        return {}

    raw_entities = classifier(prose)

    entities = [
        {
            "entity_group": e["entity_group"],
            "start": e["start"],
            "end": e["end"],
            "score": float(e["score"]),
        }
        for e in raw_entities
    ]

    entities = _merge_adjacent(entities, prose)

    turn_entities: dict[int, list[dict]] = {}

    for ent in entities:
        start, end, word = _trim_span(prose, ent["start"], ent["end"])
        if not word:
            continue

        turn_span = None
        for ts in turn_index:
            if start >= ts.start_in_prose and end <= ts.end_in_prose:
                turn_span = ts
                break

        if turn_span is None:
            continue

        rel_start = start - turn_span.start_in_prose
        rel_end = end - turn_span.start_in_prose

        if turn_span.turn_idx not in turn_entities:
            turn_entities[turn_span.turn_idx] = []
        turn_entities[turn_span.turn_idx].append({
            "entity_group": ent["entity_group"],
            "word": word,
            "start": rel_start,
            "end": rel_end,
            "score": ent["score"],
        })

    return turn_entities


def propagate_entities(turns: list[dict], detected: dict[int, list[dict]]) -> dict[int, list[dict]]:
    """Propagate detected PHI across all turns using word-boundary regex.

    Ensures consistent detection even when the model misses repeated occurrences.
    """
    phi_surfaces: dict[str, str] = {}
    for ents in detected.values():
        for e in ents:
            word = e["word"].strip()
            if word and len(word) > 1:
                phi_surfaces[word] = e["entity_group"]

    propagated: dict[int, list[dict]] = {}

    for turn_idx, turn in enumerate(turns):
        text = turn["text"]
        if not text:
            continue

        turn_ents = []
        for word, entity_group in phi_surfaces.items():
            pattern = re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)")
            for match in pattern.finditer(text):
                turn_ents.append({
                    "entity_group": entity_group,
                    "word": word,
                    "start": match.start(),
                    "end": match.end(),
                    "score": 1.0,
                })

        if turn_ents:
            propagated[turn_idx] = turn_ents

    return propagated


def merge_entity_lists(a: list[dict], b: list[dict]) -> list[dict]:
    """Merge two entity lists, resolving overlaps by keeping longer spans."""
    if not b:
        return a
    if not a:
        return b

    all_ents = sorted(a + b, key=lambda e: e["start"])
    merged = [all_ents[0]]

    for ent in all_ents[1:]:
        prev = merged[-1]
        if ent["start"] < prev["end"]:
            if ent["end"] > prev["end"]:
                prev["end"] = ent["end"]
                prev["word"] = prev.get("word", "") + ent.get("word", "")
        else:
            merged.append(ent)

    return merged


def redact_text(text: str, entities: list[dict], mapper: DeidMapper) -> str:
    """Replace detected PII spans with coded identifiers."""
    redacted = text
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        code = mapper.get_code(ent["word"], ent["entity_group"])
        redacted = redacted[: ent["start"]] + f"[{code.upper()}]" + redacted[ent["end"]:]
    return redacted


def deid_session(
    transcripts: list[dict],
    classifier,
    mapper: DeidMapper,
) -> list[dict]:
    """De-identify multiple transcripts as a single session.

    Detects entities in each transcript independently, then propagates ALL
    detected entities across ALL transcripts before redacting. This ensures
    that the same person/date/address gets the same code whether it appears
    in the partner interview, subject interview, or both.

    Args:
        transcripts: list of dicts with "utterances" key
        classifier: loaded HuggingFace pipeline
        mapper: shared DeidMapper for consistent codes

    Returns:
        List of de-identified transcript dicts (same order as input).
    """
    # Step 1: Run model pass on each transcript independently
    all_turns: list[list[dict]] = []
    all_detected: list[dict[int, list[dict]]] = []

    for transcript in transcripts:
        utterances = transcript.get("utterances", [])
        turns = [{"speaker": u["speaker"], "text": u["text"]} for u in utterances]
        all_turns.append(turns)

        logger.info(f"Running model pass on {len(turns)} turns...")
        detected = run_model_pass(classifier, turns)
        logger.info(f"Model found entities in {len(detected)} turns")
        all_detected.append(detected)

    # Step 2: Collect ALL detected PHI surfaces across all transcripts
    phi_surfaces: dict[str, str] = {}
    for detected in all_detected:
        for ents in detected.values():
            for e in ents:
                word = e["word"].strip()
                if word and len(word) > 1:
                    phi_surfaces[word] = e["entity_group"]

    logger.info(f"Unique PHI entities across session: {len(phi_surfaces)}")

    # Step 3: Propagate ALL entities across ALL transcripts
    all_propagated: list[dict[int, list[dict]]] = []
    for turns in all_turns:
        propagated: dict[int, list[dict]] = {}
        for turn_idx, turn in enumerate(turns):
            text = turn["text"]
            if not text:
                continue
            turn_ents = []
            for word, entity_group in phi_surfaces.items():
                pattern = re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)")
                for match in pattern.finditer(text):
                    turn_ents.append({
                        "entity_group": entity_group,
                        "word": word,
                        "start": match.start(),
                        "end": match.end(),
                        "score": 1.0,
                    })
            if turn_ents:
                propagated[turn_idx] = turn_ents
        all_propagated.append(propagated)

    # Step 4: Merge and redact each transcript
    results = []
    for idx, transcript in enumerate(transcripts):
        utterances = transcript.get("utterances", [])
        detected = all_detected[idx]
        propagated = all_propagated[idx]

        deid_utterances = []
        total_entities = 0
        for i, utt in enumerate(utterances):
            model_ents = detected.get(i, [])
            prop_ents = propagated.get(i, [])
            merged = merge_entity_lists(model_ents, prop_ents)
            total_entities += len(merged)

            redacted = redact_text(utt["text"], merged, mapper) if merged else utt["text"]
            deid_utterances.append({
                "speaker": utt["speaker"],
                "text": redacted,
                "start": utt.get("start"),
                "end": utt.get("end"),
                "entities": merged,
            })

        logger.info(f"Transcript {idx + 1}: {total_entities} entities redacted")

        results.append({
            "utterances": deid_utterances,
            "text": "\n".join(f"{u['speaker']}: {u['text']}" for u in deid_utterances),
            "metadata": transcript.get("metadata", {}),
        })

    return results


def cli():
    parser = argparse.ArgumentParser(
        description="De-identify transcripts using local privacy-filter model"
    )
    parser.add_argument(
        "--input", required=True, nargs="+",
        help="Path(s) to diarized_transcript.json file(s). "
        "Multiple files share a single DeidMapper for consistent coding.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for de-identified files",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    classifier = load_classifier()
    mapper = DeidMapper()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = []
    transcripts = []
    for p in args.input:
        p = Path(p)
        if not p.exists():
            logger.warning(f"File not found: {p}, skipping")
            continue
        input_paths.append(p)
        with open(p) as f:
            transcripts.append(json.load(f))

    results = deid_session(transcripts, classifier, mapper)

    for input_path, result in zip(input_paths, results):
        stem = input_path.stem.replace("_transcript", "_deid")
        json_path = output_dir / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"De-identified JSON: {json_path}")

        txt_path = output_dir / f"{stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        logger.info(f"De-identified TXT: {txt_path}")

    mapping_path = output_dir / "deid_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapper.get_mapping(), f, indent=2, ensure_ascii=False)
    logger.info(f"De-id mapping saved: {mapping_path}")


if __name__ == "__main__":
    cli()
