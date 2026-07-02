# AGENTS.md - washu_pipeline

> Local transcription + de-identification pipeline for WashU CDR interview recordings.

---

## Overview

Processes historical CDR interview recordings from WashU. Audio files are typically stitched (partner interview first, then subject interview in a single .mp3). The pipeline transcribes full audio first (whisper.cpp), uses a local LLM (Gemma 4 31B via Ollama) to detect the split point, diarizes each half (pyannote), aligns speaker labels to transcription by timestamp overlap, then de-identifies using a local privacy-filter model.

**Repo:** `git.ucsf.edu/Neurology-Pinheiro-Chagas-Lab/washu-pipeline`
**Jira:** VCDR-420

---

## Pipeline

Primary stitched workflow:

```text
.mp3 (stitched: partner interview + subject interview)
    -> transcribe.py — transcribe full audio (whisper.cpp Metal, full context)
timestamped segments (no speaker labels yet)
    -> ollama_split.py — detect split point (Gemma 4 31B via Ollama)
split_timestamp_seconds
    -> transcribe.py — diarize each half (pyannote, num_speakers=2)
    -> align whisper segments to speaker turns by timestamp overlap
partner_transcript.json + subject_transcript.json
    -> deid.py (openai/privacy-filter local model + propagation)
deid/ (coded transcripts + deid_mapping.json)
```

Legacy dyad workflow (pre-split audio):

```text
partner.mp3 + participant.mp3 or subject.mp3
    -> transcribe.py (whisper.cpp full-audio -> pyannote diarize -> align)
*_transcript.json (speaker-diarized, labeled interviewer/participant)
    -> deid.py (openai/privacy-filter local model + propagation)
*_deid.json / *_deid.txt / *_deid_clean.json
```

Legacy timestamp-split workflow:

```text
.mp3 (full interview)
    -> split_audio.py (pydub, split at timestamp)
partner.mp3 + subject.mp3
    -> transcribe.py (whisper.cpp full-audio -> pyannote diarize -> align)
*_transcript.json (speaker-diarized, labeled interviewer/participant)
    -> deid.py (openai/privacy-filter local model + propagation)
deid/ (coded transcripts + deid_mapping.json)
```

---

## Scripts & Entry Points

| Command | Script | Purpose |
|---------|--------|---------|
| `washu-run-stitched` | `run_stitched.py` | **Primary**: stitched recordings, LLM split detection |
| `washu-run-dyads` | `run_dyads.py` | Pre-split dyad folders (legacy) |
| `washu-run` | `run_pipeline.py` | Timestamp-split workflow (legacy) |
| `washu-split` | `split_audio.py` | Split audio at timestamp into partner/subject |
| `washu-transcribe` | `transcribe.py` | Whisper.cpp + pyannote transcription |
| `washu-deid` | `deid.py` | Local model de-identification with coded entities |
| `washu-view` | `viewer.py` | Generate transcript viewing HTML |

---

## Environment

Required in `.env`:

```text
HF_TOKEN=...    # HuggingFace (pyannote gated model + privacy-filter)
```

Required for stitched workflow:
- Ollama running (`ollama serve` or `brew services start ollama`)
- Gemma 4 31B pulled (`ollama pull gemma4:31b`)

No Azure/OpenAI keys needed. Processing is local after initial model downloads.

---

## Codex Rules

1. Run scripts with `uv run` from the project root.
2. Use `uv sync` after dependency or lockfile changes.
3. Treat `.env`, `deid_mapping.json`, audio files, transcripts, and `output/` as sensitive local artifacts unless the user says otherwise.
4. Never commit real `deid_mapping.json` files, PHI, or raw interview audio.
5. Speaker labels are `interviewer`/`participant`, not `agent`/`participant`.
6. Partner and subject transcripts share a single `DeidMapper` for consistent entity codes.
7. Prefer `washu-run-stitched` for stitched recordings (single .mp3 per interview session).
8. Preserve the legacy `washu-run` and `washu-run-dyads` workflows unless explicitly asked to change them.
9. Transcription order: whisper first (full audio context), then pyannote diarization, then align by timestamp overlap.
