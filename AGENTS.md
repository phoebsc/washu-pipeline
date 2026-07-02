# AGENTS.md - washu_pipeline

> Local transcription + de-identification pipeline for WashU CDR interview recordings.

---

## Overview

Processes historical CDR interview recordings from WashU. Audio files contain either pre-split dyad recordings (`partner.mp3` and `participant.mp3` or `subject.mp3`) or, for the legacy workflow, two back-to-back interviews that are split at a known timestamp. The pipeline transcribes audio locally (whisper.cpp + pyannote), then de-identifies using a local privacy-filter model.

**Repo:** `git.ucsf.edu/Neurology-Pinheiro-Chagas-Lab/washu-pipeline`
**Jira:** VCDR-420

---

## Pipeline

Primary dyad workflow:

```text
partner.mp3 + participant.mp3 or subject.mp3
    -> transcribe.py (whisper.cpp Metal + pyannote diarization)
*_transcript.json (speaker-diarized, labeled interviewer/participant)
    -> deid.py (openai/privacy-filter local model + propagation)
*_deid.json / *_deid.txt / *_deid_clean.json
```

Legacy timestamp-split workflow:

```text
.mp3 (full interview)
    -> split_audio.py (pydub, split at timestamp)
partner.mp3 + subject.mp3
    -> transcribe.py (whisper.cpp Metal + pyannote diarization)
*_transcript.json (speaker-diarized, labeled interviewer/participant)
    -> deid.py (openai/privacy-filter local model + propagation)
deid/ (coded transcripts + deid_mapping.json)
```

---

## Scripts & Entry Points

| Command | Script | Purpose |
|---------|--------|---------|
| `washu-run-dyads` | `run_dyads.py` | Primary dyad folder pipeline for pre-split partner/participant audio |
| `washu-split` | `split_audio.py` | Split legacy audio at timestamp into partner/subject |
| `washu-transcribe` | `transcribe.py` | Whisper.cpp + pyannote transcription |
| `washu-deid` | `deid.py` | Local model de-identification with coded entities |
| `washu-view` | `viewer.py` | Generate transcript viewing HTML |
| `washu-run` | `run_pipeline.py` | Legacy full pipeline orchestrator |

---

## Environment

Required in `.env`:

```text
HF_TOKEN=...    # HuggingFace (pyannote gated model + privacy-filter)
```

No Azure/OpenAI keys needed. Processing is local after initial model downloads.

---

## Codex Rules

1. Run scripts with `uv run` from the project root.
2. Use `uv sync` after dependency or lockfile changes.
3. Treat `.env`, `deid_mapping.json`, audio files, transcripts, and `output/` as sensitive local artifacts unless the user says otherwise.
4. Never commit real `deid_mapping.json` files, PHI, or raw interview audio.
5. Speaker labels are `interviewer`/`participant`, not `agent`/`participant`.
6. Partner and subject transcripts share a single `DeidMapper` for consistent entity codes.
7. Prefer the primary `washu-run-dyads` workflow for folders with pre-split `partner.mp3` and `participant.mp3` or `subject.mp3`.
8. Preserve the legacy `washu-run` timestamp-split workflow unless explicitly asked to change it.
