# CLAUDE.md — washu_pipeline

> Local transcription + de-identification pipeline for WashU CDR interview recordings.

---

## Overview

Processes historical CDR interview recordings from WashU. Audio files contain two back-to-back interviews: study partner first, then subject. The pipeline splits at a known timestamp, transcribes each half locally (whisper.cpp + pyannote), then de-identifies using a local privacy-filter model.

**Repo:** `git.ucsf.edu/Neurology-Pinheiro-Chagas-Lab/washu-pipeline`
**Jira:** VCDR-420

---

## Pipeline

```
.mp3 (full interview)
    ↓  split_audio.py (pydub, split at timestamp)
partner.mp3 + subject.mp3
    ↓  transcribe.py (whisper.cpp Metal + pyannote diarization)
*_transcript.json (speaker-diarized, labeled interviewer/participant)
    ↓  deid.py (openai/privacy-filter local model + propagation)
deid/ (coded transcripts + deid_mapping.json)
```

---

## Scripts & Entry Points

| Command | Script | Purpose |
|---------|--------|---------|
| `washu-split` | `split_audio.py` | Split audio at timestamp into partner/subject |
| `washu-transcribe` | `transcribe.py` | Whisper.cpp + pyannote transcription |
| `washu-deid` | `deid.py` | Local model de-identification with coded entities |
| `washu-run` | `run_pipeline.py` | Full pipeline orchestrator |

---

## Environment

Required in `.env`:
```
HF_TOKEN=...    # HuggingFace (pyannote gated model + privacy-filter)
```

No Azure/OpenAI keys needed — pipeline is fully local after model downloads.

---

## Rules

1. All scripts run with `uv run` from the project root
2. The `deid_mapping.json` is sensitive — never commit it with real data
3. Audio files and output/ are gitignored
4. Speaker labels are interviewer/participant (not agent/participant)
5. Partner and subject transcripts share a single DeidMapper for consistent entity codes
