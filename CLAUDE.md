# CLAUDE.md — washu_pipeline

> Local transcription + de-identification pipeline for WashU CDR interview recordings.

---

## Overview

Processes historical CDR interview recordings from WashU. Audio files contain two back-to-back interviews: study partner first, then subject. The pipeline transcribes full audio first (whisper.cpp), then uses pyannote for speaker diarization, aligning speaker labels to the transcription by timestamp overlap. For stitched recordings without a known split point, a local LLM (Gemma 4 via Ollama) detects the transition between partner and subject interviews.

**Repo:** `git.ucsf.edu/Neurology-Pinheiro-Chagas-Lab/washu-pipeline`
**Jira:** VCDR-420

---

## Pipeline (primary — stitched audio)

```
.mp3 (stitched: partner interview + subject interview)
    ↓  transcribe.py — transcribe full audio (whisper.cpp Metal)
timestamped segments (no speaker labels yet)
    ↓  ollama_split.py — detect split point (Gemma 4 27B via Ollama)
split_timestamp_seconds
    ↓  transcribe.py — diarize each half (pyannote, num_speakers=2)
    ↓  align whisper segments to speaker turns by timestamp overlap
partner_transcript.json + subject_transcript.json
    ↓  deid.py (openai/privacy-filter local model + propagation)
deid/ (coded transcripts + deid_mapping.json)
```

## Pipeline (legacy — known split timestamp)

```
.mp3 (full interview)
    ↓  split_audio.py (pydub, split at timestamp)
partner.mp3 + subject.mp3
    ↓  transcribe.py (whisper.cpp full-audio → pyannote diarize → align)
*_transcript.json (speaker-diarized, labeled interviewer/participant)
    ↓  deid.py (openai/privacy-filter local model + propagation)
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

---

## Environment

Required in `.env`:
```
HF_TOKEN=...    # HuggingFace (pyannote gated model + privacy-filter)
```

Required for stitched workflow:
- Ollama running (`ollama serve`)
- Gemma 4 27B pulled (`ollama pull gemma4:27b`)

No Azure/OpenAI keys needed — pipeline is fully local after model downloads.

---

## Rules

1. All scripts run with `uv run` from the project root
2. The `deid_mapping.json` is sensitive — never commit it with real data
3. Audio files and output/ are gitignored
4. Speaker labels are interviewer/participant (not agent/participant)
5. Partner and subject transcripts share a single DeidMapper for consistent entity codes
6. Transcription order: whisper first (full audio context), then pyannote diarization, then align
