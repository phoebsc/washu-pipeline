# AGENTS.md — washu_pipeline

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
    ↓  transcribe.py — transcribe full audio (whisper.cpp Metal, large-v3-turbo + VAD chunks)
timestamped segments (no speaker labels yet)
    ↓  ollama_split.py — detect split point (Gemma 4 31B via Ollama)
split_timestamp_seconds
    ↓  transcribe.py — diarize each half (pyannote, num_speakers=2)
    ↓  align whisper segments to speaker turns by timestamp overlap (no merge)
partner_transcript.json + subject_transcript.json
    ↓  deid.py (openai/privacy-filter local model + propagation)
deid/ (coded transcripts + deid_mapping.json)
```

Recommended stitched command:
```bash
uv run washu-run-stitched \
  --input /Users/fadchen/Desktop/interview_data \
  --output /Users/fadchen/Desktop/washu_pipeline/output \
  --model-size large-v3-turbo \
  --vad-chunked-transcription
```

VAD chunking detects the non-silent parts of the recording and transcribes them as short chunks instead of asking Whisper to process the entire tape as one long context. This keeps timestamps in the original audio timeline, but reduces the chance that quiet/noisy regions make Whisper hallucinate repeated phrases such as “The End” or looping filler.

The stitched workflow does not merge consecutive same-speaker Whisper segments after diarization. This preserves short Q/A turns and avoids turning diarization mistakes into large single-speaker blocks. Use `--skip-existing` only when you intentionally want to leave already generated outputs untouched.

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
- Gemma 4 31B pulled (`ollama pull gemma4:31b`)

Ollama performance tuning (set in the Homebrew plist or systemd unit):
```bash
# macOS: edit ~/Library/LaunchAgents/homebrew.mxcl.ollama.plist EnvironmentVariables
OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_FLASH_ATTENTION=1

# Linux (add to ~/.bashrc or systemd unit)
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_FLASH_ATTENTION=1
```
The 31B model (19GB weights) does NOT fit fully on GPU on machines with <24GB available working set. It runs with ~27%/73% CPU/GPU split, which is slow (~22 min per split detection) but produces correct results. The pipeline timeout is set to 30 minutes to accommodate this. Do NOT downgrade to the smaller model — it produces incorrect split points.

No Azure/OpenAI keys needed — pipeline is fully local after model downloads.

---

## Codex Rules

1. All scripts run with `uv run` from the project root
2. The `deid_mapping.json` is sensitive — never commit it with real data
3. Audio files and output/ are gitignored
4. Speaker labels are interviewer/participant (not agent/participant)
5. Partner and subject transcripts share a single DeidMapper for consistent entity codes
6. Stitched transcription should use `large-v3-turbo` with `--vad-chunked-transcription`
7. Stitched transcripts should preserve no-merge utterances after speaker alignment
8. Transcription order: whisper first (VAD-chunked full-audio timeline), then split detection, then pyannote diarization, then align
