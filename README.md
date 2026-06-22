# washu_pipeline

Local transcription and de-identification pipeline for WashU CDR interview recordings.

## Pipeline

```
.mp3 interview audio
    ↓  split at timestamp
partner.mp3 + subject.mp3
    ↓  whisper.cpp + pyannote diarization (fully local)
partner_transcript.json + subject_transcript.json
    ↓  openai/privacy-filter model (local inference)
deid/ (coded transcripts + entity mapping)
```

All processing runs locally after initial model downloads. No API calls.

## Setup

```bash
uv sync
cp .env.example .env  # add HF_TOKEN for pyannote gated model access
```

## Usage

### Full pipeline (single tape)

```bash
uv run washu-run \
    --audio-dir "/path/to/Extracted Audio Files (MP3 256kbps)" \
    --timestamps "/path/to/timestamp_info.txt" \
    --output output/ \
    --tape "Tape_11_Interview_(Source)_1"
```

### Full pipeline (all tapes)

```bash
uv run washu-run \
    --audio-dir "/path/to/Extracted Audio Files (MP3 256kbps)" \
    --timestamps "/path/to/timestamp_info.txt" \
    --output output/
```

### Individual steps

```bash
# Split only
uv run washu-split \
    --audio-dir "/path/to/audio" \
    --timestamps "/path/to/timestamp_info.txt" \
    --output output/split/

# Transcribe only
uv run washu-transcribe \
    --audio path/to/file.mp3 \
    --output output/transcript/

# De-identify only (multiple files share entity codes)
uv run washu-deid \
    --input output/partner_transcript.json output/subject_transcript.json \
    --output output/deid/
```

## Output structure

```
output/<tape_name>/
├── split/
│   ├── <tape>_partner.mp3
│   └── <tape>_subject.mp3
├── partner_transcript.json
├── partner_transcript.txt
├── subject_transcript.json
├── subject_transcript.txt
└── deid/
    ├── partner_deid.json
    ├── partner_deid.txt
    ├── subject_deid.json
    ├── subject_deid.txt
    └── deid_mapping.json    ← code-to-PHI lookup (keep secure)
```

## De-identification

Uses `openai/privacy-filter` (1.5B parameter token classifier) for PHI detection, with cross-transcript propagation for consistency. Detected entities are replaced with unique codes (e.g., `[PERSON_01]`, `[DATE_01]`) that remain consistent across both partner and subject transcripts within the same session.

The `deid_mapping.json` contains the code-to-original-text mapping and should be treated as sensitive data.
