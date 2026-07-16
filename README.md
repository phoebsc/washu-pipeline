# washu_pipeline

Local transcription and de-identification pipeline for WashU CDR interview recordings.

---

## Dyad folder workflow (primary)

Use this when you have a folder of dyads, each containing pre-split `partner.mp3` and `participant.mp3` files. The subject interview may also be named `subject.mp3`.

### Input structure

```
<folder>/
├── <dyad_id>/
│   ├── partner.mp3
│   └── participant.mp3  # or subject.mp3
├── <dyad_id>/
│   ├── partner.mp3
│   └── participant.mp3  # or subject.mp3
└── ...
```

### Run

```bash
uv run washu-run-dyads --input /path/to/folder
```

Process a single dyad:

```bash
uv run washu-run-dyads --input /path/to/folder --dyad <dyad_id>
```

Skip dyads that are already fully processed:

```bash
uv run washu-run-dyads --input /path/to/folder --skip-existing
```

### Output structure

Most output lands **inside each dyad subfolder**:

```
<folder>/<dyad_id>/
├── partner_transcript.json
├── partner_transcript.txt
├── subject_transcript.json
├── subject_transcript.txt
├── partner_deid.json
├── partner_deid.txt
├── subject_deid.json
├── subject_deid.txt
├── deid_mapping.json       ← code-to-PHI lookup (keep secure)
└── view.html
```

Four files are written to a **separate output directory** (project root `output/` by default):

```
output/<dyad_id>/
├── partner_deid.txt        ← plain-text redacted transcript
├── subject_deid.txt        ← plain-text redacted transcript
├── partner_deid_clean.json ← deid JSON with entities stripped
└── subject_deid_clean.json ← deid JSON with entities stripped
```

Override the output location:

```bash
uv run washu-run-dyads --input /path/to/folder --output /path/to/output
```

---

## Pipeline

```
partner.mp3 + participant.mp3 (pre-split)
    ↓  transcribe.py (whisper.cpp Metal + pyannote diarization)
*_transcript.json (speaker-diarized, labeled interviewer/participant)
    ↓  deid.py (openai/privacy-filter local model + propagation)
*_deid.json / *_deid.txt / *_deid_clean.json
```

All processing runs locally after initial model downloads. No API calls at runtime.

---

## Setup

See [setup.md](setup.md) for full macOS installation instructions from scratch.

Quick start (assumes `uv` and `ffmpeg` are already installed):

```bash
uv sync
cp .env.example .env  # add HF_TOKEN
```

---

## De-identification

Uses `openai/privacy-filter` (1.5B parameter token classifier) for PHI detection, with cross-transcript propagation for consistency. Detected entities are replaced with unique codes (e.g., `[PERSON_01]`, `[DATE_01]`) that remain consistent across both partner and subject transcripts within the same session.

The `deid_mapping.json` contains the code-to-original-text mapping and should be treated as sensitive data. The `*_deid_clean.json` files are the same transcripts with the per-utterance `entities` list removed — safe to share without the mapping.

### Review and edit de-identification

Open the Desktop launcher:

```bash
~/Desktop/Open\ WashU\ Deid\ Editor.command
```

Or run it from the project folder:

```bash
uv run washu-deid-editor --output output
```

The editor opens a local webpage with a dyad dropdown. Existing entities can be removed by clicking a highlighted span, and new entities can be added by selecting or typing exact text and choosing an entity type. Saving rewrites the edited files in `output/<dyad_id>/deid/` and refreshes `output/<dyad_id>/view.html`.

---

## Legacy: timestamp-split workflow

Use this for older recordings where partner and subject audio are combined into a single file and a timestamp file marks the split point.

### Run

```bash
uv run washu-run \
    --audio-dir "/path/to/Extracted Audio Files (MP3 256kbps)" \
    --timestamps "/path/to/timestamp_info.txt" \
    --output output/

# Single tape
uv run washu-run \
    --audio-dir "/path/to/audio" \
    --timestamps "/path/to/timestamp_info.txt" \
    --output output/ \
    --tape "Tape_11_Interview_(Source)_1"
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

### Output structure

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
    └── deid_mapping.json
```
