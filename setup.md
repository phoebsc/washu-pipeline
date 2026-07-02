# macOS Setup — washu_pipeline

Fresh macOS setup from scratch.

---

## 1. Xcode Command Line Tools

```bash
xcode-select --install
```

Follow the prompt. This provides `git`, `clang`, and other build tools.

---

## 2. Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After install, follow the instructions to add Homebrew to your PATH (printed at the end of the installer).

---

## 3. uv (Python environment manager)

```bash
brew install uv
```

---

## 4. ffmpeg (required by pydub for audio I/O)

```bash
brew install ffmpeg
```

---

## 5. Clone the repo

```bash
git clone <repo-url>
cd washu_pipeline
```

---

## 6. Install Python dependencies

```bash
uv sync
```

This creates a `.venv` and installs all dependencies from `pyproject.toml`.

---

## 7. Configure environment

```bash
cp .env.example .env
```

Open `.env` and add your HuggingFace token:

```
HF_TOKEN=hf_...
```

The token is required for the `pyannote/speaker-diarization-3.1` gated model and the `openai/privacy-filter` de-identification model. Both are downloaded automatically on first run.

---

## 8. Ollama (for stitched audio split detection)

Required only for processing stitched recordings where the split point is unknown.

### Install Ollama

```bash
brew install ollama
```

### Start Ollama server

```bash
ollama serve
```

This runs in background on port 11434. You can also launch the Ollama desktop app which starts the server automatically.

### Pull Gemma 4 31B model

```bash
ollama pull gemma4:31b
```

This downloads ~20 GB. The model requires approximately 22 GB RAM at runtime.

### Verify Ollama

```bash
ollama run gemma4:31b "Hello, respond with just OK"
```

---

## 9. Verify

```bash
uv run washu-run-stitched --help
uv run washu-run-dyads --help
```

---

## Notes

- All processing runs locally after initial model downloads (no API calls at runtime).
- First run will download ~3–4 GB of model weights to your HuggingFace cache (`~/.cache/huggingface/`).
- Apple Silicon (M1/M2/M3) is supported — the pipeline uses Metal (MPS) for transcription and CPU for de-identification.
- The stitched workflow (`washu-run-stitched`) additionally requires Ollama + Gemma 4 31B for split-point detection.
