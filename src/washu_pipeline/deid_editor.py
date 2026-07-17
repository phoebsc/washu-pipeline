"""Local web editor for reviewing and correcting de-id entities."""

import argparse
import json
import logging
import re
import shutil
import socket
import webbrowser
from copy import deepcopy
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .viewer import generate_html

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)

ENTITY_TYPES = (
    "person",
    "location",
    "date",
    "address",
    "phone",
    "email",
    "organization",
    "other",
)


def _entity_group(label: str) -> str:
    label = label.strip().lower().replace("private_", "")
    if label == "loc":
        label = "location"
    if label not in ENTITY_TYPES:
        label = "other"
    return f"private_{label}"


def _entity_label(entity_group: str) -> str:
    return entity_group.replace("private_", "").lower()


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _dyad_is_editable(path: Path) -> bool:
    deid_dir = path / "deid"
    required = (
        path / "partner_transcript.json",
        path / "subject_transcript.json",
        deid_dir / "partner_deid.json",
        deid_dir / "subject_deid.json",
    )
    return path.is_dir() and all(p.exists() for p in required)


def _safe_dyad_dir(output_root: Path, dyad_id: str) -> Path:
    decoded = unquote(dyad_id)
    if "/" in decoded or decoded in ("", ".", ".."):
        raise ValueError("Invalid dyad id")
    dyad_dir = output_root / decoded
    if not _dyad_is_editable(dyad_dir):
        raise FileNotFoundError(decoded)
    return dyad_dir


def _clean_entities(entities: list[dict], text: str) -> list[dict]:
    clean = []
    for ent in entities:
        try:
            start = int(ent["start"])
            end = int(ent["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        group = _entity_group(str(ent.get("entity_group", ent.get("type", "other"))))
        clean.append({
            "entity_group": group,
            "word": text[start:end],
            "start": start,
            "end": end,
            "score": float(ent.get("score", 1.0)),
        })

    clean.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    merged = []
    for ent in clean:
        if merged and ent["start"] < merged[-1]["end"]:
            prev = merged[-1]
            if (ent["end"] - ent["start"]) > (prev["end"] - prev["start"]):
                merged[-1] = ent
            continue
        merged.append(ent)
    return merged


def _code_for(
    word: str,
    entity_group: str,
    surface_to_code: dict[tuple[str, str], str],
    code_to_original: dict[str, str],
    counters: dict[str, int],
) -> str:
    label = _entity_label(entity_group)
    key = (label, word.strip().lower())
    if key in surface_to_code:
        return surface_to_code[key]
    counters[label] = counters.get(label, 0) + 1
    code = f"{label}_{counters[label]:02d}".upper()
    surface_to_code[key] = code
    code_to_original[code] = word
    return code


def _redact_text(
    text: str,
    entities: list[dict],
    surface_to_code: dict[tuple[str, str], str],
    code_to_original: dict[str, str],
    counters: dict[str, int],
) -> str:
    redacted = text
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        code = _code_for(
            ent["word"],
            ent["entity_group"],
            surface_to_code,
            code_to_original,
            counters,
        )
        redacted = redacted[: ent["start"]] + f"[{code}]" + redacted[ent["end"]:]
    return redacted


def _rebuild_deid(
    original: dict,
    edited: dict,
    surface_to_code: dict[tuple[str, str], str],
    code_to_original: dict[str, str],
    counters: dict[str, int],
) -> dict:
    edited_by_index = edited.get("utterances", [])
    rebuilt_utterances = []
    for idx, orig_utt in enumerate(original.get("utterances", [])):
        text = orig_utt.get("text", "")
        edited_utt = edited_by_index[idx] if idx < len(edited_by_index) else {}
        entities = _clean_entities(edited_utt.get("entities", []), text)
        rebuilt_utterances.append({
            "speaker": orig_utt.get("speaker", edited_utt.get("speaker", "")),
            "text": _redact_text(text, entities, surface_to_code, code_to_original, counters),
            "start": orig_utt.get("start"),
            "end": orig_utt.get("end"),
            "entities": entities,
        })
    return {
        "utterances": rebuilt_utterances,
        "text": "\n".join(f"{u['speaker']}: {u['text']}" for u in rebuilt_utterances),
        "metadata": original.get("metadata", edited.get("metadata", {})),
    }


def _mapping_from_codes(code_to_original: dict[str, str]) -> dict[str, str]:
    return dict(sorted(code_to_original.items()))


def list_dyads(output_root: Path) -> list[str]:
    return sorted(p.name for p in output_root.iterdir() if _dyad_is_editable(p))


def load_dyad(output_root: Path, dyad_id: str) -> dict:
    dyad_dir = _safe_dyad_dir(output_root, dyad_id)
    deid_dir = dyad_dir / "deid"
    mapping_path = deid_dir / "deid_mapping.json"
    return {
        "id": dyad_dir.name,
        "entityTypes": ENTITY_TYPES,
        "original": {
            "partner": _load_json(dyad_dir / "partner_transcript.json"),
            "subject": _load_json(dyad_dir / "subject_transcript.json"),
        },
        "deid": {
            "partner": _load_json(deid_dir / "partner_deid.json"),
            "subject": _load_json(deid_dir / "subject_deid.json"),
        },
        "mapping": _load_json(mapping_path) if mapping_path.exists() else {},
    }


def save_dyad(output_root: Path, payload: dict, deid_output_root: Path | None = None) -> dict:
    dyad_dir = _safe_dyad_dir(output_root, str(payload.get("id", "")))
    deid_dir = dyad_dir / "deid"

    for filename in ("partner_deid.json", "subject_deid.json", "deid_mapping.json"):
        path = deid_dir / filename
        backup = deid_dir / f"{filename}.before_editor"
        if path.exists() and not backup.exists():
            shutil.copy2(path, backup)

    original_partner = _load_json(dyad_dir / "partner_transcript.json")
    original_subject = _load_json(dyad_dir / "subject_transcript.json")
    edited = payload.get("deid", {})

    surface_to_code: dict[tuple[str, str], str] = {}
    code_to_original: dict[str, str] = {}
    counters: dict[str, int] = {}
    partner_deid = _rebuild_deid(
        original_partner,
        edited.get("partner", {}),
        surface_to_code,
        code_to_original,
        counters,
    )
    subject_deid = _rebuild_deid(
        original_subject,
        edited.get("subject", {}),
        surface_to_code,
        code_to_original,
        counters,
    )
    mapping = _mapping_from_codes(code_to_original)

    _write_json(deid_dir / "partner_deid.json", partner_deid)
    _write_json(deid_dir / "subject_deid.json", subject_deid)
    _write_json(deid_dir / "deid_mapping.json", mapping)
    (deid_dir / "partner_deid.txt").write_text(partner_deid["text"], encoding="utf-8")
    (deid_dir / "subject_deid.txt").write_text(subject_deid["text"], encoding="utf-8")
    (dyad_dir / "view.html").write_text(generate_html(dyad_dir), encoding="utf-8")

    # Also update deid_output if configured
    if deid_output_root is not None:
        dyad_deid_output = deid_output_root / dyad_dir.name
        dyad_deid_output.mkdir(parents=True, exist_ok=True)
        (dyad_deid_output / "partner_deid.txt").write_text(partner_deid["text"], encoding="utf-8")
        (dyad_deid_output / "subject_deid.txt").write_text(subject_deid["text"], encoding="utf-8")

    return {
        "ok": True,
        "saved": str(deid_dir),
        "mapping": mapping,
        "deid": {"partner": partner_deid, "subject": subject_deid},
    }


def _json_response(handler: BaseHTTPRequestHandler, data: dict | list, status: int = 200) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, body: str, status: int = 200) -> None:
    encoded = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def make_handler(output_root: Path, deid_output_root: Path | None = None):
    class DeidEditorHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            logger.info(fmt, *args)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path in ("/", "/index.html"):
                    _text_response(self, EDITOR_HTML)
                    return
                if parsed.path == "/api/dyads":
                    _json_response(self, {"dyads": list_dyads(output_root)})
                    return
                if parsed.path.startswith("/api/dyad/"):
                    dyad_id = parsed.path.removeprefix("/api/dyad/")
                    _json_response(self, load_dyad(output_root, dyad_id))
                    return
                _json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/save":
                _json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                _json_response(self, save_dyad(output_root, payload, deid_output_root))
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return DeidEditorHandler


def _free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def cli() -> None:
    parser = argparse.ArgumentParser(description="Open a local editor for de-id results")
    parser.add_argument(
        "--output",
        default="../output_to_be_removed",
        help="Output folder containing dyad subfolders (default: ../output_to_be_removed)",
    )
    parser.add_argument(
        "--deid-output",
        default="deid_output",
        help="Directory for deliverable deid files (default: deid_output/)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output).expanduser().resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"Output folder not found: {output_root}")

    deid_output_root = Path(args.deid_output).expanduser().resolve()

    port = _free_port(args.port)
    server = ThreadingHTTPServer((args.host, port), make_handler(output_root, deid_output_root))
    url = f"http://{args.host}:{port}"
    logger.info(f"WashU de-id editor running at {url}")
    logger.info(f"Reading and saving dyads under {output_root}")
    logger.info(f"Deid deliverables synced to {deid_output_root}")
    if not args.no_browser:
        webbrowser.open(url)
    server.serve_forever()


EDITOR_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WashU De-id Editor</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
    color: #242424;
    background: #fff;
    font-size: 0.82rem;
    line-height: 1.65;
  }
  header {
    position: sticky;
    top: 0;
    z-index: 10;
    display: grid;
    grid-template-columns: auto minmax(220px, 420px) 1fr;
    gap: 0.75rem;
    align-items: center;
    padding: 0.85rem 1.25rem;
    border-bottom: 1px solid #e6e6e6;
    background: #fafafa;
  }
  h1 { font-size: 0.95rem; font-weight: 600; color: #555; margin: 0; }
  select, input {
    min-height: 2rem;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    background: #fff;
    color: #222;
    padding: 0 0.5rem;
    font: inherit;
  }
  button {
    min-height: 2rem;
    border: 1px solid #c7c7c7;
    border-radius: 4px;
    background: #f4f4f4;
    color: #222;
    padding: 0 0.7rem;
    font: inherit;
    cursor: pointer;
  }
  button.primary { background: #2563eb; border-color: #1d4ed8; color: #fff; }
  button.danger { background: #fff5f5; border-color: #f0b4b4; color: #9b1c1c; }
  button:disabled { opacity: 0.55; cursor: default; }
  main { padding: 1.25rem; }
  nav { margin: 0.75rem 0 1.25rem; }
  nav a { color: #2563eb; margin-right: 1.2rem; text-decoration: none; }
  h2 {
    font-size: 0.85rem;
    font-weight: 600;
    color: #666;
    margin: 1.75rem 0 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #ececec;
  }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  .col-header {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8a8a8a;
    border-bottom: 1px solid #eee;
    padding-bottom: 0.45rem;
    margin-bottom: 0.65rem;
  }
  .turn { margin-bottom: 0.58rem; }
  .speaker {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-right: 0.35rem;
  }
  .speaker.interviewer { color: #2563eb; }
  .speaker.participant { color: #059669; }
  .text { color: #333; }
  .phi {
    background: #fff3cd;
    border-bottom: 2px solid #d99500;
    padding: 0 2px;
    cursor: pointer;
  }
  .phi.selected { outline: 2px solid #2563eb; outline-offset: 1px; }
  .redacted {
    background: #f0f0f0;
    border: 1px solid #ccc;
    padding: 0 4px;
    border-radius: 3px;
    font-size: 0.7rem;
    color: #666;
    font-weight: 600;
  }
  aside {
    position: fixed;
    right: 1rem;
    bottom: 1rem;
    width: min(420px, calc(100vw - 2rem));
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    box-shadow: 0 12px 30px rgb(0 0 0 / 12%);
    padding: 1rem;
    z-index: 20;
  }
  aside h3 { margin: 0 0 0.75rem; font-size: 0.78rem; color: #555; }
  .form-grid { display: grid; grid-template-columns: 1fr 140px; gap: 0.55rem; }
  .wide { grid-column: 1 / -1; }
  .row { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
  .status { color: #666; font-size: 0.72rem; }
  .mapping { border-collapse: collapse; margin-top: 1rem; }
  .mapping th, .mapping td { border: 1px solid #eee; padding: 0.35rem 0.65rem; text-align: left; }
  .mapping th { background: #f9f9f9; color: #666; }
  @media (max-width: 900px) {
    header { grid-template-columns: 1fr; }
    .columns { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<header>
  <h1>WashU De-id Editor</h1>
  <select id="dyadSelect"></select>
  <div class="status" id="status">Loading...</div>
</header>
<main id="app"></main>
<aside>
  <h3>Add Entity</h3>
  <div class="form-grid">
    <input id="entityText" class="wide" placeholder="Select text or type exact text">
    <select id="entityType"></select>
    <button id="addBtn">Add Matches</button>
    <label class="wide row"><input type="checkbox" id="allRoles" checked> Apply to every exact match in this dyad</label>
  </div>
  <div id="selectedInfo" class="status" style="margin-top:0.7rem;">Click a highlighted entity to remove it.</div>
  <div class="row" style="margin-top:0.7rem;">
    <button class="danger" id="removeBtn" disabled>Remove Selected</button>
  </div>
</aside>
<script>
const state = { dyads: [], current: null, selected: null, saving: false };
const escapeHtml = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labelOf = (group) => String(group || '').replace(/^private_/, '').toLowerCase();
const groupOf = (label) => `private_${String(label || 'other').replace(/^private_/, '').toLowerCase()}`;

function setStatus(text) { document.getElementById('status').textContent = text; }
function setSaving(isSaving) {
  state.saving = isSaving;
  document.getElementById('addBtn').disabled = isSaving;
  document.getElementById('removeBtn').disabled = isSaving || !state.selected;
}

async function init() {
  const dyadsRes = await fetch('/api/dyads');
  state.dyads = (await dyadsRes.json()).dyads || [];
  const select = document.getElementById('dyadSelect');
  select.innerHTML = state.dyads.map(id => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`).join('');
  select.addEventListener('change', () => loadDyad(select.value));
  document.getElementById('addBtn').addEventListener('click', addEntity);
  document.getElementById('removeBtn').addEventListener('click', removeSelected);
  document.addEventListener('selectionchange', syncSelectedText);
  if (state.dyads.length) await loadDyad(state.dyads[0]);
  else setStatus('No editable dyads found.');
}

async function loadDyad(id) {
  setStatus('Loading dyad...');
  state.selected = null;
  const res = await fetch(`/api/dyad/${encodeURIComponent(id)}`);
  state.current = await res.json();
  document.getElementById('entityType').innerHTML = state.current.entityTypes.map(t => `<option value="${t}">${t}</option>`).join('');
  render();
  setStatus('Loaded');
}

function renderTextWithEntities(text, entities, role, turnIndex) {
  const sorted = [...(entities || [])].sort((a, b) => a.start - b.start);
  let html = '';
  let pos = 0;
  sorted.forEach((ent, entityIndex) => {
    if (ent.start < pos) return;
    html += escapeHtml(text.slice(pos, ent.start));
    const selected = state.selected && state.selected.role === role && state.selected.turnIndex === turnIndex && state.selected.entityIndex === entityIndex;
    html += `<span class="phi${selected ? ' selected' : ''}" data-role="${role}" data-turn="${turnIndex}" data-entity="${entityIndex}" title="${escapeHtml(labelOf(ent.entity_group))}">${escapeHtml(text.slice(ent.start, ent.end))}</span>`;
    pos = ent.end;
  });
  return html + escapeHtml(text.slice(pos));
}

function renderCoded(text) {
  return escapeHtml(text || '').replace(/\[([A-Z_0-9]+)\]/g, '<span class="redacted">[$1]</span>');
}

function renderSection(role, title) {
  const orig = state.current.original[role].utterances || [];
  const deid = state.current.deid[role].utterances || [];
  const left = orig.map((utt, i) => {
    const entities = (deid[i] || {}).entities || [];
    const speaker = escapeHtml(utt.speaker || '');
    const speakerClass = speaker === 'interviewer' ? 'interviewer' : 'participant';
    return `<div class="turn"><span class="speaker ${speakerClass}">${speaker}</span><span class="text" data-role="${role}" data-turn="${i}">${renderTextWithEntities(utt.text || '', entities, role, i)}</span></div>`;
  }).join('');
  const right = deid.map((utt) => {
    const speaker = escapeHtml(utt.speaker || '');
    const speakerClass = speaker === 'interviewer' ? 'interviewer' : 'participant';
    return `<div class="turn"><span class="speaker ${speakerClass}">${speaker}</span><span class="text">${renderCoded(utt.text || '')}</span></div>`;
  }).join('');
  return `<h2 id="${role}">${title}</h2><div class="columns"><div><div class="col-header">Original (edit highlighted PHI)</div>${left}</div><div><div class="col-header">De-identified (coded)</div>${right}</div></div>`;
}

function renderMapping() {
  const rows = Object.entries(state.current.mapping || {}).sort().map(([code, original]) => `<tr><td>[${escapeHtml(code)}]</td><td>${escapeHtml(original)}</td></tr>`).join('');
  return `<h2 id="mapping">Entity Mapping</h2><table class="mapping"><tr><th>Code</th><th>Original</th></tr>${rows}</table>`;
}

function render() {
  document.getElementById('app').innerHTML = `<nav><a href="#partner">Study Partner Interview</a><a href="#subject">Subject Interview</a><a href="#mapping">Entity Mapping</a></nav>${renderSection('partner', 'Study Partner Interview')}${renderSection('subject', 'Subject Interview')}${renderMapping()}`;
  document.querySelectorAll('.phi').forEach(el => el.addEventListener('click', selectEntity));
  updateSelectedPanel();
}

function selectEntity(event) {
  const el = event.currentTarget;
  state.selected = {
    role: el.dataset.role,
    turnIndex: Number(el.dataset.turn),
    entityIndex: Number(el.dataset.entity)
  };
  const ent = state.current.deid[state.selected.role].utterances[state.selected.turnIndex].entities[state.selected.entityIndex];
  document.getElementById('entityText').value = ent.word || el.textContent;
  document.getElementById('entityType').value = labelOf(ent.entity_group);
  render();
}

function updateSelectedPanel() {
  const removeBtn = document.getElementById('removeBtn');
  const info = document.getElementById('selectedInfo');
  if (!state.selected) {
  removeBtn.disabled = true;
    info.textContent = 'Click a highlighted entity to remove it.';
    return;
  }
  const ent = state.current.deid[state.selected.role].utterances[state.selected.turnIndex].entities[state.selected.entityIndex];
  removeBtn.disabled = state.saving;
  info.textContent = `Selected: "${ent.word}" (${labelOf(ent.entity_group)})`;
}

async function removeSelected() {
  if (!state.selected) return;
  const selectedUtt = state.current.deid[state.selected.role].utterances[state.selected.turnIndex];
  const selectedEnt = selectedUtt.entities[state.selected.entityIndex];
  const removeAll = document.getElementById('allRoles').checked;
  let removed = 0;

  if (removeAll) {
    const selectedText = String(selectedEnt.word || '').toLowerCase();
    const selectedType = labelOf(selectedEnt.entity_group);
    ['partner', 'subject'].forEach(role => {
      (state.current.deid[role].utterances || []).forEach(utt => {
        const before = (utt.entities || []).length;
        utt.entities = (utt.entities || []).filter(ent => {
          const sameText = String(ent.word || '').toLowerCase() === selectedText;
          const sameType = labelOf(ent.entity_group) === selectedType;
          return !(sameText && sameType);
        });
        removed += before - utt.entities.length;
      });
    });
  } else {
    selectedUtt.entities.splice(state.selected.entityIndex, 1);
    removed = 1;
  }

  state.selected = null;
  render();
  if (removed) await autosave(`Removed ${removed} match${removed === 1 ? '' : 'es'}.`);
}

function syncSelectedText() {
  const selection = window.getSelection();
  const text = selection ? selection.toString().trim() : '';
  if (text) document.getElementById('entityText').value = text;
}

function findMatches(haystack, needle) {
  const matches = [];
  if (!needle) return matches;
  let start = 0;
  const lowerHaystack = haystack.toLowerCase();
  const lowerNeedle = needle.toLowerCase();
  while (true) {
    const idx = lowerHaystack.indexOf(lowerNeedle, start);
    if (idx === -1) break;
    matches.push({start: idx, end: idx + needle.length});
    start = idx + Math.max(needle.length, 1);
  }
  return matches;
}

function overlapsAny(entities, start, end) {
  return entities.some(e => start < e.end && end > e.start);
}

async function addEntity() {
  const text = document.getElementById('entityText').value.trim();
  const type = document.getElementById('entityType').value;
  if (!text) return;
  let added = 0;
  const roles = ['partner', 'subject'];
  const applyAll = document.getElementById('allRoles').checked;
  const selectedOnly = !applyAll && state.selected;
  roles.forEach(role => {
    const origUtterances = state.current.original[role].utterances || [];
    const deidUtterances = state.current.deid[role].utterances || [];
    origUtterances.forEach((utt, turnIndex) => {
      if (selectedOnly && (role !== state.selected.role || turnIndex !== state.selected.turnIndex)) return;
      const target = deidUtterances[turnIndex];
      if (!target) return;
      target.entities = target.entities || [];
      findMatches(utt.text || '', text).forEach(match => {
        if (overlapsAny(target.entities, match.start, match.end)) return;
        target.entities.push({
          entity_group: groupOf(type),
          word: (utt.text || '').slice(match.start, match.end),
          start: match.start,
          end: match.end,
          score: 1.0
        });
        added += 1;
      });
      target.entities.sort((a, b) => a.start - b.start);
    });
  });
  if (added) {
    state.selected = null;
    render();
    await autosave(`Added ${added} match${added === 1 ? '' : 'es'}.`);
  } else {
    setStatus('No new non-overlapping matches found.');
  }
}

async function autosave(prefix) {
  if (!state.current) return;
  setSaving(true);
  setStatus(`${prefix} Saving...`);
  const payload = { id: state.current.id, deid: deepcopyDeid(state.current.deid) };
  const res = await fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (!data.ok) {
    setStatus(`Autosave failed: ${data.error || 'unknown error'}`);
    setSaving(false);
    return;
  }
  state.current.deid = data.deid;
  state.current.mapping = data.mapping;
  render();
  setSaving(false);
  setStatus(`${prefix} Autosaved.`);
}

function deepcopyDeid(deid) {
  return JSON.parse(JSON.stringify(deid));
}

init().catch(err => setStatus(`Error: ${err.message}`));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    cli()
