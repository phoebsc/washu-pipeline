"""HTML viewer for de-identified transcripts.

Generates a side-by-side visualization showing the original transcript
alongside the de-identified version, with PHI entities highlighted.

Usage:
    uv run washu-view --input output/Tape_11_Interview_(Source)_1 --output output/Tape_11_Interview_(Source)_1/view.html
"""

import argparse
import json
import re
import logging
from html import escape
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


def _highlight_entities(text: str, entities: list[dict]) -> str:
    """Highlight detected PHI spans in the original text."""
    if not entities or not text:
        return escape(text)
    sorted_ents = sorted(entities, key=lambda e: e["start"])
    parts = []
    prev_end = 0
    for ent in sorted_ents:
        start, end = ent["start"], ent["end"]
        if start < prev_end:
            continue
        parts.append(escape(text[prev_end:start]))
        label = ent.get("entity_group", "").replace("private_", "")
        parts.append(
            f'<span class="phi" title="{escape(label)}">{escape(text[start:end])}</span>'
        )
        prev_end = end
    parts.append(escape(text[prev_end:]))
    return "".join(parts)


def _render_coded(text: str) -> str:
    """Render de-identified text with styled [CODE] tags."""
    if not text:
        return ""
    escaped = escape(text)

    def replacer(match):
        label = match.group(1)
        return f'<span class="redacted">[{label}]</span>'

    return re.sub(r"\[([A-Z_0-9]+)\]", replacer, escaped)


def _build_transcript_section(
    title: str,
    section_id: str,
    original_utterances: list[dict],
    deid_utterances: list[dict],
) -> str:
    """Build a side-by-side section for one transcript (partner or subject)."""
    left_rows = []
    right_rows = []

    for i, orig in enumerate(original_utterances):
        deid = deid_utterances[i] if i < len(deid_utterances) else {"speaker": "", "text": "", "entities": []}
        entities = deid.get("entities", [])

        speaker_class = "speaker-interviewer" if orig["speaker"] == "interviewer" else "speaker-participant"

        highlighted = _highlight_entities(orig["text"], entities)
        left_rows.append(
            f'<div class="turn">'
            f'<span class="{speaker_class}">{escape(orig["speaker"])}</span> '
            f'<span class="text">{highlighted}</span>'
            f'</div>'
        )

        coded = _render_coded(deid["text"])
        right_rows.append(
            f'<div class="turn">'
            f'<span class="{speaker_class}">{escape(deid.get("speaker", ""))}</span> '
            f'<span class="text">{coded}</span>'
            f'</div>'
        )

    return (
        f'<h2 id="{section_id}">{escape(title)}</h2>\n'
        f'<div class="columns">\n'
        f'  <div class="col">\n'
        f'    <div class="col-header">Original (PHI highlighted)</div>\n'
        f'    {"".join(left_rows)}\n'
        f'  </div>\n'
        f'  <div class="col">\n'
        f'    <div class="col-header">De-identified (coded)</div>\n'
        f'    {"".join(right_rows)}\n'
        f'  </div>\n'
        f'</div>\n'
    )


def _build_mapping_section(mapping: dict[str, str]) -> str:
    """Build an HTML table for the de-id mapping."""
    if not mapping:
        return ""
    rows = []
    for code, original in sorted(mapping.items()):
        rows.append(f'<tr><td class="code">[{escape(code.upper())}]</td><td>{escape(original)}</td></tr>')
    return (
        '<h2 id="mapping">Entity Mapping</h2>\n'
        '<table class="mapping">\n'
        '<tr><th>Code</th><th>Original</th></tr>\n'
        f'{"".join(rows)}\n'
        '</table>\n'
    )


def generate_html(tape_dir: Path) -> str:
    """Generate full HTML viewer for a processed tape directory."""
    tape_name = tape_dir.name
    deid_dir = tape_dir / "deid"

    sections = []

    for role in ("partner", "subject"):
        orig_path = tape_dir / f"{role}_transcript.json"
        deid_path = deid_dir / f"{role}_deid.json"
        if not deid_path.exists():
            deid_path = deid_dir / f"{role}_transcript.json"

        if not orig_path.exists() or not deid_path.exists():
            logger.warning(f"Missing files for {role}, skipping")
            continue

        with open(orig_path) as f:
            orig_data = json.load(f)
        with open(deid_path) as f:
            deid_data = json.load(f)

        title = "Study Partner Interview" if role == "partner" else "Subject Interview"
        sections.append(
            _build_transcript_section(
                title,
                role,
                orig_data["utterances"],
                deid_data["utterances"],
            )
        )

    mapping_path = deid_dir / "deid_mapping.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            mapping = json.load(f)
        sections.append(_build_mapping_section(mapping))

    return _HTML_TEMPLATE.format(
        tape_name=escape(tape_name),
        sections="\n".join(sections),
    )


def cli():
    parser = argparse.ArgumentParser(
        description="Generate HTML viewer for de-identified transcripts"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a processed tape output directory (containing *_transcript.json and deid/)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output HTML file path (default: <input>/view.html)",
    )
    args = parser.parse_args()

    tape_dir = Path(args.input)
    if not tape_dir.exists():
        raise FileNotFoundError(f"Directory not found: {tape_dir}")

    html = generate_html(tape_dir)

    out_path = Path(args.output) if args.output else tape_dir / "view.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    logger.info(f"HTML viewer saved: {out_path}")


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{tape_name} — De-identification Viewer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
    background: #fff;
    color: #222;
    padding: 2rem;
    font-size: 0.82rem;
    line-height: 1.7;
  }}
  h1 {{
    font-size: 1rem;
    font-weight: 500;
    color: #555;
    margin-bottom: 0.5rem;
  }}
  nav {{
    margin: 1rem 0;
    padding: 0.75rem 1rem;
    background: #f9f9f9;
    border: 1px solid #eee;
    border-radius: 4px;
    font-size: 0.75rem;
  }}
  nav a {{
    color: #2563eb;
    text-decoration: none;
    margin-right: 1.5rem;
  }}
  nav a:hover {{
    text-decoration: underline;
  }}
  h2 {{
    font-size: 0.85rem;
    font-weight: 500;
    color: #777;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #eee;
  }}
  .columns {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
  }}
  .col-header {{
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #aaa;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #eee;
    margin-bottom: 0.75rem;
  }}
  .turn {{ margin-bottom: 0.5rem; }}
  .speaker-interviewer {{
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #2563eb;
  }}
  .speaker-participant {{
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #059669;
  }}
  .text {{ color: #333; }}
  .phi {{
    background: #fff3cd;
    border-bottom: 2px solid #e6a817;
    padding: 0 2px;
    cursor: help;
  }}
  .redacted {{
    background: #f0f0f0;
    border: 1px solid #ccc;
    padding: 0 4px;
    border-radius: 3px;
    font-size: 0.7rem;
    color: #666;
    font-weight: 500;
  }}
  .mapping {{
    border-collapse: collapse;
    margin-top: 1rem;
    font-size: 0.8rem;
  }}
  .mapping th, .mapping td {{
    border: 1px solid #eee;
    padding: 0.4rem 0.8rem;
    text-align: left;
  }}
  .mapping th {{
    background: #f9f9f9;
    font-weight: 600;
    color: #666;
  }}
  .mapping .code {{
    font-weight: 500;
    color: #666;
    background: #f0f0f0;
  }}
</style>
</head>
<body>
<h1>{tape_name}</h1>
<nav>
  <a href="#partner">Study Partner Interview</a>
  <a href="#subject">Subject Interview</a>
  <a href="#mapping">Entity Mapping</a>
</nav>
{sections}
</body>
</html>
"""


if __name__ == "__main__":
    cli()
