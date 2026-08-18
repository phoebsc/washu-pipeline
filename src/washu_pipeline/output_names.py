"""Helpers for naming pipeline output folders."""

from __future__ import annotations

import re


TRAILING_DATE_RE = re.compile(r"([_-]?)\d{1,2}-\d{1,2}-\d{2,4}$")


def deid_output_name(source_name: str) -> str:
    """Return the deid_output folder name without a trailing interview date."""
    clean_name = TRAILING_DATE_RE.sub("", source_name).rstrip("_- ")
    return clean_name or source_name
