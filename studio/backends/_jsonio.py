"""Shared: parse a JSON value from model output, tolerating fences / stray prose.

Lifted so every backend (CLI, Gemini, mock) shares one tolerant parser instead
of duplicating the logic. Raises :class:`JSONParseError` (a ``ValueError``) when
no JSON value can be recovered.
"""

from __future__ import annotations

import json
import re


class JSONParseError(ValueError):
    """No JSON value could be parsed from the model output."""


def extract_json(text: str):
    """Parse a JSON value from ``text``, tolerating code fences / leading prose."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first JSON value starting at the first { or [.
    # raw_decode parses one value and ignores trailing text — O(n), not O(n^2).
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start != -1:
        try:
            return json.JSONDecoder().raw_decode(text, start)[0]
        except json.JSONDecodeError:
            pass
    raise JSONParseError(f"could not parse JSON from model output: {text[:500]}")
