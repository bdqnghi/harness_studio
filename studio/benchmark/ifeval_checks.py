"""A deterministic, self-contained verifier registry for IFEval-style
instruction-following constraints.

IFEval (google/IFEval) gives each prompt a list of ``instruction_id`` strings +
per-instruction ``kwargs``; an answer is graded by whether each constraint is
satisfied. We implement the common, unambiguously-verifiable instruction types
here and FILTER the dataset to prompts whose every instruction is supported, so
grading is faithful for the tasks we keep. Tokenization is regex-based (not
nltk), so absolute numbers won't match the official harness exactly, but the
checker is internally consistent and deterministic — which is all the optimizer
needs to hill-climb against. No external libs, no network.

Each checker: ``fn(response: str, kwargs: dict) -> bool``.
"""

from __future__ import annotations

import json
import re

_WORD_RE = re.compile(r"\b\w+\b")


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _count_sentences(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


def _relation_ok(count: int, relation: str | None, n: int) -> bool:
    rel = (relation or "at least").strip()
    if rel == "at least":
        return count >= n
    if rel == "at most":
        return count <= n
    if rel == "less than":
        return count < n
    if rel == "more than":
        return count > n
    if rel == "exactly":
        return count == n
    return count >= n  # default


# --- individual checkers --------------------------------------------------

def _keywords_existence(r, kw):
    return all(k.lower() in r.lower() for k in (kw.get("keywords") or []))


def _keywords_frequency(r, kw):
    word = (kw.get("keyword") or "").lower()
    n = kw.get("frequency") or 0
    count = len(re.findall(rf"\b{re.escape(word)}\b", r.lower())) if word else 0
    return _relation_ok(count, kw.get("relation"), n)


def _keywords_forbidden(r, kw):
    return not any(re.search(rf"\b{re.escape(w.lower())}\b", r.lower())
                   for w in (kw.get("forbidden_words") or []))


def _letter_frequency(r, kw):
    letter = (kw.get("letter") or "").lower()
    n = kw.get("let_frequency") or 0
    count = r.lower().count(letter) if letter else 0
    return _relation_ok(count, kw.get("let_relation"), n)


def _number_words(r, kw):
    return _relation_ok(_count_words(r), kw.get("relation"), kw.get("num_words") or 0)


def _number_sentences(r, kw):
    return _relation_ok(_count_sentences(r), kw.get("relation"), kw.get("num_sentences") or 0)


def _number_paragraphs(r, kw):
    paras = [p for p in re.split(r"\n\s*\n", r.strip()) if p.strip()]
    return _relation_ok(len(paras), kw.get("relation"), kw.get("num_paragraphs") or 0)


def _number_bullets(r, kw):
    bullets = len(re.findall(r"^\s*[\*\-]\s+", r, re.MULTILINE))
    return bullets == (kw.get("num_bullets") or 0)


def _highlighted_sections(r, kw):
    # markdown *highlight* or **highlight**
    n = len(re.findall(r"\*[^*\n]+\*", r))
    return n >= (kw.get("num_highlights") or 0)


def _number_placeholders(r, kw):
    return len(re.findall(r"\[[^\]\n]+\]", r)) >= (kw.get("num_placeholders") or 0)


def _postscript(r, kw):
    marker = (kw.get("postscript_marker") or "P.S.").lower()
    return marker.lower() in r.lower()


def _title(r, kw):
    return bool(re.search(r"<<[^>\n]+>>", r))


def _json_format(r, kw):
    s = r.strip()
    s = re.sub(r"^```(json)?|```$", "", s, flags=re.IGNORECASE).strip()
    try:
        json.loads(s)
        return True
    except Exception:  # noqa: BLE001
        return False


def _all_lowercase(r, kw):
    return r == r.lower()


def _all_uppercase(r, kw):
    return r == r.upper()


def _capital_word_frequency(r, kw):
    caps = len([w for w in _WORD_RE.findall(r) if w.isupper() and len(w) > 1])
    return _relation_ok(caps, kw.get("capital_relation"), kw.get("capital_frequency") or 0)


def _no_comma(r, kw):
    return "," not in r


def _end_checker(r, kw):
    phrase = (kw.get("end_phrase") or "").strip()
    return r.strip().lower().endswith(phrase.lower()) if phrase else True


def _quotation(r, kw):
    s = r.strip()
    return len(s) >= 2 and s.startswith('"') and s.endswith('"')


def _two_responses(r, kw):
    return "******" in r


def _multiple_sections(r, kw):
    spliter = kw.get("section_spliter") or "Section"
    n = kw.get("num_sections") or 0
    return len(re.findall(re.escape(spliter), r)) >= n


CHECKERS = {
    "keywords:existence": _keywords_existence,
    "keywords:frequency": _keywords_frequency,
    "keywords:forbidden_words": _keywords_forbidden,
    "keywords:letter_frequency": _letter_frequency,
    "length_constraints:number_words": _number_words,
    "length_constraints:number_sentences": _number_sentences,
    "length_constraints:number_paragraphs": _number_paragraphs,
    "detectable_format:number_bullet_lists": _number_bullets,
    "detectable_format:number_highlighted_sections": _highlighted_sections,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:json_format": _json_format,
    "detectable_format:title": _title,
    "detectable_content:number_placeholders": _number_placeholders,
    "detectable_content:postscript": _postscript,
    "change_case:english_lowercase": _all_lowercase,
    "change_case:english_capital": _all_uppercase,
    "change_case:capital_word_frequency": _capital_word_frequency,
    "punctuation:no_comma": _no_comma,
    "startend:end_checker": _end_checker,
    "startend:quotation": _quotation,
    "combination:two_responses": _two_responses,
}


def supported(instruction_id: str) -> bool:
    return instruction_id in CHECKERS


def check(instruction_id: str, response: str, kwargs: dict) -> bool:
    fn = CHECKERS.get(instruction_id)
    if fn is None:
        return False
    try:
        return bool(fn(response, kwargs or {}))
    except Exception:  # noqa: BLE001 — a malformed kwargs must not crash grading
        return False
