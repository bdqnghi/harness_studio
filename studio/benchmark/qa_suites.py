"""QA suites: per-benchmark dataset loader + grader + seed prompt, behind one
``get_suite(name)`` so wiring a target is a one-liner.

A :class:`Suite` is everything ``QABenchmark`` + the Target need that is specific
to one benchmark: how to load its tasks (cached locally — no live web), how to
grade an answer (deterministic), the shipped seed prompt (the warm-start harness
we hill-climb), and the published baseline note. Add a benchmark by writing one
loader + one grader and appending a ``_SUITES`` entry.

Datasets are fetched once to ``cache_dir`` from stable raw sources (no docker, no
``datasets`` lib, no parquet) and reused.
"""

from __future__ import annotations

import json
import re
import string
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .qa import QATask

DEFAULT_CACHE = Path("artifacts/qa_cache")


@dataclass
class Suite:
    name: str
    load: Callable[[Path, int | None], list[QATask]]   # (cache_dir, limit) -> tasks
    grader: Callable[[str, QATask], float]
    seed_prompt: str
    baseline_score: float | None = None
    baseline_note: str = ""
    domain: str = ""                                   # for the cold-start brief
    io_contract: str = ""
    temperature: float = 0.0
    extra: dict = field(default_factory=dict)


# --- shared helpers -------------------------------------------------------

def _download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size == 0:
        urllib.request.urlretrieve(url, dst)
    return dst


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _norm(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles/space."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split()).strip()


def _f1(pred: str, golds: list[str]) -> float:
    """Token-level F1 (SQuAD), max over gold answers."""
    pt = _norm(pred).split()
    if not pt:
        return 1.0 if any(not _norm(g).split() for g in golds) else 0.0
    best = 0.0
    for g in golds:
        gt = _norm(g).split()
        if not gt:
            continue
        common = sum((Counter(pt) & Counter(gt)).values())
        if common == 0:
            continue
        prec, rec = common / len(pt), common / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def _extract_tagged(text: str) -> str:
    """Answer inside <answer>…</answer>, else the last non-empty line."""
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


# --- GSM8K (grade-school math; programmatic integer match) ----------------

_GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
              "master/grade_school_math/data/test.jsonl")

_INT_RE = re.compile(r"-?\d[\d,]*")


def _gsm8k_gold(answer_field: str) -> str:
    return answer_field.split("####")[-1].strip().replace(",", "")


def _load_gsm8k(cache_dir: Path, limit: int | None) -> list[QATask]:
    path = _download(_GSM8K_URL, cache_dir / "gsm8k_test.jsonl")
    rows = _read_jsonl(path)
    if limit:
        rows = rows[:limit]
    return [
        QATask(id=str(i), question=r["question"], gold=[_gsm8k_gold(r["answer"])])
        for i, r in enumerate(rows)
    ]


def _grade_gsm8k(output: str, task: QATask) -> float:
    """Correct iff the model's final integer equals the gold integer. Prefers a
    ``#### N`` marker or an <answer> tag, else the last integer in the text."""
    gold = task.gold[0]
    marked = re.search(r"####\s*(-?\d[\d,]*)", output)
    cand = marked.group(1) if marked else _extract_tagged(output)
    nums = _INT_RE.findall(cand) or _INT_RE.findall(output)
    if not nums:
        return 0.0
    return 1.0 if nums[-1].replace(",", "") == gold else 0.0


_GSM8K_SEED = """\
# Math problem solver

You are a careful mathematician. Solve the word problem step by step, showing
your arithmetic. Do not skip steps.

When you are done, write the final numeric answer on its own line in exactly this
format (digits only, no units, no commas):

#### <answer>
"""


# --- registry -------------------------------------------------------------

_SUITES: dict[str, Suite] = {
    "gsm8k": Suite(
        name="gsm8k",
        load=_load_gsm8k,
        grader=_grade_gsm8k,
        seed_prompt=_GSM8K_SEED,
        baseline_score=None,  # model-dependent; the profiled seed is our bar
        baseline_note="GSM8K test (grade-school-math); accuracy is model-dependent",
        domain="grade-school math word problems",
        io_contract=("A math word problem. Reason step by step, then output the "
                     "final integer answer on its own line after '#### '."),
    ),
}


def list_suites() -> list[str]:
    return sorted(_SUITES)


def get_suite(name: str) -> Suite:
    if name not in _SUITES:
        raise KeyError(f"unknown QA suite {name!r}; have {list_suites()}")
    return _SUITES[name]
