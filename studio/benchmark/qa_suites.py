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
    default_limit: int | None = None                   # cap for huge remote splits
    extra: dict = field(default_factory=dict)


# --- shared helpers -------------------------------------------------------

def _download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size == 0:
        urllib.request.urlretrieve(url, dst)
    return dst


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _fetch_hf_rows(dataset: str, config: str, split: str, n: int, dst: Path) -> list[dict]:
    """Fetch up to ``n`` rows from the HF datasets-server (plain JSON over HTTP —
    no ``datasets`` lib, no parquet), caching the result to ``dst``. Pages in
    chunks of 100 (the server's max ``length``)."""
    if dst.exists() and dst.stat().st_size > 0:
        rows = _read_jsonl(dst)
        if len(rows) >= n:
            return rows[:n]
    rows: list[dict] = []
    base = "https://datasets-server.huggingface.co/rows"
    for offset in range(0, n, 100):
        length = min(100, n - offset)
        url = (f"{base}?dataset={dataset}&config={config}&split={split}"
               f"&offset={offset}&length={length}")
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        page = json.loads(urllib.request.urlopen(req, timeout=60).read())
        batch = [r["row"] for r in page.get("rows", [])]
        rows.extend(batch)
        if len(batch) < length:
            break  # ran out of data
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(json.dumps(r) for r in rows))
    return rows[:n]


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


# --- HotpotQA (multi-hop QA; distractor = paragraphs provided inline) ------

def _hotpot_context(ctx: dict) -> str:
    """Render the distractor config's parallel {title:[...], sentences:[[...]]}
    arrays into readable, numbered source paragraphs."""
    titles = ctx.get("title", []) or []
    sents = ctx.get("sentences", []) or []
    parts = []
    for i, title in enumerate(titles):
        body = "".join(sents[i]) if i < len(sents) else ""
        parts.append(f"[{i + 1}] {title}: {body}")
    return "Sources:\n" + "\n".join(parts)


def _load_hotpot(cache_dir: Path, limit: int | None) -> list[QATask]:
    n = limit or 500
    rows = _fetch_hf_rows("hotpotqa/hotpot_qa", "distractor", "validation", n,
                          cache_dir / "hotpot_distractor_val.jsonl")
    return [
        QATask(id=str(r.get("id", i)), question=r["question"],
               gold=[r["answer"]], context=_hotpot_context(r.get("context", {})))
        for i, r in enumerate(rows)
    ]


def _grade_hotpot(output: str, task: QATask) -> float:
    """SQuAD-style token-F1 of the extracted answer vs the gold answer —
    continuous in [0,1], good hill-climb signal for multi-hop QA."""
    return _f1(_extract_tagged(output), task.gold)


_HOTPOT_SEED = """\
# Multi-hop question answering

You are given a question and a set of numbered source paragraphs. Read the
sources and reason across MULTIPLE of them to find the answer — the answer
usually requires combining facts from two different paragraphs.

Base your answer ONLY on the provided sources. Keep the final answer as short as
possible (a name, entity, number, or yes/no — no explanation).

Output the final answer wrapped in tags, like: <answer>...</answer>
"""


# --- IFEval (instruction-following; programmatic constraint checks) --------

def _load_ifeval(cache_dir: Path, limit: int | None) -> list[QATask]:
    from .ifeval_checks import supported

    rows = _fetch_hf_rows("google/IFEval", "default", "train", limit or 541,
                          cache_dir / "ifeval_train.jsonl")
    tasks = []
    for r in rows:
        ids = r.get("instruction_id_list", [])
        # keep only prompts whose every constraint we can verify faithfully
        if not ids or not all(supported(i) for i in ids):
            continue
        kwargs = [{k: v for k, v in d.items() if v is not None} for d in r.get("kwargs", [])]
        tasks.append(QATask(
            id=str(r.get("key")), question=r["prompt"],
            meta={"instruction_id_list": ids, "kwargs": kwargs},
        ))
        if limit and len(tasks) >= limit:
            break
    return tasks


def _grade_ifeval(output: str, task: QATask) -> float:
    """Fraction of the prompt's instructions the response satisfies (loose,
    per-instruction accuracy). The model's raw text IS the deliverable here —
    no answer extraction."""
    from .ifeval_checks import check

    ids = task.meta.get("instruction_id_list", [])
    kwargs = task.meta.get("kwargs", [])
    if not ids:
        return 0.0
    ok = sum(check(i, output, kwargs[j] if j < len(kwargs) else {})
             for j, i in enumerate(ids))
    return ok / len(ids)


_IFEVAL_SEED = """\
You are a precise assistant. Follow EVERY explicit instruction in the user's
request exactly — word/sentence/paragraph counts, required or forbidden words,
formatting (bullets, titles, highlights, JSON), case, punctuation, and any
start/end phrasing. If multiple constraints apply, satisfy all of them at once.

Output only the requested content — no preamble, no explanation of how you
followed the instructions.
"""


# --- SearchQA (Jeopardy-style trivia; search snippets provided as context) -

def _load_searchqa(cache_dir: Path, limit: int | None) -> list[QATask]:
    n = limit or 300
    rows = _fetch_hf_rows("lucadiliello/searchqa", "default", "validation", n,
                          cache_dir / "searchqa_val.jsonl")
    out = []
    for i, r in enumerate(rows):
        golds = r.get("answers") or []
        if not golds:
            continue
        out.append(QATask(id=str(r.get("key", i)), question=r["question"],
                          gold=list(golds), context=(r.get("context") or "")[:4000]))
    return out


def _grade_searchqa(output: str, task: QATask) -> float:
    """Token-F1 of the extracted short answer vs the gold answer(s)."""
    return _f1(_extract_tagged(output), task.gold)


_SEARCHQA_SEED = """\
You are answering a Jeopardy-style trivia clue. You are given the clue and a set
of web search snippets that contain the answer.

Read the snippets, identify the single best answer to the clue, and give the
SHORTEST exact answer (a name, place, title, or number — no sentence, no
explanation).

Output the final answer wrapped in tags, like: <answer>...</answer>
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
    "hotpot": Suite(
        name="hotpot",
        load=_load_hotpot,
        grader=_grade_hotpot,
        seed_prompt=_HOTPOT_SEED,
        baseline_score=None,
        baseline_note="HotpotQA distractor (validation), token-F1; model-dependent",
        domain="multi-hop question answering over provided sources",
        io_contract=("A question plus numbered source paragraphs. Combine facts "
                     "across sources; answer concisely in <answer>…</answer>."),
        default_limit=300,
    ),
    "ifeval": Suite(
        name="ifeval",
        load=_load_ifeval,
        grader=_grade_ifeval,
        seed_prompt=_IFEVAL_SEED,
        baseline_score=None,
        baseline_note=("IFEval (verifiable-subset; per-instruction accuracy). "
                       "Regex tokenization, not the official nltk harness."),
        domain="instruction following with verifiable constraints",
        io_contract=("A request with explicit formatting/length/keyword/case "
                     "constraints. Produce content that satisfies ALL of them."),
    ),
    "searchqa": Suite(
        name="searchqa",
        load=_load_searchqa,
        grader=_grade_searchqa,
        seed_prompt=_SEARCHQA_SEED,
        baseline_score=None,
        baseline_note="SearchQA (lucadiliello/searchqa validation), token-F1; model-dependent",
        domain="Jeopardy-style trivia QA over provided search snippets",
        io_contract=("A trivia clue plus web search snippets. Give the shortest "
                     "exact answer in <answer>…</answer>."),
        default_limit=300,
    ),
}


def list_suites() -> list[str]:
    return sorted(_SUITES)


def get_suite(name: str) -> Suite:
    if name not in _SUITES:
        raise KeyError(f"unknown QA suite {name!r}; have {list_suites()}")
    return _SUITES[name]
