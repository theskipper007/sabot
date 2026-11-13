"""System 1: a small RAG pipeline.

question -> BM25 over sentence-window chunks -> top-k chunks -> one LLM call -> JSON answer.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field

from .bm25 import BM25
from .config import SystemConfig
from .data import Dataset, Example, Paragraph
from .llm import LLM


@dataclass
class Chunk:
    title: str
    sent_start: int
    sentences: list[str]

    @property
    def text(self) -> str:
        return "".join(self.sentences).strip()


def chunk_paragraph(p: Paragraph, size: int, overlap: int) -> list[Chunk]:
    """Sentence windows of `size` sentences; consecutive windows share `overlap` sentences."""
    if size < 1 or not (0 <= overlap < size):
        raise ValueError("need size >= 1 and 0 <= overlap < size")
    s = p.sentences
    if len(s) <= size:
        return [Chunk(p.title, 0, list(s))]
    stride = size - overlap
    out, i = [], 0
    while True:
        out.append(Chunk(p.title, i, list(s[i:i + size])))
        if i + size >= len(s):
            break
        i += stride
    return out


class Index:
    def __init__(self, corpus: dict[str, Paragraph], size: int, overlap: int):
        self.chunks = [c for p in corpus.values() for c in chunk_paragraph(p, size, overlap)]
        self.bm25 = BM25([f"{c.title}. {c.text}" for c in self.chunks])

    def search(self, query: str, k: int) -> list[Chunk]:
        return [self.chunks[i] for i in self.bm25.top(query, k)]


@dataclass
class RagResult:
    id: str
    answer: str
    raw: str
    parsed_json: bool
    stop_reason: str
    retrieved: list[str] = field(default_factory=list)  # "title#sent_start"
    injected: bool = False


def make_conflict(chunk: Chunk, gold: str, decoy: str) -> Chunk | None:
    """A stale/other-tenant copy of a gold-bearing chunk with the answer swapped for a decoy."""
    if not gold or gold.lower() in ("yes", "no"):
        return None
    pat = re.compile(re.escape(gold), re.I)
    if not pat.search(chunk.text):
        return None
    sents = [pat.sub(decoy, s) for s in chunk.sentences]
    return Chunk(f"{chunk.title} (archived 2009 revision, tenant: northwind-legacy)", chunk.sent_start, sents)


def format_context(chunks: list[Chunk]) -> str:
    return "\n".join(f"[{i}] {c.title}: {c.text}" for i, c in enumerate(chunks, 1))


def parse_answer(raw: str) -> tuple[str, bool]:
    """Take "answer" from the first JSON object in the reply; otherwise fall back to the
    raw text (what a lenient production parser would do)."""
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "answer" in obj:
                return str(obj["answer"]).strip(), True
        except json.JSONDecodeError:
            pass
    return raw.strip(), False


class RAG:
    def __init__(self, cfg: SystemConfig, data: Dataset, llm: LLM):
        self.cfg, self.data, self.llm = cfg, data, llm
        r = cfg.retrieval
        self.index = Index(data.corpus, r.chunk_sentences, r.chunk_overlap)
        # decoys for the conflict injection: answers of other questions, fixed order
        self._answers = [e.answer for e in data.rag + data.agent]

    def retrieve(self, ex: Example, case_seed: int) -> tuple[list[Chunk], bool]:
        r = self.cfg.retrieval
        chunks = self.index.search(ex.question, r.top_k)
        injected = False
        if r.inject_conflict:
            decoys = [a for a in self._answers if a.lower() != ex.answer.lower()
                      and a.lower() not in ("yes", "no")]
            decoy = decoys[case_seed % len(decoys)]
            # prefer a retrieved chunk carrying the answer; else the gold paragraph itself
            cands = chunks + [Chunk(t, 0, list(self.data.corpus[t].sentences)) for t in ex.gold_titles]
            for c in cands:
                bad = make_conflict(c, ex.answer, decoy)
                if bad:
                    chunks = [bad] + chunks[: max(r.top_k - 1, 0)] if r.top_k > 1 else [bad]
                    injected = True
                    break
        if r.shuffle:
            rng = random.Random(case_seed)
            chunks = chunks[:]
            rng.shuffle(chunks)
        return chunks, injected

    def answer(self, ex: Example, case_seed: int, tag: str) -> RagResult:
        chunks, injected = self.retrieve(ex, case_seed)
        user = f"Context:\n{format_context(chunks)}\nQuestion: {ex.question}"
        m = self.cfg.model
        resp = self.llm.chat(model=m.name, system=self.cfg.rag_system,
                             messages=[{"role": "user", "content": user}], tools=None,
                             temperature=m.temperature, max_tokens=m.max_tokens,
                             seed=case_seed, tag=tag)
        ans, ok = parse_answer(resp.text)
        return RagResult(id=ex.id, answer=ans, raw=resp.text, parsed_json=ok,
                         stop_reason=resp.stop_reason,
                         retrieved=[f"{c.title}#{c.sent_start}" for c in chunks], injected=injected)
