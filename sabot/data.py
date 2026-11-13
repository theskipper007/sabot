"""Dataset: HotpotQA (distractor setting), validation split.

Source : https://hotpotqa.github.io/  (Yang et al., EMNLP 2018)
Mirror : https://huggingface.co/datasets/hotpotqa/hotpot_qa  (config "distractor")
License: CC BY-SA 4.0

We take the FIRST N rows of the validation split, in the order the dataset
publishes them. No filtering, no cherry-picking. Rows [0, n_rag) feed the RAG
sub-suite, rows [n_rag, n_rag + n_agent) feed the agent sub-suite.

The retrieval corpus is the union of every context paragraph (gold + the 8
distractors HotpotQA ships per question) across all selected rows, so each
question is retrieved against ~10x(n_rag+n_agent) paragraphs, not just its own 10.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DATASET_ID = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_SPLIT = "validation"
DATASET_LICENSE = "CC BY-SA 4.0"
DATASET_URL = "https://huggingface.co/datasets/hotpotqa/hotpot_qa"
ROWS_API = "https://datasets-server.huggingface.co/rows"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "hotpotqa_distractor_validation_head.jsonl"


@dataclass
class Example:
    id: str
    question: str
    answer: str
    qtype: str  # "bridge" | "comparison"
    level: str
    gold_titles: list[str]
    context: list[tuple[str, list[str]]]  # (title, sentences)


@dataclass
class Paragraph:
    title: str
    sentences: list[str]
    restricted: bool = False

    @property
    def text(self) -> str:
        return "".join(self.sentences).strip()


@dataclass
class Dataset:
    rag: list[Example]
    agent: list[Example]
    corpus: dict[str, Paragraph] = field(default_factory=dict)
    sha256: str = ""
    rows: tuple[int, int] = (0, 0)


def fetch(n: int, out: Path = DATA_FILE) -> Path:
    """Download the first n validation rows via the HF datasets-server API."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    offset = 0
    while offset < n:
        length = min(100, n - offset)
        url = (f"{ROWS_API}?dataset={DATASET_ID.replace('/', '%2F')}&config={DATASET_CONFIG}"
               f"&split={DATASET_SPLIT}&offset={offset}&length={length}")
        req = urllib.request.Request(url, headers={"User-Agent": "sabot/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.load(r)
        for item in payload["rows"]:
            row = item["row"]
            rows.append({
                "row_idx": item["row_idx"],
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row["type"],
                "level": row["level"],
                "supporting_titles": sorted(set(row["supporting_facts"]["title"])),
                "context": [[t, s] for t, s in zip(row["context"]["title"], row["context"]["sentences"])],
            })
        offset += length
    rows.sort(key=lambda r: r["row_idx"])
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def _restricted(title: str) -> bool:
    # Deterministic ~30% of non-gold paragraphs belong to a restricted collection.
    # This ACL is sabot's construction on top of HotpotQA, used only by the agent.
    return int(hashlib.sha256(title.encode("utf-8")).hexdigest(), 16) % 10 < 3


def load(n_rag: int, n_agent: int, path: Path = DATA_FILE) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m sabot fetch-data")
    raw = path.read_bytes()
    lines = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    need = n_rag + n_agent
    if len(lines) < need:
        raise ValueError(f"need {need} rows, data file has {len(lines)}; rerun fetch-data")
    exs = [Example(id=r["id"], question=r["question"], answer=r["answer"], qtype=r["type"],
                   level=r["level"], gold_titles=r["supporting_titles"],
                   context=[(t, s) for t, s in r["context"]]) for r in lines[:need]]
    gold = {t for e in exs for t in e.gold_titles}
    corpus: dict[str, Paragraph] = {}
    for e in exs:
        for title, sents in e.context:
            if title not in corpus:
                corpus[title] = Paragraph(title, list(sents),
                                          restricted=(title not in gold) and _restricted(title))
    return Dataset(rag=exs[:n_rag], agent=exs[n_rag:need], corpus=corpus,
                   sha256=hashlib.sha256(raw).hexdigest(), rows=(0, need))
