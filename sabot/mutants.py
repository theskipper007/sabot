"""The mutant catalogue. Each mutant is a pure function SystemConfig -> SystemConfig.

Rules every mutant follows:
  * It is broken by construction: it removes or corrupts something the system was
    explicitly built to do (see `why`), independent of whether any eval notices.
  * It applies or raises. A prompt edit whose target text is missing raises
    MutantNotApplied instead of silently producing an unchanged (equivalent) system.
  * It touches exactly the sub-systems listed in `targets`; the other sub-suite is not rerun.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import (CONSTRAINT, FEWSHOT_2, FORMAT_SPEC, GROUNDING, MODEL_TIERS, SystemConfig)


class MutantNotApplied(Exception):
    pass


@dataclass(frozen=True)
class Mutant:
    id: str
    surface: str        # prompt | retrieval | model | agent
    name: str
    why: str            # why this is a real defect
    targets: tuple      # ("rag",), ("agent",), or ("rag", "agent")
    fn: Callable[[SystemConfig, str], SystemConfig]
    na_backends: tuple = ()  # backends on which the mutant cannot be expressed
    na_reason: str = ""

    def apply(self, cfg: SystemConfig, backend: str) -> SystemConfig:
        if backend in self.na_backends:
            raise MutantNotApplied(f"{self.id}: not applicable on {backend}: {self.na_reason}")
        out = self.fn(cfg.clone(), backend)
        if out == cfg:
            raise MutantNotApplied(f"{self.id}: produced an identical config")
        return out


def _edit(text: str, old: str, new: str, count: int | None = 1) -> str:
    n = text.count(old)
    if n == 0 or (count is not None and n != count):
        raise MutantNotApplied(f"target {old[:40]!r} found {n}x, expected {count}")
    return text.replace(old, new)


def _p(fn):
    def wrap(cfg: SystemConfig, backend: str) -> SystemConfig:
        cfg.rag_system = fn(cfg.rag_system)
        return cfg
    return wrap


def _set(path: str, value):
    def wrap(cfg: SystemConfig, backend: str) -> SystemConfig:
        obj = cfg
        *head, last = path.split(".")
        for h in head:
            obj = getattr(obj, h)
        setattr(obj, last, value(getattr(obj, last)) if callable(value) else value)
        return cfg
    return wrap


def _downgrade(cfg: SystemConfig, backend: str) -> SystemConfig:
    cfg.model.name = MODEL_TIERS[backend][1]
    return cfg


def _swap(cfg: SystemConfig, backend: str) -> SystemConfig:
    cfg.agent.bindings = {"search": "read", "read": "search"}
    return cfg


def _remove_tool(cfg: SystemConfig, backend: str) -> SystemConfig:
    cfg.agent.tools = [t for t in cfg.agent.tools if t != "search"]
    return cfg


RAG, AGENT, BOTH = ("rag",), ("agent",), ("rag", "agent")

CATALOGUE: list[Mutant] = [
    # ---- prompt -----------------------------------------------------------
    Mutant("P1", "prompt", "delete a constraint word ('only' in the answer-length rule)",
           "the length constraint is what keeps answers to a bare span", RAG,
           _p(lambda s: _edit(s, CONSTRAINT, CONSTRAINT.replace("Give only the", "Give the")))),
    Mutant("P2", "prompt", "remove the grounding instruction",
           "without it the model may answer from parametric memory and never abstain", RAG,
           _p(lambda s: _edit(s, GROUNDING + "\n", ""))),
    Mutant("P3", "prompt", "drop the output-format spec",
           "downstream code expects JSON with cited passages", RAG,
           _p(lambda s: _edit(s, FORMAT_SPEC + "\n", ""))),
    Mutant("P4", "prompt", "delete a few-shot example (the comparison example)",
           "removes the only demonstration of comparison questions", RAG,
           _p(lambda s: _edit(s, "\n\n" + FEWSHOT_2, ""))),
    Mutant("P5", "prompt", "soften must -> should",
           "turns hard requirements (grounding, JSON) into suggestions", RAG,
           _p(lambda s: _edit(s, "must", "should", count=None))),
    # ---- retrieval ----------------------------------------------------------
    Mutant("R1", "retrieval", "truncate retrieval to top-1",
           "HotpotQA questions need two documents; top-1 cannot hold both", RAG,
           _set("retrieval.top_k", 1)),
    Mutant("R2", "retrieval", "shuffle rank order",
           "throws away the ranker's ordering signal", RAG,
           _set("retrieval.shuffle", True)),
    Mutant("R3", "retrieval", "inject a wrong-year/wrong-tenant document",
           "a stale (2009) copy from another tenant, with the answer replaced by a decoy, is ranked first", RAG,
           _set("retrieval.inject_conflict", True)),
    Mutant("R4", "retrieval", "halve chunk overlap (2 -> 1 sentences)",
           "facts spanning a window boundary can no longer land in one chunk", RAG,
           _set("retrieval.chunk_overlap", lambda v: v // 2)),
    # ---- model --------------------------------------------------------------
    Mutant("M1", "model", "downgrade the model one tier",
           "ships a weaker/older model than the one the system was validated on", BOTH, _downgrade),
    Mutant("M2", "model", "temperature 0 -> 1",
           "makes a deterministic system stochastic", BOTH,
           _set("model.temperature", 1.0), na_backends=("anthropic",),
           na_reason="Claude Haiku 5.5 rejects non-default temperature (and API version 2026-10-21 "
                     "rejects the field on every model); the baseline already samples at the API default"),
    Mutant("M3", "model", "cut max_tokens by 40%",
           "less room for the answer; long answers truncate", BOTH,
           lambda c, b: (setattr(c.model, "max_tokens", int(c.model.max_tokens * 0.6)),
                         setattr(c.agent, "max_tokens", int(c.agent.max_tokens * 0.6)), c)[-1]),
    # ---- agent --------------------------------------------------------------
    Mutant("A1", "agent", "skip the authorisation tool call",
           "restricted documents reach the model: an access-control bypass", AGENT,
           _set("agent.authorize", False)),
    Mutant("A2", "agent", "swap two tool calls (search <-> read handlers)",
           "a wiring bug: each tool name runs the other tool's handler", AGENT, _swap),
    Mutant("A3", "agent", "make a tool return an error (read -> 503)",
           "the agent can no longer read documents", AGENT,
           _set("agent.erroring_tools", ["read"])),
    Mutant("A4", "agent", "remove a tool (search)",
           "the agent can only open documents whose exact title it guesses", AGENT, _remove_tool),
]

BY_ID = {m.id: m for m in CATALOGUE}
