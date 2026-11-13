"""System 2: a simple tool-using agent (model-driven tool loop).

Tools exposed to the model: search(query), read(title), finish(answer).
The framework (not the model) enforces access control: before any document text or
snippet is returned, it calls authorize(user, title) and withholds restricted
documents. This is the "authorisation tool call" the skip-auth mutant removes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .bm25 import BM25
from .config import TOOL_SPECS, SystemConfig
from .data import Dataset, Example
from .llm import LLM

USER_ID = "analyst-42"  # every eval case runs as an ordinary analyst without restricted access


@dataclass
class Trace:
    calls: list[tuple[str, dict]] = field(default_factory=list)  # (tool, args) as the model issued them
    authorize_calls: int = 0
    restricted_leaked: list[str] = field(default_factory=list)   # restricted titles whose text reached the model
    tool_errors: int = 0


@dataclass
class AgentResult:
    id: str
    answer: str
    finished: bool
    steps: int
    stop_reasons: list[str]
    trace: Trace


class Tools:
    def __init__(self, data: Dataset):
        self.data = data
        self.titles = list(data.corpus)
        self.bm25 = BM25([f"{t}. {data.corpus[t].text}" for t in self.titles])
        self._lower = {t.lower(): t for t in self.titles}

    def authorize(self, user: str, title: str) -> bool:
        return not self.data.corpus[title].restricted

    def search(self, query: str) -> list[str]:
        return [self.titles[i] for i in self.bm25.top(query, 5)]

    def resolve(self, title: str) -> str | None:
        return self._lower.get(title.strip().lower())


class Agent:
    def __init__(self, cfg: SystemConfig, data: Dataset, llm: LLM):
        self.cfg, self.data, self.llm = cfg, data, llm
        self.tools = Tools(data)

    # --- tool handlers (the framework side) --------------------------------
    def _gate(self, title: str, tr: Trace) -> bool:
        """True if the document may be shown. Baseline: authorize() is called every time."""
        if self.cfg.agent.authorize:
            tr.authorize_calls += 1
            return self.tools.authorize(USER_ID, title)
        return True

    def _h_search(self, arg: str, tr: Trace) -> str:
        lines = []
        for t in self.tools.search(arg):
            if not self._gate(t, tr):
                continue  # restricted docs are silently filtered from results
            if self.data.corpus[t].restricted:
                tr.restricted_leaked.append(t)
            first = self.data.corpus[t].sentences[0].strip() if self.data.corpus[t].sentences else ""
            lines.append(f"- {t}: {first}")
        return "\n".join(lines) if lines else "No results."

    def _h_read(self, arg: str, tr: Trace) -> str:
        t = self.tools.resolve(arg)
        if t is None:
            return f"No document titled '{arg}'."
        if not self._gate(t, tr):
            return f"ACCESS DENIED: '{t}' is in a restricted collection."
        if self.data.corpus[t].restricted:
            tr.restricted_leaked.append(t)
        return f"{t}\n{self.data.corpus[t].text}"

    def _dispatch(self, name: str, args: dict, tr: Trace) -> str:
        a = self.cfg.agent
        if name not in a.tools or name == "finish":
            return f"ERROR: unknown tool '{name}'."
        if name in a.erroring_tools:
            tr.tool_errors += 1
            return "ERROR 503: document service unavailable. Try again later."
        handler = a.bindings.get(name, name)
        # The model's single string argument is passed positionally, whatever its key,
        # so a swapped binding feeds a query to read() and a title to search().
        arg = str(args.get("query", args.get("title", next(iter(args.values()), "")))) if args else ""
        return getattr(self, f"_h_{handler}")(arg, tr)

    # --- loop ---------------------------------------------------------------
    def answer(self, ex: Example, case_seed: int, tag: str) -> AgentResult:
        a, m = self.cfg.agent, self.cfg.model
        tools = [TOOL_SPECS[t] for t in a.tools]
        msgs: list[dict] = [{"role": "user", "content": ex.question}]
        tr = Trace()
        stops = []
        for step in range(1, a.max_steps + 1):
            resp = self.llm.chat(model=m.name, system=self.cfg.agent_system, messages=msgs, tools=tools,
                                 temperature=m.temperature, max_tokens=a.max_tokens,
                                 seed=case_seed + step, tag=tag)
            stops.append(resp.stop_reason)
            if not resp.tool_calls:
                # no tool call: the turn ends; the free text is NOT accepted as an answer
                return AgentResult(ex.id, "", False, step, stops, tr)
            msgs.append({"role": "assistant", "content": resp.text,
                         "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in resp.tool_calls]})
            for c in resp.tool_calls:
                tr.calls.append((c.name, c.args))
                if c.name == "finish" and "finish" in a.tools:
                    return AgentResult(ex.id, str(c.args.get("answer", "")).strip(), True, step, stops, tr)
            for c in resp.tool_calls:
                msgs.append({"role": "tool", "tool_call_id": c.id, "name": c.name,
                             "content": self._dispatch(c.name, c.args, tr)})
        return AgentResult(ex.id, "", False, a.max_steps, stops, tr)
