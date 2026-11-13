"""LLM backends behind one tiny interface, plus an on-disk response cache.

Neutral message format used by the systems:
    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "args"}]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}

Backends:
    ollama     local models over http://localhost:11434 (temperature, seed, num_predict)
    anthropic  Claude Messages API via the official SDK (optional dependency)
    fake       deterministic offline stub used by --dry-run; never produces real numbers
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "llm"

# USD per 1M tokens (input, output). Ollama is local: $0.
# Anthropic prices as published 2026-11 (claude-api skill table, cached 2026-11-03).
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Response:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "stop"  # "stop" | "length" | "tool_use"
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    cached: bool = False


# --dry-run cost estimate: fake models stand in for these, and the fake backend counts
# chars/4, so scale by 1.3 (Haiku 5.5's tokenizer counts ~30% more) to stay pessimistic.
DRY_RUN_PRICE_AS = {"fake-large": "claude-haiku-5-5", "fake-small": "claude-haiku-4-5"}
DRY_RUN_TOKEN_FUDGE = 1.3


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    est_usd: float = 0.0  # dry-run only: what the same calls would cost on Anthropic
    by_model: dict = field(default_factory=dict)

    def add(self, r: Response, backend: str, model: str):
        self.calls += 1
        if r.cached:
            self.cache_hits += 1
            return  # a cache hit costs nothing on this run
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        bm = self.by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
        bm["calls"] += 1
        bm["input_tokens"] += r.input_tokens
        bm["output_tokens"] += r.output_tokens
        if backend == "anthropic":
            pin, pout = PRICES.get(model, (0.0, 0.0))
            c = r.input_tokens / 1e6 * pin + r.output_tokens / 1e6 * pout
            self.usd += c
            bm["usd"] += c
        elif backend == "fake":
            pin, pout = PRICES[DRY_RUN_PRICE_AS.get(model, "claude-haiku-5-5")]
            self.est_usd += DRY_RUN_TOKEN_FUDGE * (r.input_tokens / 1e6 * pin + r.output_tokens / 1e6 * pout)


class LLM:
    """Wraps a backend with caching and usage accounting. Thread-safe."""

    def __init__(self, backend: str, cache: bool = True, host: str = "http://localhost:11434",
                 max_usd: float | None = None):
        self.backend = backend
        self.max_usd = max_usd
        self.cache = cache and backend != "fake"
        self.host = host
        self.usage = Usage()
        self._lock = threading.Lock()
        self._client = None
        if backend == "anthropic":
            import anthropic  # optional dependency
            # Pin the endpoint: an inherited ANTHROPIC_BASE_URL (e.g. from a dev tool's own
            # environment) must not silently redirect the run. Override explicitly if needed.
            self._client = anthropic.Anthropic(
                base_url=os.environ.get("SABOT_ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
                api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # ---- public -----------------------------------------------------------
    def chat(self, *, model: str, system: str, messages: list[dict], tools: list[dict] | None,
             temperature: float, max_tokens: int, seed: int, tag: str) -> Response:
        req = {"backend": self.backend, "model": model, "system": system, "messages": messages,
               "tools": tools or [], "temperature": temperature, "max_tokens": max_tokens,
               "seed": seed, "tag": tag}
        key = hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = CACHE_DIR / key[:2] / f"{key}.json"
        if self.cache and path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            r = Response(text=d["text"], tool_calls=[ToolCall(**t) for t in d["tool_calls"]],
                         stop_reason=d["stop_reason"], input_tokens=d["input_tokens"],
                         output_tokens=d["output_tokens"], model=d["model"], cached=True)
        else:
            if self.max_usd is not None and self.usage.usd >= self.max_usd:
                raise BudgetExceeded(f"spend ${self.usage.usd:.4f} reached the ${self.max_usd} cap; aborting")
            r = getattr(self, f"_{self.backend}")(req)
            if self.cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "text": r.text, "tool_calls": [t.__dict__ for t in r.tool_calls],
                    "stop_reason": r.stop_reason, "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens, "model": r.model}, ensure_ascii=False),
                    encoding="utf-8")
        with self._lock:
            self.usage.add(r, self.backend, model)
        return r

    def model_info(self, model: str) -> dict:
        if self.backend == "ollama":
            try:
                with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as f:
                    tags = json.load(f)
                for m in tags.get("models", []):
                    if m["name"] == model or m["model"] == model:
                        return {"name": m["name"], "digest": m["digest"],
                                "details": m.get("details", {})}
            except Exception as e:  # noqa: BLE001
                return {"name": model, "error": str(e)}
            return {"name": model, "error": "not pulled"}
        return {"name": model}

    def server_version(self) -> str:
        if self.backend == "ollama":
            try:
                with urllib.request.urlopen(f"{self.host}/api/version", timeout=10) as f:
                    return json.load(f).get("version", "")
            except Exception:  # noqa: BLE001
                return ""
        if self.backend == "anthropic":
            import anthropic
            return f"anthropic-python {anthropic.__version__}"
        return "fake"

    # ---- backends ---------------------------------------------------------
    def _ollama(self, req: dict) -> Response:
        msgs = [{"role": "system", "content": req["system"]}]
        for m in req["messages"]:
            if m["role"] == "assistant":
                om = {"role": "assistant", "content": m.get("content", "")}
                if m.get("tool_calls"):
                    om["tool_calls"] = [{"function": {"name": t["name"], "arguments": t["args"]}}
                                        for t in m["tool_calls"]]
                msgs.append(om)
            elif m["role"] == "tool":
                msgs.append({"role": "tool", "content": m["content"], "tool_name": m["name"]})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        body = {"model": req["model"], "messages": msgs, "stream": False,
                "options": {"temperature": req["temperature"], "seed": req["seed"],
                            "num_predict": req["max_tokens"], "num_ctx": 8192}}
        if req["tools"]:
            body["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
                for t in req["tools"]]
        data = json.dumps(body).encode()
        hreq = urllib.request.Request(f"{self.host}/api/chat", data=data,
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(hreq, timeout=600) as f:
            d = json.load(f)
        msg = d.get("message", {})
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append(ToolCall(id=f"call_{i}", name=fn.get("name", ""), args=args))
        stop = "length" if d.get("done_reason") == "length" else ("tool_use" if calls else "stop")
        return Response(text=msg.get("content", "") or "", tool_calls=calls, stop_reason=stop,
                        input_tokens=d.get("prompt_eval_count", 0) or 0,
                        output_tokens=d.get("eval_count", 0) or 0, model=d.get("model", req["model"]))

    def _anthropic(self, req: dict) -> Response:  # pragma: no cover - needs credentials
        msgs: list[dict] = []
        for m in req["messages"]:
            if m["role"] == "assistant":
                content = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for t in m.get("tool_calls") or []:
                    content.append({"type": "tool_use", "id": t["id"], "name": t["name"], "input": t["args"]})
                msgs.append({"role": "assistant", "content": content})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                    msgs[-1]["content"].append(block)
                else:
                    msgs.append({"role": "user", "content": [block]})
            else:
                msgs.append({"role": "user", "content": m["content"]})
        # No sampling parameters: Claude Haiku 5.5 accepts only the API defaults, and API
        # version 2026-10-21 rejects `temperature` on every model. Both tiers therefore
        # sample at the API default; the temperature mutant is N/A on this backend.
        kw = dict(model=req["model"], max_tokens=req["max_tokens"], system=req["system"],
                  messages=msgs)
        if req["model"].startswith("claude-haiku-5"):
            # thinking would spend max_tokens on hidden reasoning; keep the request plain
            kw["thinking"] = {"type": "disabled"}
        if req["tools"]:
            kw["tools"] = [{"name": t["name"], "description": t["description"],
                            "input_schema": t["parameters"]} for t in req["tools"]]
        import anthropic
        for attempt in range(6):
            try:
                r = self._client.messages.create(**kw)
                break
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, args=dict(b.input)) for b in r.content if b.type == "tool_use"]
        stop = {"max_tokens": "length", "tool_use": "tool_use", "refusal": "refusal"}.get(r.stop_reason, "stop")
        return Response(text=text, tool_calls=calls, stop_reason=stop,
                        input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens, model=r.model)

    def _fake(self, req: dict) -> Response:
        """Offline stub. Answers with the first capitalised phrase of the first passage
        (RAG) or does search -> read -> finish (agent). Exists only to exercise the
        plumbing; its 'scores' mean nothing."""
        msgs = req["messages"]
        n_in = sum(len(str(m.get("content", ""))) for m in msgs) // 4 + len(req["system"]) // 4
        if not req["tools"]:
            ctx = msgs[-1]["content"]
            m = re.search(r"\[1\][^\n]*\n([^\n]+)", ctx)
            words = (m.group(1) if m else "unknown").split()[:3]
            return Response(text=json.dumps({"answer": " ".join(words), "passages": [1]}),
                            input_tokens=n_in, output_tokens=12, model="fake")
        names = {t["name"] for t in req["tools"]}
        n_tool = sum(1 for m in msgs if m["role"] == "tool")
        question = msgs[0]["content"]
        if n_tool == 0 and "search" in names:
            return Response(text="", tool_calls=[ToolCall("c0", "search", {"query": question})],
                            stop_reason="tool_use", input_tokens=n_in, output_tokens=10, model="fake")
        if n_tool == 1 and "read" in names:
            last = msgs[-1]["content"]
            m = re.search(r"^- (.+?):", last, re.M)
            title = m.group(1) if m else question.split()[0]
            return Response(text="", tool_calls=[ToolCall("c1", "read", {"title": title})],
                            stop_reason="tool_use", input_tokens=n_in, output_tokens=10, model="fake")
        last = msgs[-1]["content"] if msgs[-1]["role"] == "tool" else ""
        ans = " ".join(last.split()[:2]) or "unknown"
        return Response(text="", tool_calls=[ToolCall("c2", "finish", {"answer": ans})],
                        stop_reason="tool_use", input_tokens=n_in, output_tokens=10, model="fake")


def env_has_anthropic() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
