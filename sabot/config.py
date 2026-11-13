"""The system under test, as data. Mutants are functions SystemConfig -> SystemConfig.

Everything a mutant may touch lives here: prompt text, retrieval parameters, model
parameters and the agent's tool wiring. The baseline values below were fixed before
any mutant was run and are not tuned against the results.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# RAG prompt. Each mutant-targeted span is a separate sentence so prompt
# mutants are exact, checkable string edits (see mutants.py).
# --------------------------------------------------------------------------
GROUNDING = ("You must answer using only the numbered context passages. "
             "If the passages do not contain the answer, answer \"I don't know\".")
CONSTRAINT = "Give only the shortest span that answers the question: a name, date, number, or yes/no."
FORMAT_SPEC = ('You must respond with a single JSON object and nothing else: '
               '{"answer": "<short answer>", "passages": [<numbers of the passages you used>]}')

FEWSHOT_1 = """Example 1
Context:
[1] Lake Tahoma: Lake Tahoma is a reservoir in Ellison County. It was created in 1931 when the Marlow Dam was completed.
[2] Marlow Dam: The Marlow Dam is a concrete arch dam designed by engineer Ruth Okafor.
Question: Who designed the dam that created Lake Tahoma?
{"answer": "Ruth Okafor", "passages": [1, 2]}"""

FEWSHOT_2 = """Example 2
Context:
[1] The Gray Finches: The Gray Finches are an indie band formed in Leeds in 2004.
[2] Coral Static: Coral Static is a band formed in Perth in 2011.
Question: Which band was formed first, The Gray Finches or Coral Static?
{"answer": "The Gray Finches", "passages": [1, 2]}"""

RAG_SYSTEM = "\n".join([
    "You are a question-answering assistant for a document library.",
    GROUNDING,
    CONSTRAINT,
    FORMAT_SPEC,
    "",
    FEWSHOT_1,
    "",
    FEWSHOT_2,
])

AGENT_SYSTEM = "\n".join([
    "You are a research assistant with access to a document library through tools.",
    "Use the search tool to find relevant documents and the read tool to read them.",
    "Questions may need facts from two different documents.",
    "When you know the answer, call the finish tool with the shortest answer span "
    "(a name, date, number, or yes/no).",
    "Do not use documents you were denied access to.",
])

TOOL_SPECS = {
    "search": {"name": "search",
               "description": "Search the document library. Returns up to 5 document titles, each with its first sentence.",
               "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                              "required": ["query"]}},
    "read": {"name": "read",
             "description": "Read the full text of a document by its exact title.",
             "parameters": {"type": "object", "properties": {"title": {"type": "string"}},
                            "required": ["title"]}},
    "finish": {"name": "finish",
               "description": "Submit the final answer.",
               "parameters": {"type": "object", "properties": {"answer": {"type": "string"}},
                              "required": ["answer"]}},
}


@dataclass
class ModelConfig:
    name: str
    temperature: float = 0.0
    max_tokens: int = 256


@dataclass
class RetrievalConfig:
    top_k: int = 4
    chunk_sentences: int = 4
    chunk_overlap: int = 2  # sentences shared by consecutive chunks
    shuffle: bool = False
    inject_conflict: bool = False


@dataclass
class AgentConfig:
    tools: list[str] = field(default_factory=lambda: ["search", "read", "finish"])
    max_steps: int = 8
    max_tokens: int = 512
    authorize: bool = True  # framework calls authorize(user, title) before returning any document
    # tool name -> handler name; baseline is the identity wiring
    bindings: dict[str, str] = field(default_factory=lambda: {"search": "search", "read": "read"})
    erroring_tools: list[str] = field(default_factory=list)


@dataclass
class SystemConfig:
    model: ModelConfig
    rag_system: str = RAG_SYSTEM
    agent_system: str = AGENT_SYSTEM
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def clone(self) -> "SystemConfig":
        return copy.deepcopy(self)


# Model tiers per backend: (system model, one-tier-down model).
MODEL_TIERS = {
    "ollama": ("qwen2.5:3b-instruct", "qwen2.5:1.5b-instruct"),
    "anthropic": ("claude-haiku-5-5", "claude-haiku-4-5"),
    "fake": ("fake-large", "fake-small"),
}


def baseline(backend: str) -> SystemConfig:
    return SystemConfig(model=ModelConfig(name=MODEL_TIERS[backend][0]))
