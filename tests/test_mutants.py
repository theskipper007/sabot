"""Mutation application: every mutant applies, changes exactly what it claims, and
never silently degrades to an equivalent (unchanged) system."""
import pytest

from sabot.agent import Agent
from sabot.config import CONSTRAINT, FEWSHOT_2, FORMAT_SPEC, GROUNDING, baseline
from sabot.data import Dataset, Example, Paragraph
from sabot.llm import LLM
from sabot.mutants import BY_ID, CATALOGUE, MutantNotApplied, _edit
from sabot.rag import RAG, chunk_paragraph, make_conflict, parse_answer


@pytest.fixture
def base():
    return baseline("fake")


def test_catalogue_has_the_16_documented_mutants():
    assert [m.id for m in CATALOGUE] == ["P1", "P2", "P3", "P4", "P5", "R1", "R2", "R3", "R4",
                                         "M1", "M2", "M3", "A1", "A2", "A3", "A4"]
    from collections import Counter
    assert Counter(m.surface for m in CATALOGUE) == {"prompt": 5, "retrieval": 4, "model": 3, "agent": 4}


@pytest.mark.parametrize("m", CATALOGUE, ids=lambda m: m.id)
def test_every_mutant_changes_the_config(m, base):
    out = m.apply(base, "fake")
    assert out != base
    assert base == baseline("fake"), "apply() must not mutate the baseline in place"


def test_prompt_mutants_are_exact_edits(base):
    s = base.rag_system
    p1 = BY_ID["P1"].apply(base, "fake").rag_system
    assert "Give only the shortest" in s and "Give only the shortest" not in p1 and "Give the shortest" in p1
    p2 = BY_ID["P2"].apply(base, "fake").rag_system
    assert GROUNDING in s and GROUNDING not in p2 and "I don't know" not in p2
    p3 = BY_ID["P3"].apply(base, "fake").rag_system
    assert FORMAT_SPEC not in p3 and CONSTRAINT in p3
    p4 = BY_ID["P4"].apply(base, "fake").rag_system
    assert FEWSHOT_2 not in p4 and "Example 1" in p4
    p5 = BY_ID["P5"].apply(base, "fake").rag_system
    assert "must" not in p5 and p5.count("should") == s.count("must") == 2
    # each prompt mutant changes nothing outside the RAG prompt
    for mid in ("P1", "P2", "P3", "P4", "P5"):
        out = BY_ID[mid].apply(base, "fake")
        out.rag_system = base.rag_system
        assert out == base


def test_edit_raises_when_target_missing():
    with pytest.raises(MutantNotApplied):
        _edit("abc", "zzz", "y")
    with pytest.raises(MutantNotApplied):
        _edit("aa", "a", "b", count=1)  # ambiguous target


def test_config_mutants(base):
    assert BY_ID["R1"].apply(base, "fake").retrieval.top_k == 1
    assert BY_ID["R2"].apply(base, "fake").retrieval.shuffle is True
    assert BY_ID["R3"].apply(base, "fake").retrieval.inject_conflict is True
    assert BY_ID["R4"].apply(base, "fake").retrieval.chunk_overlap == base.retrieval.chunk_overlap // 2 == 1
    assert BY_ID["M1"].apply(base, "fake").model.name == "fake-small"
    assert BY_ID["M1"].apply(baseline("anthropic"), "anthropic").model.name == "claude-haiku-4-5"
    assert BY_ID["M2"].apply(base, "fake").model.temperature == 1.0
    m3 = BY_ID["M3"].apply(base, "fake")
    assert m3.model.max_tokens == int(256 * 0.6) and m3.agent.max_tokens == int(512 * 0.6)
    assert BY_ID["A1"].apply(base, "fake").agent.authorize is False
    assert BY_ID["A2"].apply(base, "fake").agent.bindings == {"search": "read", "read": "search"}
    assert BY_ID["A3"].apply(base, "fake").agent.erroring_tools == ["read"]
    assert "search" not in BY_ID["A4"].apply(base, "fake").agent.tools


def test_temperature_mutant_is_na_on_anthropic():
    with pytest.raises(MutantNotApplied, match="not applicable"):
        BY_ID["M2"].apply(baseline("anthropic"), "anthropic")


# ---- behaviour of the mutated system (no model calls needed) -------------
def tiny_data():
    paras = {
        "Alpha": Paragraph("Alpha", ["Alpha was founded in 1901.", " It is in Spain.", " Its mayor is Bo.", " x.", " y."]),
        "Beta": Paragraph("Beta", ["Beta is a river.", " It flows through Alpha."]),
        "Secret": Paragraph("Secret", ["Secret is a restricted memo about Alpha."], restricted=True),
    }
    ex = Example("q1", "When was Alpha founded?", "1901", "bridge", "easy", ["Alpha"],
                 [(t, p.sentences) for t, p in paras.items()])
    ex2 = Example("q2", "Which river flows through Alpha?", "Beta", "bridge", "easy", ["Beta"], [])
    return Dataset(rag=[ex], agent=[ex2], corpus=paras)


def test_chunk_overlap():
    p = Paragraph("T", [f"s{i}." for i in range(8)])
    c2 = chunk_paragraph(p, 4, 2)
    c1 = chunk_paragraph(p, 4, 1)
    assert [c.sent_start for c in c2] == [0, 2, 4]
    assert [c.sent_start for c in c1] == [0, 3, 6]
    assert all(len(c.sentences) <= 4 for c in c1 + c2)
    assert c1[-1].sentences == ["s6.", "s7."]
    with pytest.raises(ValueError):
        chunk_paragraph(p, 4, 4)


def test_conflict_injection_replaces_answer():
    from sabot.rag import Chunk
    c = Chunk("Alpha", 0, ["Alpha was founded in 1901."])
    bad = make_conflict(c, "1901", "1877")
    assert "1877" in bad.text and "1901" not in bad.text and "2009" in bad.title
    assert make_conflict(c, "yes", "no") is None


def test_retrieval_mutants_change_what_the_model_sees():
    d = tiny_data()
    llm = LLM("fake")
    base = baseline("fake")
    b_chunks, _ = RAG(base, d, llm).retrieve(d.rag[0], 1)
    r1, _ = RAG(BY_ID["R1"].apply(base, "fake"), d, llm).retrieve(d.rag[0], 1)
    assert len(r1) == 1 and len(b_chunks) > 1
    r3, inj = RAG(BY_ID["R3"].apply(base, "fake"), d, llm).retrieve(d.rag[0], 1)
    assert inj and "archived" in r3[0].title and "1901" not in r3[0].text
    shuffled = {tuple(c.title + str(c.sent_start) for c in
                      RAG(BY_ID["R2"].apply(base, "fake"), d, llm).retrieve(d.rag[0], s)[0]) for s in range(20)}
    assert len(shuffled) > 1  # shuffle actually permutes
    assert all(sorted(x) == sorted(c.title + str(c.sent_start) for c in b_chunks) for x in shuffled)


def test_agent_mutants_change_tool_behaviour():
    from sabot.agent import Trace
    d = tiny_data()
    llm = LLM("fake")
    base = baseline("fake")
    ag = Agent(base, d, llm)
    tr = Trace()
    assert "ACCESS DENIED" in ag._dispatch("read", {"title": "Secret"}, tr)
    assert tr.authorize_calls == 1 and tr.restricted_leaked == []
    assert "Secret" not in ag._dispatch("search", {"query": "restricted memo"}, tr)

    a1 = Agent(BY_ID["A1"].apply(base, "fake"), d, llm)
    tr = Trace()
    assert "restricted memo" in a1._dispatch("read", {"title": "Secret"}, tr)
    assert tr.authorize_calls == 0 and tr.restricted_leaked == ["Secret"]

    a2 = Agent(BY_ID["A2"].apply(base, "fake"), d, llm)
    assert a2._dispatch("read", {"title": "Alpha"}, Trace()).startswith("- ")      # ran search
    assert "No document titled" in a2._dispatch("search", {"query": "founded"}, Trace())  # ran read

    a3 = Agent(BY_ID["A3"].apply(base, "fake"), d, llm)
    assert a3._dispatch("read", {"title": "Alpha"}, Trace()).startswith("ERROR 503")

    a4 = Agent(BY_ID["A4"].apply(base, "fake"), d, llm)
    assert "unknown tool" in a4._dispatch("search", {"query": "x"}, Trace())


def test_parse_answer():
    assert parse_answer('{"answer": "Ruth Okafor", "passages": [1]}') == ("Ruth Okafor", True)
    assert parse_answer('Sure! {"answer": "1901"} done') == ("1901", True)
    assert parse_answer("The answer is 1901.") == ("The answer is 1901.", False)
