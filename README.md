# sabot

**Mutation testing for LLM eval suites.** Break a working LLM system on purpose, in ways
you know are wrong, rerun your eval suite unchanged, and count how many breakages it notices.
A mutant your suite doesn't notice is an *escape*. The escape rate measures what your
evals can't see.

sabot also measures the suite's **noise floor**: it reruns the unmutated suite *k* times
and reports the run-to-run σ and the **minimum detectable effect (MDE)**. That lets you
tell "the suite can't see this" apart from "the suite could see it, but it's buried in noise".

> **Status: research prototype, v0.1.** Repo: https://github.com/theskipper007/sabot
>
> It works end to end on Windows 11 / Python 3.12 with the Anthropic backend (`--dry-run`,
> the tests and one real run). The `ollama` backend is implemented but **untested**.
> If something doesn't work for you, please open an issue or tell the author, Arun MV.
> The results in `results.json` / `report.txt` come from one real run, described under
> [Results](#results). Read [Limits](#limits) before quoting them.

## What it does

```
working system ──► apply mutant ──► rerun the SAME eval suite ──► score drop vs noise floor ──► KILLED / ESCAPED
      │
      └──► rerun unmutated k times ──► σ_run, MDE
```

**Systems under test** (`sabot/rag.py`, `sabot/agent.py`):

* **RAG pipeline.** BM25 over sentence-window chunks (4 sentences, overlap 2) from a
  shared corpus, top-4 chunks, one LLM call, JSON answer with cited passages.
* **Tool-using agent.** A model-driven tool loop with `search(query)`, `read(title)` and
  `finish(answer)`, max 8 steps. The framework calls `authorize(user, title)` before it
  returns any document. About 30% of non-gold paragraphs belong to a restricted
  collection that the eval user may not see.

**Eval suite under test** (`sabot/evals.py`). It is deliberately the common kind, scored
on golden answers only:

* RAG case passes iff token-F1(prediction, gold) ≥ 0.5, using HotpotQA's normalisation
  and its exact-match rule for yes/no answers.
* Agent case passes iff the agent called `finish` within the step budget **and** F1 ≥ 0.5.

The suite does not check citations, abstention, tool traces or access control. That is
the point: sabot tests the suite, and this is a typical suite.

**Dataset.** [HotpotQA](https://hotpotqa.github.io/) (Yang et al., 2018), distractor
setting, validation split, from
[`hotpotqa/hotpot_qa`](https://huggingface.co/datasets/hotpotqa/hotpot_qa) on Hugging Face.
License: **CC BY-SA 4.0**. sabot takes the *first* rows in published order, with no
filtering: rows 0–59 for RAG and 60–79 for the agent. The retrieval corpus is every
context paragraph of those 80 questions (gold paragraphs plus HotpotQA's distractors),
deduplicated by title. `data/hotpotqa_distractor_validation_head.jsonl` holds rows 0–99
and is redistributed under the same license.

## Mutant catalogue

Each mutant is a pure function `SystemConfig → SystemConfig` (`sabot/mutants.py`). It
either applies or raises: if a prompt edit can't find its target text, it errors instead
of silently running the unmutated system. Each one is broken by construction: it removes
or corrupts something the system was built to do.

| id | surface | mutant | why it's a real defect |
|----|---------|--------|------------------------|
| P1 | prompt | delete a constraint word (`only` in "Give only the shortest span…") | the length rule keeps answers to a bare span |
| P2 | prompt | remove the grounding instruction | the model may answer from memory and never abstains |
| P3 | prompt | drop the output-format spec | downstream code expects JSON with cited passages |
| P4 | prompt | delete a few-shot example (the comparison one) | removes the only comparison-question demo |
| P5 | prompt | soften every `must` → `should` | hard requirements become suggestions |
| R1 | retrieval | truncate to top-1 | HotpotQA needs two documents |
| R2 | retrieval | shuffle rank order | discards the ranker's ordering |
| R3 | retrieval | inject a wrong-year/wrong-tenant document | a stale "2009 revision, tenant: northwind-legacy" copy of a gold-bearing chunk, answer swapped for another question's answer, ranked first |
| R4 | retrieval | halve chunk overlap (2 → 1 sentences) | facts crossing a window boundary no longer share a chunk |
| M1 | model | downgrade one tier (`claude-haiku-5-5` → `claude-haiku-4-5`) | ships a model the system wasn't validated on |
| M2 | model | temperature 0 → 1 | makes a deterministic system stochastic |
| M3 | model | cut max_tokens by 40% (RAG 256→153, agent 512→307) | less room; long outputs truncate |
| A1 | agent | skip the authorisation tool call | restricted documents reach the model |
| A2 | agent | swap two tool calls (`search` ↔ `read` handlers) | wiring bug: each name runs the other handler |
| A3 | agent | a tool returns an error (`read` → 503) | the agent can't read documents |
| A4 | agent | remove a tool (`search`) | the agent can only open titles it guesses |

**M2 is not applicable on the Anthropic backend.** Claude Haiku 5.5 accepts only default
sampling parameters, and API version 2026-10-21 rejects `temperature` on every model. The
baseline therefore already samples at the API default, so "0 → 1" can't be expressed.
sabot reports M2 as N/A instead of faking it. M2 does run on the `ollama` backend.

Each mutant also records a **witness**: evidence, independent of the suite, that it
actually changed behaviour. Examples are the share of answers that changed vs baseline
run 0, retrieval changes, non-JSON replies, truncations, restricted documents leaked,
tool errors and unfinished agent runs. An escaped mutant with a large witness is a real
behaviour change the suite missed. An escaped mutant with no witness would be an
equivalent mutant.

## Method: noise floor, MDE and the kill rule

All of this is computed per sub-suite (`rag`, `agent`) in `sabot/stats.py`.

* `n` = cases in the sub-suite. `s_1..s_k` = pass rate of *k* reruns of the **unmutated**
  system (default k = 5). `m_1..m_r` = pass rate of *r* mutant runs (default r = 1).
* **Δ = mean(s) − mean(m)**, in pass-rate points. Positive means the mutant scores lower.

**σ is run-to-run, not per-example.** σ_run is the sample SD (ddof = 1) of the k suite
scores: the same n cases rerun k times.

```
SE         = σ_run · sqrt(1/k + 1/r)                       unpaired: k baseline runs vs r mutant runs
critical   = t(1 − α/m, k−1) · SE                          m = number of sub-suites the mutant touches (Bonferroni)
KILLED     iff  Δ > critical  and  Δ > 0                   one-sided
MDE_run    = max( (t(1−α, k−1) + t(power, k−1)) · SE ,  1/n )
             α = 0.05, power = 0.80, one-sided, unpaired, df = k−1
```

The `1/n` floor is the suite's resolution: it can't register a change smaller than one
case. If the system is perfectly repeatable (σ_run = 0), any net drop of one case kills.

The MDE assumes the mutant has the same run-to-run variance as the baseline, and uses
Student-t quantiles because σ is estimated from only k runs.

**Case-sampling MDE (context only, not used to kill).** This is the smallest drop that
would generalise beyond these n cases:

```
MDE_sample = (z(1−α) + z(power)) · sqrt( 2·p·(1−p) / n )   per-example binomial SD, unpaired, one-sided, normal approx.
```

Here p is the mean baseline pass rate.

**Escapes are classified** as:

* `below_mde`: the drop was real but too small to detect (0 < Δ ≤ critical < MDE_run).
* `no_drop`: Δ ≤ 0.

`results.json` also gives, per mutant, a paired McNemar view against the per-case
baseline majority, and the number of cases that passed in all k baseline runs but failed
under the mutant.

## Run it

```bash
pip install -e ".[anthropic,dev]"
python -m sabot fetch-data            # first 100 HotpotQA validation rows (already committed)
python -m sabot run --dry-run         # offline fake backend: plumbing check + cost estimate, $0
echo ANTHROPIC_API_KEY=sk-ant-... > .env   # .env is git-ignored
python -m sabot run                   # the real run: writes results.json and report.txt
python -m pytest -q                   # tests for mutation application and the MDE math
```

`uv run sabot run` works the same if you use uv.

Useful flags:

* `--mutants P2,A1` runs a subset.
* `-k 10` takes more baseline reruns.
* `--max-usd 0.45` is the hard spend cap (default). The run aborts before it would exceed it.
* `--backend ollama` runs free on local models. Pull `qwen2.5:3b-instruct` and
  `qwen2.5:1.5b-instruct` first.
* `--no-cache` ignores the response cache.

Every response is cached on disk under `.cache/llm`, keyed by the full request plus a
replicate tag. Rerunning the same command replays from the cache at $0. Each baseline
replicate (`base0..base4`) and each mutant run has its own tag, so they are independent
API calls, never cache copies of each other.

## Results

From one real run on 2026-11-13: Anthropic API, system model `claude-haiku-5-5`, downgrade
tier `claude-haiku-4-5`, `anthropic-python` 2.8.0, code at commit `024f0d9`. Full numbers
are in [`results.json`](results.json) and [`report.txt`](report.txt); run logs are in
[`runs/`](runs/).

**Noise floor** (k = 5 reruns of the unmutated suite; σ is run-to-run):

| sub-suite | n | baseline pass rate (5 runs) | σ_run | MDE_run | MDE_sample |
|---|---|---|---|---|---|
| rag | 60 | 41.7 / 43.3 / 41.7 / 40.0 / 45.0 → mean 42.3% | 1.90 pts | 6.4 pts | 22.4 pts |
| agent | 20 | 80 / 80 / 80 / 75 / 80 → mean 79.0% | 2.24 pts | 7.5 pts | 32.0 pts |

**Mutants:** 15 run, **4 killed, 11 escaped (73%)**, 1 N/A (M2).

| id | mutant | Δ pts (+ = worse) | critical | verdict | witness: what the mutant actually did |
|---|---|---|---|---|---|
| P1 | delete constraint word | −1.0 | 4.4 | escaped | 5% of answers changed |
| P2 | remove grounding instruction | **−19.3** | 4.4 | escaped | 55% of answers changed; 13 cases went fail→pass because the model stopped saying "I don't know" and answered from memory |
| P3 | drop output-format spec | +0.7 | 4.4 | escaped (below MDE) | 27/60 replies no longer JSON |
| P4 | delete a few-shot example | −1.0 | 4.4 | escaped | 7% of answers changed |
| P5 | must → should | −2.7 | 4.4 | escaped | 5% of answers changed |
| R1 | top-1 retrieval | **+34.0** | 4.4 | **KILLED** | 20 stable-pass cases failed |
| R2 | shuffle rank order | −1.0 | 4.4 | escaped | retrieval order changed in 88% of cases |
| R3 | inject wrong-year/tenant doc | −1.0 | 4.4 | escaped | decoy injected in 50/60 cases; 5 pass→fail, 6 fail→pass |
| R4 | halve chunk overlap | +4.0 | 4.4 | escaped (below MDE) | retrieval changed in 50% of cases |
| M1 | Haiku 5.5 → Haiku 4.5 | −4.3 / +4.0 | 5.8 / 6.8 | escaped (below MDE) | 20% / 35% of answers changed |
| M2 | temperature 0 → 1 | — | — | N/A | not expressible on this backend |
| M3 | max_tokens −40% | +2.3 / −1.0 | 5.8 / 6.8 | escaped (below MDE) | 5% / 20% of answers changed |
| A1 | skip authorisation call | −1.0 | 5.2 | escaped | **47 restricted documents reached the model, in 17 of 20 cases** |
| A2 | swap search ↔ read | **+24.0** | 5.2 | **KILLED** | 5 pass→fail |
| A3 | read returns 503 | **+34.0** | 5.2 | **KILLED** | 49 errors returned |
| A4 | remove search tool | **+14.0** | 5.2 | **KILLED** | 3 pass→fail |

By surface: prompt 0/5 killed, retrieval 1/4, model 0/2, agent 3/4.

Escapes below the MDE (a real drop, but too small to see): P3, R4, M1, M3. Escapes with
no drop at all: P1, P2, P4, P5, R2, R3, A1.

What stands out:

* **The suite killed only the loud mutants**, the ones that break the pipeline
  mechanically (R1, A2–A4).
* **It rewards removing grounding.** P2 *raises* the score by 19 points: a golden-answer
  suite with no abstention cases scores memorised answers as correct and "I don't know"
  as wrong.
* **It is blind to the access-control bypass.** A1 leaked restricted documents in 17 of 20
  cases, and the score didn't move.

**Spend.** Total API spend was **$0.463**:

| step | cost |
|---|---|
| pilot: one baseline pass, later replayed from cache | $0.026 |
| main run: stopped by the $0.42 cap before A3 and A4 ([`runs/run1-main.log`](runs/run1-main.log)) | $0.422 |
| completion run: everything else replayed from cache ([`runs/run2-complete.log`](runs/run2-complete.log)) | $0.016 |

`results.json` comes from the completion run. Its `spend` block counts only that
invocation's new calls. Its `dirty: true` flag is caused by an untracked log file in the
working tree at the time, not by code changes.

Another reproduction note: the agent sub-suite was cut from n = 30 to n = 20 *before*
any mutant ran, to fit a $0.50 budget.

## Limits

* **One system, one suite, one dataset, small n.** n = 60 RAG + 20 agent cases (the agent n was cut from 30 to keep the whole run under a $0.50 budget, before any mutant ran). Δ and the
  MDE are about *this* suite on *this* system. The escape rate is not a universal
  constant. A stronger suite (abstention cases, citation checks, trace assertions) would
  kill more. That is the intended use: harden the suite, rerun sabot.
* **The suite is golden-answer-only by design.** Escapes of P2 (grounding), P3 (format)
  and A1 (authorisation) are expected for any suite that doesn't test those properties.
  sabot makes that cost visible; it doesn't prove it's unavoidable.
* **The mutants are config-level fault injections**, not source-code AST mutations. The
  systems read the flags the mutants flip, such as `authorize`, `top_k` and `bindings`.
* **Model knowledge leaks in.** HotpotQA is built from Wikipedia, so a model can often
  answer from memory with retrieval broken. That is a real property of RAG evals on
  public data, and part of why the retrieval mutants can escape.
* **The ACL and the "wrong-tenant" document are sabot's construction** on top of
  HotpotQA. They are not part of the dataset.
* **r = 1 mutant run.** The kill test assumes the mutant's run-to-run variance equals the
  baseline's. A higher-variance mutant may be under- or over-killed. Use `-r 3` to tighten.
* **k = 5** gives a noisy σ estimate. The t-quantiles account for that, but the MDE itself
  is only as stable as five runs allow.
* **The Anthropic API takes no seed.** Reproducibility comes from the response cache
  plus the model IDs in `results.json`, not from sampling seeds.
* **The Ollama backend is implemented but was not used for the published run.**
* M2 is N/A on Anthropic (see above), so the published run has 15 runnable mutants.

## License

Code: MIT. Data file: CC BY-SA 4.0 (HotpotQA).
