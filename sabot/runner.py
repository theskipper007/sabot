"""Orchestration: baseline reruns -> noise floor -> each mutant -> killed/escaped -> report."""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import __version__
from .agent import Agent
from .config import MODEL_TIERS, SystemConfig, baseline
from .data import DATASET_CONFIG, DATASET_ID, DATASET_LICENSE, DATASET_SPLIT, DATASET_URL, Dataset, load
from .evals import PASS_F1, agent_pass, normalize, rag_pass
from .llm import LLM
from .mutants import BY_ID, CATALOGUE, MutantNotApplied
from .rag import RAG
from .stats import ALPHA, POWER, is_killed, mcnemar, noise_floor

ROOT = Path(__file__).resolve().parent.parent


def git_commit() -> dict:
    def g(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return {"commit": g("rev-parse", "HEAD") or None, "dirty": bool(g("status", "--porcelain"))}
    except FileNotFoundError:
        return {"commit": None, "dirty": None}


def case_seed(i: int) -> int:
    return 1000 + i


def run_suite(cfg: SystemConfig, data: Dataset, llm: LLM, tag: str, which: tuple, workers: int) -> dict:
    out = {}
    if "rag" in which:
        rag = RAG(cfg, data, llm)
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda ie: rag.answer(ie[1], case_seed(ie[0]), tag), enumerate(data.rag)))
        out["rag"] = [{"id": r.id, "answer": r.answer, "pass": rag_pass(r.answer, e.answer),
                       "json": r.parsed_json, "stop": r.stop_reason, "retrieved": r.retrieved,
                       "injected": r.injected} for r, e in zip(res, data.rag)]
    if "agent" in which:
        ag = Agent(cfg, data, llm)
        off = len(data.rag)
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda ie: ag.answer(ie[1], case_seed(off + ie[0]), tag), enumerate(data.agent)))
        out["agent"] = [{"id": r.id, "answer": r.answer, "pass": agent_pass(r.answer, r.finished, e.answer),
                         "finished": r.finished, "steps": r.steps, "stops": r.stop_reasons,
                         "tools": [c[0] for c in r.trace.calls], "authorize_calls": r.trace.authorize_calls,
                         "restricted_leaked": r.trace.restricted_leaked, "tool_errors": r.trace.tool_errors}
                        for r, e in zip(res, data.agent)]
    return out


def score(cases: list[dict]) -> float:
    return sum(c["pass"] for c in cases) / len(cases)


def changed_frac(a: list[dict], b: list[dict]) -> float:
    return sum(normalize(x["answer"]) != normalize(y["answer"]) for x, y in zip(a, b)) / len(a)


def witness(mid: str, sub: str, base0: list[dict], mut: list[dict]) -> dict:
    """Evidence that the mutant is active (not an equivalent mutant), independent of the suite."""
    w = {"answers_changed_vs_base0": round(changed_frac(base0, mut), 4)}
    if sub == "rag":
        w["retrieval_changed"] = round(sum(x["retrieved"] != y["retrieved"] for x, y in zip(base0, mut)) / len(mut), 4)
        w["non_json_replies"] = sum(not c["json"] for c in mut)
        w["truncated_replies"] = sum(c["stop"] == "length" for c in mut)
        w["refusals"] = sum(c["stop"] == "refusal" for c in mut)
        if mid == "R3":
            w["cases_with_injected_doc"] = sum(c["injected"] for c in mut)
    else:
        w["restricted_docs_leaked"] = sum(len(c["restricted_leaked"]) for c in mut)
        w["cases_leaking"] = sum(bool(c["restricted_leaked"]) for c in mut)
        w["authorize_calls"] = sum(c["authorize_calls"] for c in mut)
        w["tool_errors_returned"] = sum(c["tool_errors"] for c in mut)
        w["unfinished"] = sum(not c["finished"] for c in mut)
        w["truncated_turns"] = sum(s == "length" for c in mut for s in c["stops"])
    return w


def run(backend: str = "anthropic", n_rag: int = 60, n_agent: int = 30, k: int = 5, r: int = 1,
        mutant_ids: list[str] | None = None, workers: int = 8, out: Path | None = None,
        cache: bool = True, max_usd: float | None = 0.45, log=print) -> dict:
    t0 = time.time()
    dry = backend == "fake"
    data = load(n_rag, n_agent)
    llm = LLM(backend, cache=cache, max_usd=max_usd)
    base = baseline(backend)
    subs = ("rag", "agent")
    n = {"rag": len(data.rag), "agent": len(data.agent)}

    log(f"[sabot] backend={backend} model={base.model.name} n_rag={n['rag']} n_agent={n['agent']} k={k} r={r}")
    base_runs = []
    for i in range(k):
        base_runs.append(run_suite(base, data, llm, f"base{i}", subs, workers))
        log(f"[sabot] baseline {i + 1}/{k}: " + "  ".join(f"{s}={score(base_runs[-1][s]):.3f}" for s in subs)
            + f"  (calls {llm.usage.calls}, ${llm.usage.usd:.4f}, est ${llm.usage.est_usd:.4f})")

    floors = {s: noise_floor([score(b[s]) for b in base_runs], n[s], r=r) for s in subs}
    majority = {s: [sum(b[s][j]["pass"] for b in base_runs) > k / 2 for j in range(n[s])] for s in subs}
    stable = {s: [all(b[s][j]["pass"] for b in base_runs) for j in range(n[s])] for s in subs}
    base_churn = {s: (sum(changed_frac(base_runs[0][s], b[s]) for b in base_runs[1:]) / (k - 1)) for s in subs}

    results = []
    for m in CATALOGUE:
        if mutant_ids and m.id not in mutant_ids:
            continue
        rec = {"id": m.id, "surface": m.surface, "name": m.name, "why": m.why, "targets": list(m.targets)}
        try:
            cfg = m.apply(base, backend)
        except MutantNotApplied as e:
            rec.update(verdict="N/A", reason=str(e))
            results.append(rec)
            log(f"[sabot] {m.id} N/A: {e}")
            continue
        runs = [run_suite(cfg, data, llm, f"mut:{m.id}:{j}", m.targets, workers) for j in range(r)]
        per = {}
        any_kill = False
        for s in m.targets:
            mscores = [score(x[s]) for x in runs]
            killed, delta, crit = is_killed(floors[s], mscores, n_tests=len(m.targets))
            any_kill |= killed
            last = runs[-1][s]
            per[s] = {
                "scores": mscores, "delta": delta, "critical": crit, "killed": killed,
                "below_mde_run": (not killed) and 0 < delta < floors[s].mde_run,
                "below_mde_sample": delta < floors[s].mde_sample,
                "stable_pass_cases_failed": sum(1 for j, c in enumerate(last) if stable[s][j] and not c["pass"]),
                "mcnemar": mcnemar(majority[s], [c["pass"] for c in last]),
                "witness": witness(m.id, s, base_runs[0][s], last),
                "cases": last,
            }
        rec["subsuites"] = per
        rec["verdict"] = "KILLED" if any_kill else "ESCAPED"
        deltas = [per[s]["delta"] for s in m.targets]
        rec["max_delta"] = max(deltas)
        if rec["verdict"] == "ESCAPED":
            rec["escape_kind"] = ("below_mde" if any(per[s]["below_mde_run"] for s in m.targets)
                                  else "no_drop")
        results.append(rec)
        log(f"[sabot] {m.id} {rec['verdict']:7s} " + "  ".join(
            f"{s}: d={per[s]['delta'] * 100:+.1f}pts crit={per[s]['critical'] * 100:.1f}" for s in m.targets)
            + f"  (${llm.usage.usd:.4f})")

    models = {name: llm.model_info(name) for name in MODEL_TIERS[backend]}
    res = {
        "tool": {"name": "sabot", "version": __version__, **git_commit()},
        "dry_run": dry,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "backend": backend, "backend_version": llm.server_version()},
        "models": {"system": base.model.name, "downgrade_tier": MODEL_TIERS[backend][1], "info": models,
                   "sampling": ("API default (no temperature sent)" if backend == "anthropic"
                                else f"temperature={base.model.temperature}")},
        "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "split": DATASET_SPLIT, "url": DATASET_URL,
                    "license": DATASET_LICENSE, "rows": list(data.rows), "file_sha256": data.sha256,
                    "corpus_paragraphs": len(data.corpus),
                    "restricted_paragraphs": sum(p.restricted for p in data.corpus.values())},
        "suite": {"rag": f"pass iff token-F1 >= {PASS_F1}",
                  "agent": f"pass iff finish() called within max_steps and token-F1 >= {PASS_F1}"},
        "seeds": {"case_seed": "1000 + case index (rag first, then agent)",
                  "baseline_tags": [f"base{i}" for i in range(k)],
                  "mutant_tags": "mut:<id>:<j>", "shuffle_seed": "case_seed",
                  "note": "the Anthropic API takes no seed; tags only key the response cache"},
        "stats": {"alpha": ALPHA, "power": POWER, "k": k, "r": r,
                  "sigma": "run-to-run SD of the sub-suite pass rate over k reruns (ddof=1)",
                  "kill_rule": "Delta > t_{1-alpha/m, k-1} * sigma_run * sqrt(1/k + 1/r), m = sub-suites touched",
                  "mde_run": "max((t_{1-alpha,k-1} + t_{1-beta,k-1}) * sigma_run * sqrt(1/k + 1/r), 1/n), one-sided, unpaired",
                  "mde_sample": "(z_{1-alpha} + z_{1-beta}) * sqrt(2 p (1-p) / n), one-sided, unpaired, binomial"},
        "baseline": {s: {"scores": [score(b[s]) for b in base_runs], "n": n[s],
                         "mean": floors[s].mean, "sigma_run": floors[s].sigma_run, "se": floors[s].se,
                         "mde_run": floors[s].mde_run, "mde_run_unfloored": floors[s].mde_run_raw,
                         "mde_sample": floors[s].mde_sample,
                         "stable_pass_cases": sum(stable[s]),
                         "answer_churn_between_reruns": base_churn[s],
                         "runs": [b[s] for b in base_runs]} for s in subs},
        "mutants": results,
        "summary": summarize(results),
        "spend": {"api_calls": llm.usage.calls, "cache_hits": llm.usage.cache_hits,
                  "input_tokens": llm.usage.input_tokens, "output_tokens": llm.usage.output_tokens,
                  "usd": round(llm.usage.usd, 4), "cap_usd": max_usd,
                  "by_model": llm.usage.by_model,
                  "dry_run_estimated_usd_on_anthropic": round(llm.usage.est_usd, 4) if dry else None,
                  "note": "USD counts only calls made in this invocation (cache hits are free)"},
        "wallclock_s": round(time.time() - t0, 1),
    }
    out = out or ROOT / ("results.dry.json" if dry else "results.json")
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    rpt = report(res)
    (out.parent / ("report.dry.txt" if dry else "report.txt")).write_text(rpt, encoding="utf-8")
    log(rpt)
    return res


def summarize(results: list[dict]) -> dict:
    ran = [m for m in results if m["verdict"] != "N/A"]
    by = {}
    for m in ran:
        d = by.setdefault(m["surface"], {"total": 0, "killed": 0, "escaped": 0})
        d["total"] += 1
        d["killed" if m["verdict"] == "KILLED" else "escaped"] += 1
    killed = sum(m["verdict"] == "KILLED" for m in ran)
    esc = [m for m in ran if m["verdict"] == "ESCAPED"]
    return {"mutants_total": len(results), "mutants_run": len(ran), "not_applicable": len(results) - len(ran),
            "killed": killed, "escaped": len(esc),
            "escape_rate": (len(esc) / len(ran)) if ran else None,
            "escaped_below_mde": [m["id"] for m in esc if m.get("escape_kind") == "below_mde"],
            "escaped_no_drop": [m["id"] for m in esc if m.get("escape_kind") == "no_drop"],
            "by_surface": by}


def report(res: dict) -> str:
    L = []
    b = res["baseline"]
    tag = " [DRY RUN - fake backend, numbers are meaningless]" if res["dry_run"] else ""
    L.append(f"sabot {res['tool']['version']} @ {(res['tool']['commit'] or 'uncommitted')[:10]}{tag}")
    L.append(f"system  : RAG (n={b['rag']['n']}) + tool agent (n={b['agent']['n']}) on {res['models']['system']}, "
             f"HotpotQA distractor/validation rows {res['dataset']['rows'][0]}-{res['dataset']['rows'][1] - 1}")
    L.append("suite   : golden-answer token-F1 >= 0.5 (agent: also must call finish)")
    L.append(f"noise   : k={res['stats']['k']} reruns of the unmutated suite, alpha={res['stats']['alpha']}, "
             f"power={res['stats']['power']}, one-sided")
    for s in ("rag", "agent"):
        x = b[s]
        L.append(f"  {s:5s} mean {x['mean'] * 100:5.1f}%  sigma_run {x['sigma_run'] * 100:4.2f} pts  "
                 f"MDE_run {x['mde_run'] * 100:4.1f} pts  MDE_sample {x['mde_sample'] * 100:4.1f} pts  "
                 f"answer churn between reruns {x['answer_churn_between_reruns'] * 100:.0f}%")
    L.append("")
    L.append(f"{'id':3s} {'surface':9s} {'mutant':48s} {'delta pts':>14s} {'crit':>6s}  verdict   answers changed")
    for m in res["mutants"]:
        if m["verdict"] == "N/A":
            L.append(f"{m['id']:3s} {m['surface']:9s} {m['name'][:48]:48s} {'':>14s} {'':>6s}  N/A       (not expressible on this backend)")
            continue
        ds = "/".join(f"{m['subsuites'][s]['delta'] * 100:+.1f}" for s in m["targets"])
        cr = "/".join(f"{m['subsuites'][s]['critical'] * 100:.1f}" for s in m["targets"])
        ch = "/".join(f"{m['subsuites'][s]['witness']['answers_changed_vs_base0'] * 100:.0f}%" for s in m["targets"])
        v = m["verdict"] + ("*" if m.get("escape_kind") == "below_mde" else "")
        L.append(f"{m['id']:3s} {m['surface']:9s} {m['name'][:48]:48s} {ds:>14s} {cr:>6s}  {v:9s} {ch}")
    s = res["summary"]
    L.append("")
    L.append("model mutants touch both sub-suites (shown rag/agent); killed if either drops (Bonferroni alpha/2).")
    L.append("* = escaped with a real but sub-MDE drop (0 < delta < MDE_run).")
    L.append("")
    for surf, d in s["by_surface"].items():
        L.append(f"  {surf:9s} {d['killed']}/{d['total']} killed")
    if s["mutants_run"]:
        L.append(f"result  : {s['mutants_run']} mutants run, {s['killed']} killed, {s['escaped']} escaped "
                 f"({s['escape_rate'] * 100:.0f}%), {s['not_applicable']} N/A")
        L.append(f"          escaped below MDE: {', '.join(s['escaped_below_mde']) or 'none'}; "
                 f"escaped with no drop at all: {', '.join(s['escaped_no_drop']) or 'none'}")
    L.append(f"resolution: this suite can detect a drop of {b['rag']['mde_run'] * 100:.1f} pts (rag) / "
             f"{b['agent']['mde_run'] * 100:.1f} pts (agent) between runs, and {b['rag']['mde_sample'] * 100:.0f} / "
             f"{b['agent']['mde_sample'] * 100:.0f} pts that would generalise beyond these cases.")
    sp = res["spend"]
    L.append(f"spend   : {sp['api_calls']} calls ({sp['cache_hits']} cached), {sp['input_tokens']:,} in / "
             f"{sp['output_tokens']:,} out tokens, ${sp['usd']:.4f}; wallclock {res['wallclock_s']}s")
    if res["dry_run"]:
        L.append(f"estimate: the same calls on Anthropic would cost about "
                 f"${sp['dry_run_estimated_usd_on_anthropic']:.3f} (chars/4 x 1.3, Haiku 5.5 / Haiku 4.5 prices)")
    return "\n".join(L) + "\n"
