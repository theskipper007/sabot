"""Noise floor, minimum detectable effect (MDE) and the kill decision.

Notation (per sub-suite):
    n            number of eval cases in the sub-suite
    s_1..s_k     suite score (pass rate, 0..1) of k independent reruns of the UNMUTATED system
    m_1..m_r     suite score of r runs of the mutated system (default r = 1)
    Delta        mean(s) - mean(m)            (positive = the mutant scores lower)

1) Run-to-run noise floor (this is what the kill decision uses)
    sigma_run  = sample SD of s_1..s_k (ddof = 1).
                 sigma is RUN-TO-RUN: the same n cases rerun k times; it is NOT a
                 per-example SD and says nothing about sampling of the cases themselves.
    SE         = sigma_run * sqrt(1/k + 1/r)
                 unpaired (mean of k baseline runs vs mean of r mutant runs), assuming the
                 mutant has the same run-to-run variance as the baseline.
    df         = k - 1          (the variance is estimated from the k baseline runs only)
    critical   = t_{1-alpha', df} * SE
    MDE_run    = max( (t_{1-alpha, df} + t_{1-beta, df}) * SE ,  1/n )
                 one-sided, alpha = 0.05, power 1 - beta = 0.80. The 1/n floor is the
                 resolution of the suite: it cannot register a change smaller than one case.
    killed     iff Delta > critical  and  Delta > 0
                 alpha' = alpha / (number of sub-suites the mutant touches)  (Bonferroni).

2) Case-sampling MDE (reported for context, not used to kill)
    MDE_sample = (z_{1-alpha} + z_{1-beta}) * sqrt(2 * p * (1 - p) / n)
                 p = mean baseline pass rate. Per-example binomial SD, UNPAIRED, one-sided,
                 two independent case sets of size n, normal approximation. This is the
                 smallest drop you could claim generalises beyond these n cases.

3) Paired view (reported, not used to kill): McNemar counts against the per-case
   baseline majority vote, with a one-sided exact binomial p-value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats as st

ALPHA = 0.05
POWER = 0.80


def sd(xs: list[float]) -> float:
    k = len(xs)
    if k < 2:
        raise ValueError("need k >= 2 baseline runs to estimate run-to-run SD")
    mu = sum(xs) / k
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (k - 1))


@dataclass
class NoiseFloor:
    n: int
    k: int
    r: int
    mean: float
    sigma_run: float
    se: float
    mde_run: float
    mde_run_raw: float   # before the 1/n floor
    mde_sample: float
    alpha: float = ALPHA
    power: float = POWER


def noise_floor(baseline_scores: list[float], n: int, r: int = 1,
                alpha: float = ALPHA, power: float = POWER) -> NoiseFloor:
    k = len(baseline_scores)
    mu = sum(baseline_scores) / k
    s = sd(baseline_scores)
    se = s * math.sqrt(1 / k + 1 / r)
    df = k - 1
    raw = float((st.t.ppf(1 - alpha, df) + st.t.ppf(power, df)) * se)
    p = mu
    mde_sample = float((st.norm.ppf(1 - alpha) + st.norm.ppf(power)) * math.sqrt(2 * p * (1 - p) / n))
    return NoiseFloor(n=n, k=k, r=r, mean=mu, sigma_run=s, se=se, mde_run=max(raw, 1 / n),
                      mde_run_raw=raw, mde_sample=mde_sample, alpha=alpha, power=power)


def critical(nf: NoiseFloor, n_tests: int = 1) -> float:
    return float(st.t.ppf(1 - nf.alpha / n_tests, nf.k - 1) * nf.se)


def is_killed(nf: NoiseFloor, mutant_scores: list[float], n_tests: int = 1) -> tuple[bool, float, float]:
    """Returns (killed, delta, critical_value)."""
    delta = nf.mean - sum(mutant_scores) / len(mutant_scores)
    crit = critical(nf, n_tests)
    return bool(delta > crit and delta > 0), float(delta), crit


def mcnemar(base_majority: list[bool], mutant: list[bool]) -> dict:
    b = sum(1 for x, y in zip(base_majority, mutant) if x and not y)   # pass -> fail
    c = sum(1 for x, y in zip(base_majority, mutant) if not x and y)   # fail -> pass
    p = float(st.binom.sf(b - 1, b + c, 0.5)) if b + c else 1.0       # P(X >= b)
    return {"pass_to_fail": b, "fail_to_pass": c, "p_one_sided": p}
