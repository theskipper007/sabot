"""MDE and kill-decision math, checked against hand-computed values."""
import math

import pytest
from scipy import stats as st

from sabot.evals import f1, rag_pass, agent_pass
from sabot.stats import critical, is_killed, mcnemar, noise_floor, sd


def test_sd_is_sample_sd():
    assert sd([0.70, 0.72, 0.74, 0.70, 0.74]) == pytest.approx(math.sqrt(0.0004), rel=1e-9)
    with pytest.raises(ValueError):
        sd([0.5])


def test_mde_run_formula_by_hand():
    scores = [0.70, 0.72, 0.74, 0.70, 0.74]  # mean 0.72, sd 0.02
    nf = noise_floor(scores, n=60, r=1)
    se = 0.02 * math.sqrt(1 / 5 + 1 / 1)
    t_a, t_b = 2.131846786, 0.940964577  # t_{0.95,4}, t_{0.80,4}
    assert nf.sigma_run == pytest.approx(0.02)
    assert nf.se == pytest.approx(se)
    assert nf.mde_run_raw == pytest.approx((t_a + t_b) * se, rel=1e-6)
    assert nf.mde_run == pytest.approx(max((t_a + t_b) * se, 1 / 60), rel=1e-6)
    assert nf.mde_run == pytest.approx(0.06730, abs=1e-4)


def test_mde_run_floors_at_resolution_when_noise_is_zero():
    nf = noise_floor([0.7] * 5, n=40)
    assert nf.sigma_run == 0 and nf.mde_run_raw == 0
    assert nf.mde_run == pytest.approx(1 / 40)


def test_mde_sample_formula_by_hand():
    nf = noise_floor([0.5, 0.5, 0.5, 0.5, 0.5], n=50)
    z = 1.644853627 + 0.841621234
    assert nf.mde_sample == pytest.approx(z * math.sqrt(2 * 0.25 / 50), rel=1e-6)  # ~24.9 pts
    assert nf.mde_sample == pytest.approx(0.2486, abs=1e-3)


def test_more_reruns_shrink_mde():
    a = noise_floor([0.70, 0.74, 0.72], n=60)
    b = noise_floor([0.70, 0.74, 0.72, 0.70, 0.74, 0.72, 0.70, 0.74, 0.72], n=60)
    assert b.mde_run < a.mde_run


def test_kill_decision():
    nf = noise_floor([0.70, 0.72, 0.74, 0.70, 0.74], n=60)
    crit = st.t.ppf(0.95, 4) * nf.se
    assert critical(nf) == pytest.approx(crit)
    k, d, c = is_killed(nf, [0.72 - crit - 1e-6])
    assert k and d == pytest.approx(crit + 1e-6)
    k, d, c = is_killed(nf, [0.72 - crit + 1e-6])
    assert not k
    k, _, _ = is_killed(nf, [0.80])  # improvement is never a kill
    assert not k
    # Bonferroni: two sub-suites -> stricter critical value
    assert critical(nf, n_tests=2) > critical(nf, n_tests=1)
    assert isinstance(is_killed(nf, [0.5])[0], bool)


def test_kill_with_zero_noise_needs_any_drop():
    nf = noise_floor([0.7] * 5, n=60)
    assert is_killed(nf, [0.7 - 1 / 60])[0]
    assert not is_killed(nf, [0.7])[0]


def test_mcnemar():
    base = [True] * 6 + [False] * 4
    mut = [False, False, False, True, True, True, True, False, False, False]
    r = mcnemar(base, mut)
    assert r["pass_to_fail"] == 3 and r["fail_to_pass"] == 1
    assert r["p_one_sided"] == pytest.approx(5 / 16)  # P(X>=3), X~Bin(4, .5)
    assert mcnemar([True], [True])["p_one_sided"] == 1.0


def test_f1_and_pass_rules():
    assert f1("The Ruth Okafor", "Ruth Okafor") == 1.0
    assert f1("yes", "yes") == 1.0 and f1("yes, both", "yes") == 0.0
    assert f1("Paris, France", "Paris") == pytest.approx(2 / 3)
    assert rag_pass("Paris, France", "Paris") and not rag_pass("Lyon", "Paris")
    assert not agent_pass("Paris", False, "Paris") and agent_pass("Paris", True, "Paris")
