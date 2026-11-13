"""The eval suite under test. It is deliberately the common kind: golden answers, scored
on final-answer correctness only. It is held fixed across every run, baseline and mutant.

  rag   case passes iff token-F1(prediction, gold) >= 0.5   (HotpotQA/SQuAD normalisation)
  agent case passes iff the agent called finish() within max_steps AND token-F1 >= 0.5

It does NOT check citations, abstention, latency, tool traces or access control. That
is a property of the suite being tested, not of sabot; see README "Limits".
"""
from __future__ import annotations

import re
import string
from collections import Counter

PASS_F1 = 0.5


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred), normalize(gold)
    # HotpotQA rule: yes/no/noanswer must match exactly
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        return float(p == g)
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pt), same / len(gt)
    return 2 * prec * rec / (prec + rec)


def rag_pass(answer: str, gold: str) -> bool:
    return f1(answer, gold) >= PASS_F1


def agent_pass(answer: str, finished: bool, gold: str) -> bool:
    return finished and f1(answer, gold) >= PASS_F1
