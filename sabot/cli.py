"""sabot CLI.

    python -m sabot fetch-data            download the HotpotQA rows (once)
    python -m sabot run --dry-run         offline plumbing check + cost estimate, no API calls
    python -m sabot run                   the real run (Anthropic; needs ANTHROPIC_API_KEY)
    python -m sabot list                  print the mutant catalogue
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal KEY=VALUE loader so the API key can live in a git-ignored .env file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sabot", description="Mutation testing for LLM eval suites")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch-data", help="download HotpotQA validation rows")
    f.add_argument("--rows", type=int, default=100)

    r = sub.add_parser("run", help="baseline noise floor + every mutant")
    r.add_argument("--dry-run", action="store_true", help="offline fake backend; no API calls, no real numbers")
    r.add_argument("--backend", choices=["anthropic", "ollama", "fake"], default="anthropic")
    r.add_argument("--n-rag", type=int, default=60)
    r.add_argument("--n-agent", type=int, default=30)
    r.add_argument("-k", type=int, default=5, help="baseline reruns for the noise floor (default 5)")
    r.add_argument("-r", type=int, default=1, help="runs per mutant (default 1)")
    r.add_argument("--mutants", default="", help="comma-separated ids, e.g. P1,R3 (default: all)")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--max-usd", type=float, default=0.45, help="hard spend cap; aborts when reached")
    r.add_argument("--no-cache", action="store_true", help="ignore the on-disk response cache")
    r.add_argument("--out", type=Path, default=None)

    sub.add_parser("list", help="print the mutant catalogue")
    a = ap.parse_args(argv)

    if a.cmd == "fetch-data":
        from .data import fetch
        print(f"wrote {fetch(a.rows)}")
        return 0
    if a.cmd == "list":
        from .mutants import CATALOGUE
        for m in CATALOGUE:
            print(f"{m.id:3s} {m.surface:9s} {m.name}  -- {m.why}")
        return 0

    load_dotenv()
    backend = "fake" if a.dry_run else a.backend
    if backend == "anthropic":
        from .llm import env_has_anthropic
        if not env_has_anthropic():
            ap.error("ANTHROPIC_API_KEY is not set. Put it in .env (git-ignored) or the environment, "
                     "or use --dry-run.")
    from .runner import run
    ids = [x.strip() for x in a.mutants.split(",") if x.strip()] or None
    run(backend=backend, n_rag=a.n_rag, n_agent=a.n_agent, k=a.k, r=a.r, mutant_ids=ids,
        workers=a.workers, out=a.out, cache=not a.no_cache,
        max_usd=a.max_usd if backend == "anthropic" else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
