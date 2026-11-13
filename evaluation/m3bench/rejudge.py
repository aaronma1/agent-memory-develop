"""Re-judge saved M3-Bench answers with a different judge model (no re-ingest, no re-answer).

Input is a ``results.jsonl`` produced by ``run.py``, or any JSONL whose rows carry ``question`` /
``answer`` / ``response`` (e.g. converted official M3-Agent outputs). The judge prompt is the same
as in ``run.py`` (official M3-Agent ``prompt_agent_verify_answer_referencing``); an empty answer
counts as wrong.

The judge endpoint is any OpenAI-compatible service, for example:
  - Ollama: ``--base-url http://127.0.0.1:11434/v1 --model <tag>``
  - OpenRouter gpt-4o (official setting): ``--base-url https://openrouter.ai/api/v1
    --model openai/gpt-4o-2024-11-20 --api-key-env OPENROUTER_API_KEY``

Artifacts go to ``--out-dir`` (default ``judge_<model>/`` next to the input file):
``judged.jsonl`` (original rows + ``judge_model`` / ``judge_verdict`` / ``judge_correct``) and
``metrics.json``. The original ``results.jsonl`` is not modified.

Usage:
  python -m evaluation.m3bench.rejudge outputs/m3bench/<run_id>/results.jsonl \
      --base-url http://127.0.0.1:11434/v1 --model deepseek-v4.1-flash
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import openai

from evaluation.m3bench import prompts
from evaluation.m3bench.run import _chat, _slug, _write_json, compute_metrics
from evaluation.shared.env import load_evaluation_env

logger = logging.getLogger("m3bench.rejudge")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def verdict_is_yes(verdict: str) -> bool:
    """Drop a reasoning model's ``<think>`` block before looking for "yes", so the word "yes"
    inside the reasoning cannot count as a correct verdict."""
    tail = _THINK_RE.sub("", verdict).split("</think>")[-1]
    return "yes" in tail.lower()


def _judge_one(client: openai.OpenAI, model: str, row: dict[str, Any],
               max_tokens: int) -> dict[str, Any]:
    out = dict(row)
    response = str(row.get("response") or "").strip()
    out["judge_model"] = model
    if not response:
        out.update(judge_verdict="", judge_correct=False, judge_error="")
        return out
    try:
        verdict = _chat(
            client, model,
            [{"role": "user", "content": prompts.judge_prompt(
                str(row["question"]), str(row["answer"]), response)}],
            temperature=0.0, max_tokens=max_tokens,
        )
        out.update(judge_verdict=verdict, judge_correct=verdict_is_yes(verdict), judge_error="")
    except Exception as exc:
        logger.exception("Judging failed: %s", row.get("question_id", ""))
        out.update(judge_verdict="", judge_correct=False,
                   judge_error=f"{type(exc).__name__}: {exc}"[:500])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Re-judge saved M3-Bench answers")
    p.add_argument("results", type=Path, help="results.jsonl")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="",
                   help="name of the env var holding the API key; default EMPTY")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="reasoning models think first, so leave enough tokens")
    args = p.parse_args(argv)
    load_evaluation_env()  # API keys may live in evaluation/environment/.env
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else "EMPTY"
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is empty")
    client = openai.OpenAI(base_url=args.base_url, api_key=api_key,
                           timeout=openai.Timeout(timeout=300.0, connect=10.0), max_retries=2)

    rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    out_dir = args.out_dir or args.results.parent / f"judge_{_slug(args.model)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        judged = list(pool.map(lambda r: _judge_one(client, args.model, r, args.max_tokens), rows))

    with (out_dir / "judged.jsonl").open("w", encoding="utf-8") as fh:
        for row in judged:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    scored = [
        {
            "mode": str(r.get("mode") or "default"),
            "types": list(r.get("types") or []),
            "correct": r["judge_correct"],
            "response": r.get("response") or "",
            "error": r.get("error") or r["judge_error"],
        }
        for r in judged
    ]
    metrics = {
        "source": str(args.results),
        "judge_model": args.model,
        "judge_host": args.base_url,
        "judge_errors": sum(bool(r["judge_error"]) for r in judged),
        "modes": compute_metrics(scored),
    }
    _write_json(out_dir / "metrics.json", metrics)
    for mode, stats in metrics["modes"].items():
        logger.info("[%s] %s accuracy %.2f%% (%d/%d)", mode, args.model,
                    100 * stats["accuracy"], stats["correct"], stats["questions"])
    logger.info("Artifacts: %s", out_dir)
    return 0 if not metrics["judge_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
