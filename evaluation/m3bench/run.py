"""M3-Bench (web subset) end-to-end eval: ingest -> multimodal retrieval -> answer -> judge.

Pipeline:
  1. Ingest: one Scope per video (``org=m3bench-<scope_tag>``, ``user=v_<video_id>``) via
     ``MemoryAPI.add(source=VIDEO, pipeline=video, infer=true)`` through VideoNormalizer ->
     VideoMemoryExtractor, producing CLM (clip) / ELM (event) memories. Normalization output is
     cached per model combination, so reruns only redo construction and retrieval.
  2. Answer: the M3-Agent control loop (``[Search]`` / ``[Answer]``, answer forced on the last
     round). ``iterative`` starts from empty knowledge for up to ``--rounds`` rounds; ``single``
     retrieves once with the question and answers directly.
  3. Judge: the official M3-Agent ``prompt_agent_verify_answer_referencing``; an output containing
     "yes" counts as correct, an empty answer counts as wrong (same as
     ``m3_agent/control.py::eval_answer``).

Only locally downloaded videos are evaluated (see ``evaluation.m3bench.download``); coverage is
written to metrics.json.

Usage:
  python -m evaluation.m3bench.run --data-root ~/data/m3bench/web --max-videos 1 --max-qa 2
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import openai
from dotenv import load_dotenv

from evaluation.m3bench import plugins, prompts
from evaluation.m3bench.download import find_output
from evaluation.shared.config_loader import load_layer
from evaluation.shared.env import EVALUATION_DIR, load_evaluation_env
from jiuwen_memory.api import (
    Credentials,
    MemoryAPI,
    Surface,
    assemble_runtime,
    build_dev_authenticator,
    new_request_context,
)
from jiuwen_memory.common.security.types import RequestSecurityContext
from jiuwen_memory.common.type_def import Context, MemoryUnit, Modality, Scope
from jiuwen_memory.config import Config
from jiuwen_memory.retrieval.types import DisclosureLevel, RetrievedItem

logger = logging.getLogger("m3bench")

_PROJECT_ROOT = EVALUATION_DIR.parent
_DEFAULT_DATA_ROOT = Path("~/data/m3bench/web")
_DEFAULT_OUTPUT_ROOT = EVALUATION_DIR / "outputs" / "m3bench"
_LOCOMO_CONFIG = EVALUATION_DIR / "locomo" / "config.yml"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEXT_MODEL = "google/gemini-2.5-flash"
DEFAULT_VLM_MODEL = "google/gemini-2.5-flash"
# Official M3-Agent judge model (control.py::gpt_model), called through OpenRouter.
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-2024-11-20"
DEFAULT_ASR_MODEL = "whisper-large-v3-turbo"

_IDENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


# --------------------------------------------------------------------------- #
# Environment and model endpoints
# --------------------------------------------------------------------------- #


def _set_default(name: str, value: str) -> None:
    """An explicit empty string in ``.env`` also counts as unset."""
    if not os.environ.get(name, "").strip():
        os.environ[name] = value


def _load_env() -> None:
    load_evaluation_env()
    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)
    _set_default("M3BENCH_LLM_BASE_URL", OPENROUTER_BASE_URL)
    _set_default("M3BENCH_LLM_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
    _set_default("M3BENCH_TEXT_MODEL", DEFAULT_TEXT_MODEL)
    _set_default("M3BENCH_VLM_MODEL", DEFAULT_VLM_MODEL)
    _set_default("M3BENCH_ANSWER_MODEL", os.environ["M3BENCH_TEXT_MODEL"])
    _set_default("M3BENCH_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    _set_default("M3BENCH_ASR_MODEL", DEFAULT_ASR_MODEL)
    # The judge stays on the official model: with an OpenRouter key it does not follow the
    # other roles onto a self-hosted vLLM.
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        _set_default("M3BENCH_JUDGE_BASE_URL", OPENROUTER_BASE_URL)
        _set_default("M3BENCH_JUDGE_API_KEY", os.environ["OPENROUTER_API_KEY"])
    for role in ("VLM", "ANSWER", "JUDGE"):
        _set_default(f"M3BENCH_{role}_BASE_URL", os.environ["M3BENCH_LLM_BASE_URL"])
        _set_default(f"M3BENCH_{role}_API_KEY", os.environ["M3BENCH_LLM_API_KEY"])
    missing = [
        name
        for name in ("M3BENCH_LLM_API_KEY", "M3BENCH_VLM_API_KEY",
                     "M3BENCH_ANSWER_API_KEY", "M3BENCH_JUDGE_API_KEY")
        if not os.environ.get(name, "").strip()
    ]
    if missing:
        raise RuntimeError(
            "Missing model API key: set OPENROUTER_API_KEY in evaluation/environment/.env "
            "or the repo-root .env, or set each of " + ", ".join(missing)
        )


def _host(url: str) -> str:
    return urlparse(url).netloc or url


def _endpoints_meta() -> dict[str, dict[str, str]]:
    """Record only model names and hosts, never keys."""
    env = os.environ
    meta = {
        role.lower(): {
            "model": env[f"M3BENCH_{model}_MODEL"],
            "host": _host(env[f"M3BENCH_{base}_BASE_URL"]),
        }
        for role, model, base in (
            ("TEXT", "TEXT", "LLM"),
            ("VLM", "VLM", "VLM"),
            ("ANSWER", "ANSWER", "ANSWER"),
            ("JUDGE", "JUDGE", "JUDGE"),
        )
    }
    asr_url = env.get("M3BENCH_ASR_BASE_URL", "").strip()
    meta["asr"] = (
        {"model": env["M3BENCH_ASR_MODEL"], "host": _host(asr_url)}
        if asr_url else {"model": "silent", "host": ""}
    )
    embed_url = env.get("EMBED_BASE_URL", "").strip()
    meta["embedder"] = (
        {"model": env.get("EMBED_MODEL_NAME", ""), "host": _host(embed_url)}
        if embed_url else {"model": "hashing", "host": ""}
    )
    return meta


def _client(role: str) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=os.environ[f"M3BENCH_{role}_BASE_URL"],
        api_key=os.environ[f"M3BENCH_{role}_API_KEY"],
        timeout=openai.Timeout(timeout=180.0, connect=10.0),
        max_retries=2,
    )


def _chat(
    client: openai.OpenAI,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    attempts: int = 5,
) -> str:
    delay = 5.0
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError) as exc:
            if attempt == attempts:
                raise
            logger.warning("%s call failed (%d/%d): %s, retrying in %.0fs",
                           model, attempt, attempts, type(exc).__name__, delay)
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- #
# Data selection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Video:
    video_id: str
    path: Path
    qa_list: list[dict[str, Any]]


@dataclass
class Coverage:
    videos_total: int = 0
    qas_total: int = 0
    videos_on_disk: int = 0
    qas_on_disk: int = 0


def _select_videos(args: argparse.Namespace) -> tuple[list[Video], Coverage]:
    with args.annotations.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected annotation format: {args.annotations}")
    coverage = Coverage(videos_total=len(data),
                        qas_total=sum(len(v.get("qa_list", [])) for v in data.values()))
    wanted = [v.strip() for v in args.video_ids.split(",") if v.strip()]
    unknown = sorted(set(wanted) - set(data))
    if unknown:
        raise RuntimeError(f"Videos not in annotations: {', '.join(unknown)}")
    videos: list[Video] = []
    for video_id in wanted or list(data):
        path = find_output(args.videos_dir, video_id)
        if path is None or path.stat().st_size == 0:
            if wanted:
                raise RuntimeError(f"Video not downloaded: {video_id} "
                                   "(run evaluation.m3bench.download first)")
            continue
        qa_list = list(data[video_id].get("qa_list", []))
        coverage.videos_on_disk += 1
        coverage.qas_on_disk += len(qa_list)
        videos.append(Video(video_id, path.resolve(), qa_list))
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    if not videos:
        raise RuntimeError(f"No videos to evaluate under {args.videos_dir}")
    return videos, coverage


# --------------------------------------------------------------------------- #
# Kernel assembly
# --------------------------------------------------------------------------- #


def _slug(value: str) -> str:
    return _IDENT_RE.sub("_", value).strip("_") or "default"


def _cache_dir(args: argparse.Namespace) -> Path:
    env = os.environ
    asr = env["M3BENCH_ASR_MODEL"] if env.get("M3BENCH_ASR_BASE_URL", "").strip() else "silent"
    key = "__".join(
        _slug(part) for part in (
            env["M3BENCH_VLM_MODEL"], env["M3BENCH_TEXT_MODEL"], asr, f"c{args.chunk_seconds}",
        )
    )
    return args.cache_root / key


def _scope_org(scope_tag: str) -> str:
    return f"m3bench-{scope_tag}"


def _scope_user(video_id: str) -> str:
    return f"v_{video_id}"


def _build_config(
    args: argparse.Namespace, videos: list[Video], scope_tag: str, out_dir: Path,
) -> dict[str, Any]:
    layer = load_layer(str(args.config))
    payload = copy.deepcopy(dict(layer.get("memory_api") or layer))
    globals_ = payload.setdefault("globals", {})
    globals_.setdefault("log_file", str(out_dir / "engine.log"))

    routes = payload["normalizer"]["default"]["params"]["routes"]
    video_route = routes["video"]
    video_params = video_route.setdefault("params", {})
    video_params["chunk_seconds"] = args.chunk_seconds
    video_params["local_media_roots"] = [
        {
            "scope": {"org": _scope_org(scope_tag), "user": _scope_user(v.video_id)},
            "root": str(args.videos_dir.resolve()),
        }
        for v in videos
    ]
    if not args.no_cache:
        routes["video"] = {
            "target": plugins.CACHED_VIDEO_TARGET,
            "params": {"cache_dir": str(_cache_dir(args)), "inner": video_route},
        }

    if not os.environ.get("M3BENCH_ASR_BASE_URL", "").strip():
        logger.warning("M3BENCH_ASR_BASE_URL not set: using silent ASR, "
                       "memories come from the visual stream only")
        payload["asr"] = {"video": {"target": plugins.SILENT_ASR_TARGET}}

    if os.environ.get("EMBED_BASE_URL", "").strip():
        payload["embedder"] = {"default": {"target": "openai"}}
        globals_.update({
            "embedder_model": os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-m3"),
            "embedder_base_url": os.environ["EMBED_BASE_URL"],
            "embedder_api_key": os.environ.get("EMBED_API_KEY", ""),
            "embedder_dim": int(os.environ.get("EMBEDDING_MODEL_DIMENTION", "1024")),
            "embedder_max_batch": 64,
        })
    else:
        logger.warning("EMBED_BASE_URL not set: "
                       "vector channel uses the kernel default hashing embedder")

    if args.storage == "deploy":
        locomo = load_layer(str(_LOCOMO_CONFIG))["memory_api"]
        for key in ("kv_store", "fulltext_store", "vector_store"):
            payload[key] = copy.deepcopy(locomo[key])
        globals_["vector_enabled"] = True
        globals_.setdefault("layers_index_enabled", False)
    return payload


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #


class Kernel:
    """Thin eval-side wrapper over MemoryAPI: uniform Scope and security context."""

    _IDENTITY = "m3bench-root"

    def __init__(self, api: MemoryAPI, scope_tag: str) -> None:
        self.api = api
        self._org = _scope_org(scope_tag)
        # Same as longmemeval/harness.py: in-process eval uses the dev authenticator's root
        # identity; the target Scope comes from call arguments, not an actor derived from data.
        self._authenticator = build_dev_authenticator(identities={
            self._IDENTITY: {"actor": {"org": "local", "user": "developer"}, "role": "root"},
        })

    def scope(self, video_id: str) -> Scope:
        return Scope(org=self._org, user=_scope_user(video_id))

    def security(self, scope: Scope) -> RequestSecurityContext:
        del scope  # fresh context per call; identity is always the eval root
        auth = self._authenticator.authenticate(Credentials(api_key=self._IDENTITY))
        return new_request_context(auth, surface=Surface.INTERNAL)

    def unit_count(self, video_id: str) -> int:
        scope = self.scope(video_id)
        page = self.api.list(scope, security=self.security(scope), offset=0, limit=1)
        return int(page.count)

    def ingest(self, video: Video) -> list[MemoryUnit]:
        scope = self.scope(video.video_id)
        uri = video.path.as_uri()
        return self.api.add(
            uri,
            scope,
            Modality.VIDEO,
            security=self.security(scope),
            assets=[uri],
            system_metadata={
                "infer": "true",
                "pipeline": "video",
                "payload_id": video.video_id,
            },
        )

    def search(self, video_id: str, query: str, top_k: int) -> list[RetrievedItem]:
        scope = self.scope(video_id)
        result = self.api.search(
            query,
            Context(scope),
            security=self.security(scope),
            top_k=top_k,
            disclosure=DisclosureLevel.L2,
        )
        return list(result.items)


def _ingest_one(kernel: Kernel, video: Video, skip_write: bool) -> dict[str, Any]:
    started = time.perf_counter()
    record: dict[str, Any] = {"video_id": video.video_id, "path": str(video.path)}
    try:
        existing = kernel.unit_count(video.video_id)
        if existing or skip_write:
            record.update(status="reused" if existing else "missing", units=existing)
        else:
            units = kernel.ingest(video)
            levels = [str(u.system_metadata.get("memory_level", "")) for u in units]
            record.update(status="ok", units=len(units),
                          clm=levels.count("clm"), elm=levels.count("elm"))
    except Exception as exc:  # one failed video does not stop the run
        logger.exception("Ingest failed for video %s", video.video_id)
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
    record["seconds"] = round(time.perf_counter() - started, 2)
    return record


# --------------------------------------------------------------------------- #
# Answering and judging
# --------------------------------------------------------------------------- #


def _mmss(value: Any) -> str:
    try:
        seconds = int(float(value))
    except (TypeError, ValueError):
        return "?"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _knowledge_key(item: RetrievedItem, index: int) -> str:
    meta = item.system_metadata or {}
    level = {"clm": "CLIP", "elm": "EVENT"}.get(str(meta.get("memory_level", "")), "MEMORY")
    if "start_seconds" in meta:
        return f"{level} {_mmss(meta.get('start_seconds'))}-{_mmss(meta.get('end_seconds'))}"
    return f"{level} {index}"


@dataclass
class AnswerSettings:
    mode: str
    rounds: int
    top_k: int
    temperature: float
    max_tokens: int


@dataclass
class Trace:
    seen: set[str] = field(default_factory=set)
    steps: list[dict[str, Any]] = field(default_factory=list)


class Answerer:
    def __init__(self, kernel: Kernel, settings: AnswerSettings) -> None:
        self._kernel = kernel
        self._settings = settings
        self._answer_client = _client("ANSWER")
        self._answer_model = os.environ["M3BENCH_ANSWER_MODEL"]
        self._judge_client = _client("JUDGE")
        self._judge_model = os.environ["M3BENCH_JUDGE_MODEL"]

    def _searched(self, video_id: str, query: str, trace: Trace) -> str:
        knowledge: dict[str, list[str]] = {}
        hits = []
        for index, item in enumerate(self._kernel.search(video_id, query, self._settings.top_k)):
            hits.append({
                "unit_id": item.unit_id,
                "score": round(float(item.score), 4),
                "key": _knowledge_key(item, index),
            })
            # As in the official loop: each round shows only memories not seen before.
            if item.unit_id in trace.seen:
                continue
            trace.seen.add(item.unit_id)
            knowledge.setdefault(_knowledge_key(item, index), []).append(item.content)
        trace.steps.append({"type": "search", "query": query, "hits": hits,
                            "new": sum(len(v) for v in knowledge.values())})
        text = "Searched knowledge: " + json.dumps(knowledge, ensure_ascii=False)
        if not knowledge:
            text += prompts.EMPTY_SEARCH_HINT
        return text

    def answer(self, video_id: str, question: str) -> tuple[str, Trace]:
        trace = Trace()
        settings = self._settings
        if settings.mode == "single":
            rounds = 1
            first = self._searched(video_id, question, trace)
        else:
            rounds = settings.rounds
            first = "Searched knowledge: {}"
        messages = [
            {"role": "system", "content": prompts.CONTROL_SYSTEM_PROMPT.format(question=question)},
            {"role": "user", "content": first},
        ]
        for idx in range(rounds):
            messages[-1]["content"] += prompts.CONTROL_INSTRUCTION
            if idx == rounds - 1:
                messages[-1]["content"] += prompts.FINAL_ROUND_HINT
            output = _chat(self._answer_client, self._answer_model, messages,
                           temperature=settings.temperature, max_tokens=settings.max_tokens)
            messages.append({"role": "assistant", "content": output})
            match = re.search(prompts.ACTION_PATTERN, output.split("</think>")[-1], re.DOTALL)
            action, content = (match.group(1), match.group(2)) if match else ("Search", None)
            trace.steps.append({"type": "generate", "round": idx, "output": output,
                                "action": action})
            if action == "Answer":
                return (content or "").strip(), trace
            if idx == rounds - 1:
                break
            query = (content or "").strip()
            searched = (self._searched(video_id, query, trace) if query
                        else "Searched knowledge: {}" + prompts.EMPTY_SEARCH_HINT)
            messages.append({"role": "user", "content": searched})
        return "", trace

    def judge(self, question: str, ground_truth: str, response: str) -> tuple[bool, str]:
        if not response:
            return False, ""
        verdict = _chat(
            self._judge_client,
            self._judge_model,
            [{"role": "user", "content": prompts.judge_prompt(question, ground_truth, response)}],
            temperature=0.0,
            max_tokens=2048,
        )
        return "yes" in verdict.lower(), verdict


def _qa_one(answerer: Answerer, video: Video, qa: dict[str, Any], mode: str) -> dict[str, Any]:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "question_id": qa.get("question_id", ""),
        "video_id": video.video_id,
        "mode": mode,
        "types": list(qa.get("type") or []),
        "question": qa["question"],
        "answer": qa["answer"],
    }
    try:
        response, trace = answerer.answer(video.video_id, qa["question"])
        correct, verdict = answerer.judge(qa["question"], qa["answer"], response)
        record.update(response=response, correct=correct, verdict=verdict,
                      trace=trace.steps, error="")
    except Exception as exc:
        logger.exception("Question %s failed", record["question_id"])
        record.update(response="", correct=False, verdict="", trace=[],
                      error=f"{type(exc).__name__}: {exc}"[:500])
    record["seconds"] = round(time.perf_counter() - started, 2)
    return record


# --------------------------------------------------------------------------- #
# Metrics and artifacts
# --------------------------------------------------------------------------- #


def _acc(correct: int, total: int) -> float:
    return round(correct / total, 4) if total else 0.0


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in sorted({r["mode"] for r in records}):
        rows = [r for r in records if r["mode"] == mode]
        correct = sum(bool(r["correct"]) for r in rows)
        per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in rows:
            for qtype in row["types"] or ["(untyped)"]:
                per_type[qtype][0] += int(bool(row["correct"]))
                per_type[qtype][1] += 1
        by_mode[mode] = {
            "questions": len(rows),
            "correct": correct,
            "accuracy": _acc(correct, len(rows)),
            "no_answer": sum(not r["response"] for r in rows),
            "errors": sum(bool(r["error"]) for r in rows),
            "by_type": {
                qtype: {"questions": n, "correct": c, "accuracy": _acc(c, n)}
                for qtype, (c, n) in sorted(per_type.items())
            },
        }
    return by_mode


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def _parallel(fn: Callable[[Any], dict[str, Any]], items: list[Any], workers: int,
              writer: JsonlWriter, label: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix=label) as pool:
        futures = [pool.submit(fn, item) for item in items]
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            writer.write(record)
            results.append(record)
            logger.info("[%s %d/%d] %s", label, done, len(items), _brief(record))
    return results


def _brief(record: dict[str, Any]) -> str:
    if "question_id" in record:
        mark = "✓" if record["correct"] else "✗"
        return f"{record['question_id']} {record['mode']} {mark} {record['seconds']}s"
    return (f"{record['video_id']} {record['status']} units={record.get('units', 0)} "
            f"{record['seconds']}s")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3-Bench web subset end-to-end eval")
    p.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT,
                   help="download.py --out-dir (contains annotations/ and videos/)")
    p.add_argument("--annotations", type=Path, default=None,
                   help="default <data-root>/annotations/web.json")
    p.add_argument("--video-ids", default="", help="comma-separated; default all downloaded videos")
    p.add_argument("--max-videos", type=int, default=0, help="0 means no limit")
    p.add_argument("--max-qa", type=int, default=0,
                   help="at most N questions per video; 0 means all")
    p.add_argument("--mode", choices=("single", "iterative", "both"), default="both")
    p.add_argument("--rounds", type=int, default=5,
                   help="max iterative rounds (official total_round=5)")
    p.add_argument("--top-k", type=int, default=10, help="results per retrieval")
    p.add_argument("--answer-temperature", type=float, default=0.0)
    p.add_argument("--answer-max-tokens", type=int, default=4096)
    p.add_argument("--chunk-seconds", type=int, default=30,
                   help="video clip length in seconds (official 30s)")
    p.add_argument("--ingest-concurrency", type=int, default=2)
    p.add_argument("--qa-concurrency", type=int, default=4)
    p.add_argument("--storage", choices=("memory", "deploy"), default="memory",
                   help="memory: in-process stores; deploy: Redis/ES/Milvus from locomo/config.yml")
    p.add_argument("--scope-tag", default="",
                   help="reuse an existing Scope (deploy storage only); default this run_id")
    p.add_argument("--skip-write", action="store_true",
                   help="answer only, no ingest (needs --scope-tag)")
    p.add_argument("--no-cache", action="store_true",
                   help="do not read or write the video normalization cache")
    p.add_argument("--cache-root", type=Path, default=None,
                   help="normalization cache root; default <data-root>/memory_cache")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yml"))
    p.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-id", default="")
    args = p.parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    args.annotations = (args.annotations or args.data_root / "annotations" / "web.json")
    args.annotations = args.annotations.expanduser().resolve()
    args.videos_dir = args.data_root / "videos"
    args.cache_root = (args.cache_root or args.data_root / "memory_cache").expanduser().resolve()
    if not args.run_id:
        stamp = datetime.now(UTC).astimezone().strftime("%Y%m%d-%H%M%S")
        args.run_id = f"{stamp}-{uuid4().hex[:8]}"
    if args.skip_write and not args.scope_tag:
        raise RuntimeError("--skip-write requires --scope-tag")
    if args.skip_write and args.storage != "deploy":
        raise RuntimeError("--skip-write only works with --storage deploy; "
                           "in-process stores do not persist")
    if args.rounds < 1 or args.top_k < 1:
        raise RuntimeError("--rounds and --top-k must be positive")
    return args


def _setup_logging(out_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s [%(threadName)s] %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(out_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    for noisy in ("httpx", "openai", "agent_memory"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _load_env()
    out_dir = args.output_root.expanduser().resolve() / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir)
    scope_tag = _slug(args.scope_tag) if args.scope_tag else args.run_id

    videos, coverage = _select_videos(args)
    modes = ["single", "iterative"] if args.mode == "both" else [args.mode]
    meta: dict[str, Any] = {
        "run_id": args.run_id,
        "scope_org": _scope_org(scope_tag),
        "git_commit": _git_commit(),
        "started_at": datetime.now(UTC).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "endpoints": _endpoints_meta(),
        "cache_dir": "" if args.no_cache else str(_cache_dir(args)),
        "coverage": vars(coverage),
        "videos": [v.video_id for v in videos],
    }
    _write_json(out_dir / "run_meta.json", meta)
    logger.info("run %s: %d videos, modes %s, output %s", args.run_id, len(videos), modes, out_dir)

    # plugins registered with the Producers at import time, before this assembly.
    payload = _build_config(args, videos, scope_tag, out_dir)
    runtime = assemble_runtime(config=Config.from_dict(payload))
    try:
        kernel = Kernel(runtime.api, scope_tag)
        ingest = _parallel(lambda v: _ingest_one(kernel, v, args.skip_write), videos,
                           args.ingest_concurrency, JsonlWriter(out_dir / "ingest.jsonl"),
                           "ingest")
        usable = {r["video_id"] for r in ingest if r["status"] in ("ok", "reused")}
        jobs = [
            (video, qa, mode)
            for video in videos if video.video_id in usable
            for qa in (video.qa_list[: args.max_qa] if args.max_qa > 0 else video.qa_list)
            for mode in modes
        ]
        answerer = {
            mode: Answerer(kernel, AnswerSettings(
                mode=mode, rounds=args.rounds, top_k=args.top_k,
                temperature=args.answer_temperature, max_tokens=args.answer_max_tokens,
            ))
            for mode in modes
        }
        records = _parallel(lambda job: _qa_one(answerer[job[2]], *job), jobs,
                            args.qa_concurrency, JsonlWriter(out_dir / "results.jsonl"), "qa")
    finally:
        runtime.close()

    metrics = {
        "run_id": args.run_id,
        "coverage": {
            **vars(coverage),
            "videos_selected": len(videos),
            "videos_ingested": len(usable),
            "videos_failed": sorted(r["video_id"] for r in ingest
                                    if r["status"] not in ("ok", "reused")),
            "questions_evaluated": len(jobs) // max(1, len(modes)),
        },
        "ingest_seconds": round(sum(r["seconds"] for r in ingest), 1),
        "modes": compute_metrics(records),
    }
    _write_json(out_dir / "metrics.json", metrics)
    meta["finished_at"] = datetime.now(UTC).isoformat()
    _write_json(out_dir / "run_meta.json", meta)
    for mode, stats in metrics["modes"].items():
        logger.info("[%s] accuracy %.2f%% (%d/%d), no answer %d, errors %d", mode,
                    100 * stats["accuracy"], stats["correct"], stats["questions"],
                    stats["no_answer"], stats["errors"])
        for qtype, row in stats["by_type"].items():
            logger.info("    %-30s %.2f%% (%d/%d)", qtype, 100 * row["accuracy"],
                        row["correct"], row["questions"])
    return 0 if not metrics["coverage"]["videos_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
