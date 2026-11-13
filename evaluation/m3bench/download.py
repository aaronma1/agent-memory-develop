"""Download M3-Bench web subset videos (YouTube), with pause and resume.

The M3-Bench (ICLR 2026) web subset only publishes YouTube links (`data/annotations/web.json`
on the master branch of GitHub `ByteDance-Seed/m3-agent`); the videos must be downloaded locally.

Usage:
  .venv/bin/python -m evaluation.m3bench.download --out-dir ~/data/m3bench/web

Pause and resume:
  - Create a `PAUSE` file in `<out-dir>`, or press Ctrl-C once: exits after the current video;
    press Ctrl-C again to exit immediately (yt-dlp resumes the `.part` file next time).
  - Rerun the same command to resume: videos already `ok` or permanently unavailable in
    `manifest.jsonl` are skipped.

When YouTube rate-limits ("Sign in to confirm you're not a bot" / HTTP 429) it backs off
exponentially; after repeated blocks it exits and suggests retrying later or using
`--cookies-from-browser`.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/ByteDance-Seed/m3-agent/master/"
    "data/annotations/web.json"
)
# Prefer YouTube progressive mp4 (format 18, 360p audio+video), so no local ffmpeg merge.
DEFAULT_FORMAT = "18/b[height<=480][vcodec!=none][acodec!=none]"

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"  # permanent: private, removed, region-locked
STATUS_BLOCKED = "blocked"  # rate-limited / bot check, retryable
STATUS_ERROR = "error"  # other failure, retryable

_UNAVAILABLE_MARKERS = (
    "Private video",
    "This video is not available",
    "Video unavailable",
    "has not made this video available in your country",
    "account associated with this video has been terminated",
    "This video has been removed",
)
_BLOCKED_MARKERS = (
    "confirm you’re not a bot",
    "confirm you're not a bot",
    "HTTP Error 429",
    "Too Many Requests",
    "Please sign in",
)


@dataclass(frozen=True)
class Video:
    video_id: str
    url: str
    qa_count: int


@dataclass(frozen=True)
class Outcome:
    status: str
    message: str = ""
    path: str = ""
    size_bytes: int = 0


class StopFlag:
    """First SIGINT/SIGTERM requests a graceful stop; the second exits immediately."""

    def __init__(self) -> None:
        self.requested = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        del frame
        if self.requested:
            print("\nsecond interrupt: exiting now", file=sys.stderr)
            raise SystemExit(128 + signum)
        self.requested = True
        print("\nstop requested: finishing current video, then exiting", file=sys.stderr)


def load_annotations(out_dir: Path, annotations: Path | None) -> dict[str, dict]:
    path = annotations or out_dir / "annotations" / "web.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching annotations -> {path}")
        with urllib.request.urlopen(ANNOTATIONS_URL, timeout=60) as resp:
            path.write_bytes(resp.read())
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"unexpected annotations format in {path}")
    return data


def videos_from(data: dict[str, dict]) -> list[Video]:
    return [
        Video(video_id=vid, url=str(item["video_url"]), qa_count=len(item.get("qa_list", [])))
        for vid, item in data.items()
    ]


def load_manifest(path: Path) -> dict[str, dict]:
    """Return the latest row per video (later rows for a video override earlier ones)."""
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "video_id" in row:
                latest[row["video_id"]] = row
    return latest


def append_manifest(path: Path, video: Video, outcome: Outcome) -> None:
    row = {
        "video_id": video.video_id,
        "status": outcome.status,
        "path": outcome.path,
        "size_bytes": outcome.size_bytes,
        "qa_count": video.qa_count,
        "message": outcome.message[-500:],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify(stderr: str) -> str:
    if any(marker in stderr for marker in _UNAVAILABLE_MARKERS):
        return STATUS_UNAVAILABLE
    if any(marker in stderr for marker in _BLOCKED_MARKERS):
        return STATUS_BLOCKED
    return STATUS_ERROR


def find_output(videos_dir: Path, video_id: str) -> Path | None:
    for candidate in sorted(videos_dir.glob(f"{video_id}.*")):
        if candidate.suffix not in {".part", ".ytdl", ".temp"}:
            return candidate
    return None


def download_one(video: Video, videos_dir: Path, args: argparse.Namespace) -> Outcome:
    existing = find_output(videos_dir, video.video_id)
    if existing is not None and existing.stat().st_size > 0:
        return Outcome(STATUS_OK, "already on disk", str(existing), existing.stat().st_size)
    cmd = [
        args.yt_dlp,
        "--no-playlist",
        "--no-progress",
        "--continue",
        "--retries", "3",
        "-f", args.format,
        "-o", str(videos_dir / "%(id)s.%(ext)s"),
    ]
    if args.deno:
        cmd += ["--js-runtimes", f"deno:{args.deno}"]
    if args.cookies_from_browser:
        cmd += ["--cookies-from-browser", args.cookies_from_browser]
    if args.cookies:
        cmd += ["--cookies", str(args.cookies)]
    if args.rate_limit:
        cmd += ["--limit-rate", args.rate_limit]
    cmd.append(video.url)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return Outcome(STATUS_ERROR, f"timeout after {args.timeout}s")
    output = find_output(videos_dir, video.video_id)
    if proc.returncode == 0 and output is not None:
        return Outcome(STATUS_OK, "", str(output), output.stat().st_size)
    stderr = proc.stderr.strip() or proc.stdout.strip()
    return Outcome(classify(stderr), stderr)


def summarize(manifest: dict[str, dict], videos: list[Video]) -> str:
    counts: dict[str, int] = {}
    size = 0
    qa_ok = 0
    by_id = {v.video_id: v for v in videos}
    for vid, row in manifest.items():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if row["status"] == STATUS_OK:
            size += int(row.get("size_bytes", 0))
            qa_ok += by_id[vid].qa_count if vid in by_id else 0
    pending = len(videos) - sum(counts.values())
    qa_total = sum(v.qa_count for v in videos)
    parts = ", ".join(f"{k}={n}" for k, n in sorted(counts.items()))
    return (
        f"{parts or 'nothing yet'}, not tried={pending} | "
        f"{size / 1e9:.1f} GB on disk | QAs covered {qa_ok}/{qa_total}"
    )


def sleep_interruptible(seconds: float, stop: StopFlag, pause_file: Path) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop.requested or pause_file.exists():
            return
        time.sleep(min(1.0, deadline - time.monotonic()))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--annotations", type=Path, default=None,
                   help="local web.json (default: <out-dir>/annotations/web.json, "
                        "fetched if missing)")
    p.add_argument("--format", default=DEFAULT_FORMAT)
    p.add_argument("--yt-dlp", default=shutil.which("yt-dlp") or "yt-dlp")
    p.add_argument("--deno", default=shutil.which("deno") or "",
                   help="path to deno for YouTube JS challenges (default: from PATH)")
    p.add_argument("--cookies-from-browser", default="",
                   help="e.g. 'firefox' or 'chrome'; use a throwaway account")
    p.add_argument("--cookies", type=Path, default=None, help="Netscape cookies.txt")
    p.add_argument("--rate-limit", default="", help="yt-dlp --limit-rate, e.g. 5M")
    p.add_argument("--sleep-min", type=float, default=8.0)
    p.add_argument("--sleep-max", type=float, default=20.0)
    p.add_argument("--block-backoff", type=float, default=600.0,
                   help="first wait (s) after being blocked; doubles each time")
    p.add_argument("--max-blocks", type=int, default=4,
                   help="stop after this many consecutive blocked videos")
    p.add_argument("--timeout", type=int, default=1800, help="per-video timeout (s)")
    p.add_argument("--max-videos", type=int, default=0, help="stop after N attempts (0 = all)")
    p.add_argument("--retry-unavailable", action="store_true",
                   help="also retry videos previously marked permanently unavailable")
    p.add_argument("--status", action="store_true", help="print progress and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir.expanduser()
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    pause_file = out_dir / "PAUSE"

    videos = videos_from(load_annotations(out_dir, args.annotations))
    manifest = load_manifest(manifest_path)
    print(summarize(manifest, videos))
    if args.status:
        return 0
    if pause_file.exists():
        print(f"{pause_file} exists: delete it to resume")
        return 0

    skip = {STATUS_OK} if args.retry_unavailable else {STATUS_OK, STATUS_UNAVAILABLE}
    todo = [v for v in videos if manifest.get(v.video_id, {}).get("status") not in skip]
    print(f"{len(todo)} videos to try (pause: touch {pause_file} or Ctrl-C)")

    stop = StopFlag()
    consecutive_blocks = 0
    backoff = args.block_backoff
    attempts = 0
    for video in todo:
        if stop.requested or pause_file.exists():
            break
        if args.max_videos and attempts >= args.max_videos:
            break
        attempts += 1
        outcome = download_one(video, videos_dir, args)
        append_manifest(manifest_path, video, outcome)
        manifest[video.video_id] = {"status": outcome.status, "size_bytes": outcome.size_bytes}
        size_mb = outcome.size_bytes / 1e6
        print(f"[{attempts}/{len(todo)}] {video.video_id}: {outcome.status}"
              + (f" ({size_mb:.0f} MB)" if outcome.status == STATUS_OK else "")
              + (f" - {outcome.message.splitlines()[-1][:120]}" if outcome.message
                 and outcome.status != STATUS_OK else ""))

        if outcome.status == STATUS_BLOCKED:
            consecutive_blocks += 1
            if consecutive_blocks >= args.max_blocks:
                print(f"blocked {consecutive_blocks} times in a row: stopping. "
                      "Wait an hour or more, or rerun with --cookies-from-browser.")
                break
            print(f"blocked: backing off {backoff / 60:.0f} min")
            sleep_interruptible(backoff, stop, pause_file)
            backoff *= 2
            continue
        consecutive_blocks = 0
        backoff = args.block_backoff
        sleep_interruptible(random.uniform(args.sleep_min, args.sleep_max), stop, pause_file)

    print(summarize(load_manifest(manifest_path), videos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
