"""Protocol-level self-test of the three vLLM services started by serve.sh.

Checks the request shapes run.py and the kernel actually use, not just "the server responds":
  - chat: plain text completion; ``video_url`` + base64 data URL (as video_pipeline._offline_vu)
  - asr: ``response_format=verbose_json`` + ``timestamp_granularities=[segment]``; the response
    must carry a ``segments`` list (as video_asr.OpenAITranscriptionVideoAsr)
  - embed: ``/v1/embeddings`` returns 1024-dim vectors (bge-m3, matching the Milvus dimension)
Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import struct
import sys
import tempfile
import wave
from pathlib import Path

import openai


def _video_data_url() -> str:
    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (320, 240))
        for i in range(12):  # 3 seconds: first half red, second half blue
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:] = (0, 0, 255) if i < 6 else (255, 0, 0)
            cv2.putText(frame, "M3", (110, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            writer.write(frame)
        writer.release()
        data = path.read_bytes()
    return "data:video/mp4;base64," + base64.b64encode(data).decode("ascii")


def _wav_bytes(seconds: float = 3.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440 * n / rate)))
            for n in range(int(seconds * rate))
        )
        wav.writeframes(frames)
    return buf.getvalue()


def check_chat(client: openai.OpenAI, model: str) -> None:
    text = client.chat.completions.create(
        model=model, max_tokens=8, temperature=0,
        messages=[{"role": "user", "content": "Reply with the single word OK."}],
    ).choices[0].message.content or ""
    assert text.strip(), "chat returned empty text"
    video = client.chat.completions.create(
        model=model, max_tokens=32, temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": _video_data_url()}},
            {"type": "text", "text": "What background colors appear in this video?"},
        ]}],
    ).choices[0].message.content or ""
    assert video.strip(), "video_url request returned empty text"
    print(f"chat  OK  text={text.strip()!r} video={video.strip()[:80]!r}")


def check_asr(client: openai.OpenAI, model: str) -> None:
    response = client.audio.transcriptions.create(
        model=model,
        file=("probe.wav", _wav_bytes(), "audio/wav"),
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )
    payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    segments = payload.get("segments")
    assert isinstance(segments, list), (
        "transcription response has no segments list: this vLLM version lacks verbose_json "
        "segment timestamps, so the video ASR adapter cannot use it; upgrade vLLM or leave "
        "M3BENCH_ASR_BASE_URL unset"
    )
    print(f"asr   OK  segments={len(segments)}")


def check_embed(client: openai.OpenAI, model: str, dim: int) -> None:
    # The second input ("video memory" in Chinese) checks non-English text.
    data = client.embeddings.create(model=model, input=["video memory", "视频记忆"]).data
    sizes = {len(item.embedding) for item in data}
    assert sizes == {dim}, f"embedding dim {sizes} != {dim}"
    print(f"embed OK  dim={dim}")


def main() -> int:
    p = argparse.ArgumentParser()
    for name in ("chat", "asr", "embed"):
        p.add_argument(f"--{name}", required=True)
        p.add_argument(f"--{name}-model", required=True)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--embed-dim", type=int, default=1024)
    args = p.parse_args()

    def client(url: str) -> openai.OpenAI:
        return openai.OpenAI(base_url=url, api_key=args.api_key, timeout=300, max_retries=0)

    checks = (
        ("chat", lambda: check_chat(client(args.chat), args.chat_model)),
        ("asr", lambda: check_asr(client(args.asr), args.asr_model)),
        ("embed", lambda: check_embed(client(args.embed), args.embed_model, args.embed_dim)),
    )
    failed = []
    for name, check in checks:
        try:
            check()
        except Exception as exc:  # collect every failure before exiting
            failed.append(name)
            print(f"{name:5} FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
