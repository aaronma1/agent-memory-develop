"""M3-Bench eval-only plugins: silent ASR and a video normalization cache.

Both are registered only inside the eval process; they are not part of the kernel and do not
change its default assembly.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jiuwen_memory.common.base import PluginType
from jiuwen_memory.common.errors import ValidationError
from jiuwen_memory.common.normalizer.base import (
    LocalMediaPreflight,
    Normalizer,
    NormalizerProducer,
)
from jiuwen_memory.common.normalizer.normalizer_impl.video_asr import (
    VideoAsrProducer,
    VideoAsrService,
)
from jiuwen_memory.common.type_def import Modality, RawPayload

SILENT_ASR_TARGET = "m3bench_silent"
CACHED_VIDEO_TARGET = "m3bench_cached_video"


class SilentVideoAsr(VideoAsrService):
    """Placeholder when no ASR endpoint is configured: returns an empty transcript, so the
    pipeline falls back to the visual stream only (see F05)."""

    def plugin_type(self) -> PluginType:
        return PluginType.ASR

    def health(self) -> None:
        return None

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None,
        chunk_seconds: int,
    ) -> list[dict[str, str]]:
        del audio_path, language, chunk_seconds
        return []


class CachedVideoNormalizer(Normalizer):
    """Cache the JSON envelope produced by video normalization, keyed by ``payload.id``.

    Normalization (per-clip VLM + ASR) is the most expensive step; with the cache, reruns only
    redo construction and retrieval, so changing retrieval/answer settings does not call the
    vision model again. run.py keeps a separate cache directory per VLM/ASR combination.
    """

    def __init__(self, inner: Normalizer, cache_dir: Path) -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def modalities(self) -> list[Modality]:
        return self._inner.modalities()

    def structured_modalities(self) -> frozenset[Modality]:
        return self._inner.structured_modalities()

    @staticmethod
    def plugin_type() -> PluginType:
        return PluginType.NORMALIZER

    def health(self) -> None:
        self._inner.health()

    def local_media_preflight(self, modality: Modality) -> LocalMediaPreflight | None:
        return self._inner.local_media_preflight(modality)

    def cache_path(self, payload_id: str) -> Path:
        return self._cache_dir / f"{payload_id}.json"

    def normalize(self, payload: RawPayload) -> str:
        path = self.cache_path(payload.id) if payload.id else None
        if path is not None and path.is_file():
            return path.read_text(encoding="utf-8")
        envelope = self._inner.normalize(payload)
        if path is not None:
            _atomic_write(path, envelope)
        return envelope


def _atomic_write(path: Path, text: str) -> None:
    json.loads(text)  # cache only valid envelopes
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@VideoAsrProducer.register(SILENT_ASR_TARGET)
def _build_silent_asr(config: Mapping[str, Any]) -> VideoAsrService:
    del config
    return SilentVideoAsr()


@NormalizerProducer.register(CACHED_VIDEO_TARGET)
def _build_cached_video(config) -> Normalizer:
    cache_dir = str(config.get("cache_dir", "")).strip()
    if not cache_dir:
        raise ValidationError(f"{CACHED_VIDEO_TARGET} requires cache_dir")
    inner = config.get("inner")
    if not isinstance(inner, Mapping) or not str(inner.get("target", "")).strip():
        raise ValidationError(f"{CACHED_VIDEO_TARGET} requires an inner component mapping")
    normalizer = NormalizerProducer.build(
        str(inner["target"]), inner.get("params", {}), config.ctx,
    )
    return CachedVideoNormalizer(normalizer, Path(cache_dir).expanduser())
