"""Wrappers for Voyage AI, Jina AI, and Cohere multimodal embedding APIs."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

QueryInputType = Literal["text", "image"]


# All providers downsample above ~2M pixels; 1536px max side stays well within
# every provider's effective range while keeping payloads small.
_MAX_SIDE = 1536
_JPEG_QUALITY = 80


def _prepare_image(source: str | Path | bytes) -> Image.Image:
    """Open image from path or bytes, fix orientation, convert to RGB, resize."""
    from PIL import ImageOps

    if isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.LANCZOS)
    return img


def _to_data_uri(source: str | Path | bytes) -> str:
    """Convert path or bytes to a compressed data URI for embedding APIs."""
    img = _prepare_image(source)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


@dataclass
class EmbeddingResult:
    provider: str
    vector: list[float]


# ---------------------------------------------------------------------------
# Retry with exponential backoff (all providers)
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
BASE_DELAY = 10.0  # seconds
RETRYABLE_HTTP_STATUS = (429, 500, 502, 503, 504)

# Per-provider concurrency limits to avoid triggering rate limits
_sem_voyage = asyncio.Semaphore(5)
_sem_cohere = asyncio.Semaphore(5)


class _RateLimitedSemaphore:
    """Semaphore that enforces a minimum delay between acquisitions."""

    def __init__(self, concurrency: int, min_interval: float):
        self._sem = asyncio.Semaphore(concurrency)
        self._min_interval = min_interval
        self._last_release: float = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self._sem.acquire()
        async with self._lock:
            import time
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_release)
            if wait > 0:
                await asyncio.sleep(wait)
        return self

    async def __aexit__(self, *exc):
        import time
        self._last_release = time.monotonic()
        self._sem.release()


# 100 RPM = ~1.67 req/s → min 0.7s between requests (with safety margin)
_sem_jina = _RateLimitedSemaphore(concurrency=1, min_interval=0.7)


def _get_retry_delay(attempt: int, response: httpx.Response | None = None) -> float:
    """Exponential backoff with jitter. Honors Retry-After header for 429."""
    if response is not None and response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after) + random.uniform(0, 1)
            except (TypeError, ValueError):
                pass
    return BASE_DELAY * (2**attempt) + random.uniform(0, 1)


def _is_retryable_http(status_code: int) -> bool:
    return status_code in RETRYABLE_HTTP_STATUS


# ---------------------------------------------------------------------------
# Voyage AI  – voyage-multimodal-3.5  (REST API)
# ---------------------------------------------------------------------------

VOYAGE_API_URL = "https://api.voyageai.com/v1/multimodalembeddings"


async def _voyage_post(payload: dict) -> list[float]:
    """POST to Voyage API with retry on rate limit and server errors."""
    headers = {
        "Authorization": f"Bearer {settings.voyage_api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(VOYAGE_API_URL, json=payload, headers=headers)
                if _is_retryable_http(resp.status_code) and attempt < MAX_RETRIES - 1:
                    delay = _get_retry_delay(attempt, resp)
                    logger.info(
                        "Voyage %d, retry in %.1fs (attempt %d/%d)",
                        resp.status_code, delay, attempt + 1, MAX_RETRIES,
                    )
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp.json()["data"][0]["embedding"]
        except httpx.HTTPStatusError as e:
            if _is_retryable_http(e.response.status_code) and attempt < MAX_RETRIES - 1:
                last_error = e
                delay = _get_retry_delay(attempt, e.response)
                logger.info(
                    "Voyage %d, retry in %.1fs (attempt %d/%d)",
                    e.response.status_code, delay, attempt + 1, MAX_RETRIES,
                )
                await asyncio.sleep(delay)
                continue
            raise
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = _get_retry_delay(attempt)
                logger.info("Voyage error %s, retry in %.1fs (attempt %d/%d)", e, delay, attempt + 1, MAX_RETRIES)
                await asyncio.sleep(delay)
                continue
            raise
    raise last_error or RuntimeError("Voyage request failed after retries")


def _voyage_payload(content: list[dict], input_type: str, output_dimension: int) -> dict:
    return {
        "inputs": [{"content": content}],
        "model": "voyage-multimodal-3.5",
        "input_type": input_type,
        "output_dimension": output_dimension,
    }


async def _embed_voyage_image(
    source: str | Path | bytes,
    dimensions: int,
    *,
    input_type: str = "document",
) -> list[float]:
    """input_type: 'document' for documents to index, 'query' for search queries."""
    data_uri = _to_data_uri(source)
    content = [{"type": "image_base64", "image_base64": data_uri}]
    payload = _voyage_payload(content, input_type, dimensions)
    return await _voyage_post(payload)


async def _embed_voyage_text(text: str, dimensions: int) -> list[float]:
    content = [{"type": "text", "text": text}]
    payload = _voyage_payload(content, "query", dimensions)
    return await _voyage_post(payload)


# ---------------------------------------------------------------------------
# Jina AI  – jina-embeddings-v4  (REST API)
# ---------------------------------------------------------------------------

JINA_API_URL = "https://api.jina.ai/v1/embeddings"


async def _jina_post(payload: dict) -> list[float]:
    """POST to Jina API with retry on rate limit and server errors."""
    headers = {"Authorization": f"Bearer {settings.jina_api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(JINA_API_URL, json=payload, headers=headers)
                if _is_retryable_http(resp.status_code) and attempt < MAX_RETRIES - 1:
                    delay = _get_retry_delay(attempt, resp)
                    logger.info(
                        "Jina %d, retry in %.1fs (attempt %d/%d)",
                        resp.status_code, delay, attempt + 1, MAX_RETRIES,
                    )
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp.json()["data"][0]["embedding"]
        except httpx.HTTPStatusError as e:
            if _is_retryable_http(e.response.status_code) and attempt < MAX_RETRIES - 1:
                last_error = e
                delay = _get_retry_delay(attempt, e.response)
                logger.info(
                    "Jina %d, retry in %.1fs (attempt %d/%d)",
                    e.response.status_code, delay, attempt + 1, MAX_RETRIES,
                )
                await asyncio.sleep(delay)
                continue
            raise
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = _get_retry_delay(attempt)
                logger.info("Jina error %s, retry in %.1fs (attempt %d/%d)", e, delay, attempt + 1, MAX_RETRIES)
                await asyncio.sleep(delay)
                continue
            raise
    raise last_error or RuntimeError("Jina request failed after retries")


JINA_MAX_PIXELS = 3_000_000


def _to_data_uri_jina(source: str | Path | bytes) -> str:
    """Convert to data URI, downscaling if needed to stay under JINA_MAX_PIXELS."""
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    if w * h > JINA_MAX_PIXELS:
        ratio = (JINA_MAX_PIXELS / (w * h)) ** 0.5
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


async def _embed_jina_image(source: str | Path | bytes, dimensions: int, *, task: str = "retrieval.passage") -> list[float]:
    """task: 'retrieval.passage' for documents to index, 'retrieval.query' for search queries."""
    data_uri = _to_data_uri_jina(source)
    payload = {
        "model": "jina-embeddings-v4",
        "input": [{"image": data_uri}],
        "task": task,
        "dimensions": dimensions,
    }
    return await _jina_post(payload)


async def _embed_jina_text(text: str, dimensions: int) -> list[float]:
    payload = {
        "model": "jina-embeddings-v4",
        "input": [{"text": text}],
        "task": "retrieval.query",
        "dimensions": dimensions,
    }
    return await _jina_post(payload)


# ---------------------------------------------------------------------------
# Cohere  – embed-v4.0
# ---------------------------------------------------------------------------

def _is_retryable_cohere_error(exc: Exception) -> bool:
    """Check if Cohere API error is retryable (rate limit or server error)."""
    status = getattr(exc, "status_code", None)
    return status in RETRYABLE_HTTP_STATUS if status is not None else False


async def _embed_cohere_image(
    source: str | Path | bytes,
    dimensions: int,
    *,
    input_type: str = "search_document",
) -> list[float]:
    """input_type: 'search_document' for documents to index, 'search_query' for search queries."""
    import cohere  # type: ignore[import-untyped]

    co = cohere.ClientV2(api_key=settings.cohere_api_key)
    data_uri = _to_data_uri(source)
    loop = asyncio.get_event_loop()

    def _call(data: str):
        return co.embed(
            model="embed-v4.0",
            input_type=input_type,
            embedding_types=["float"],
            images=[data],
            output_dimension=dimensions,
        )

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            result = await loop.run_in_executor(None, lambda d=data_uri: _call(d))
            return list(result.embeddings.float_[0])
        except Exception as exc:
            last_error = exc
            if _is_retryable_cohere_error(exc) and attempt < MAX_RETRIES - 1:
                delay = _get_retry_delay(attempt)
                logger.info(
                    "Cohere %s, retry in %.1fs (attempt %d/%d)",
                    exc, delay, attempt + 1, MAX_RETRIES,
                )
                await asyncio.sleep(delay)
                continue
            if _is_retryable_cohere_error(exc):
                raise  # Exhausted retries on rate limit/server error
            raise  # Non-retryable error
    raise last_error or RuntimeError("Cohere image embed failed after retries")


async def _embed_cohere_text(text: str, dimensions: int) -> list[float]:
    import cohere  # type: ignore[import-untyped]

    co = cohere.ClientV2(api_key=settings.cohere_api_key)
    loop = asyncio.get_event_loop()
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            result = await loop.run_in_executor(
                None,
                lambda: co.embed(
                    model="embed-v4.0",
                    input_type="search_query",
                    embedding_types=["float"],
                    texts=[text],
                    output_dimension=dimensions,
                ),
            )
            return list(result.embeddings.float_[0])
        except Exception as exc:
            last_error = exc
            if _is_retryable_cohere_error(exc) and attempt < MAX_RETRIES - 1:
                delay = _get_retry_delay(attempt)
                logger.info(
                    "Cohere text %s, retry in %.1fs (attempt %d/%d)",
                    exc, delay, attempt + 1, MAX_RETRIES,
                )
                await asyncio.sleep(delay)
                continue
            raise
    raise last_error or RuntimeError("Cohere text embed failed after retries")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

async def embed_image(source: str | Path | bytes) -> list[EmbeddingResult]:
    """Embed a single image (path or bytes) with all three providers concurrently."""
    results: list[EmbeddingResult] = []
    errors: list[str] = []

    async def _run(name: str, coro, sem: asyncio.Semaphore):
        try:
            async with sem:
                vec = await coro
            results.append(EmbeddingResult(provider=name, vector=vec))
        except Exception as exc:
            logger.exception("Embedding failed for %s", name)
            errors.append(f"{name}: {exc}")

    await asyncio.gather(
        _run("voyage_2048", _embed_voyage_image(source, 2048), _sem_voyage),
        _run("jina_2048", _embed_jina_image(source, 2048), _sem_jina),
        _run("cohere_1536", _embed_cohere_image(source, 1536), _sem_cohere),
    )

    if errors:
        logger.warning("Some providers failed: %s", "; ".join(errors))

    return results


async def embed_query(
    query_type: QueryInputType,
    value: str | bytes,
) -> list[EmbeddingResult]:
    """Embed a search query (text or image bytes) with all three providers."""
    results: list[EmbeddingResult] = []
    errors: list[str] = []

    async def _run(name: str, coro, sem: asyncio.Semaphore):
        try:
            async with sem:
                vec = await coro
            results.append(EmbeddingResult(provider=name, vector=vec))
        except Exception as exc:
            logger.exception("Query embedding failed for %s", name)
            errors.append(f"{name}: {exc}")

    if query_type == "text":
        assert isinstance(value, str)
        await asyncio.gather(
            _run("voyage_2048", _embed_voyage_text(value, 2048), _sem_voyage),
            _run("jina_2048", _embed_jina_text(value, 2048), _sem_jina),
            _run("cohere_1536", _embed_cohere_text(value, 1536), _sem_cohere),
        )
    else:
        await asyncio.gather(
            _run("voyage_2048", _embed_voyage_image(value, 2048, input_type="query"), _sem_voyage),
            _run("jina_2048", _embed_jina_image(value, 2048, task="retrieval.query"), _sem_jina),
            _run("cohere_1536", _embed_cohere_image(value, 1536, input_type="search_query"), _sem_cohere),
        )

    if errors:
        logger.warning("Some providers failed: %s", "; ".join(errors))

    return results
