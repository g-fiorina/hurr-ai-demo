"""Search / compare route."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app.ai_clients import embed_query
from app.database import db
from app.models import Embedding, Image, QueryHistory, QueryResult

logger = logging.getLogger(__name__)
router = APIRouter()

TOP_K = 10

VECTOR_FIELD_MAP = {
    # "voyage_1024": Embedding.voyage_1024_vector,  # disabled – max search quality
    "voyage_2048": Embedding.voyage_2048_vector,
    # "jina_1024": Embedding.jina_1024_vector,  # disabled – max search quality
    "jina_2048": Embedding.jina_2048_vector,
    # "cohere_1024": Embedding.cohere_1024_vector,  # disabled – max search quality
    "cohere_1536": Embedding.cohere_1536_vector,
}

CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _content_type(filename: str) -> str:
    return CONTENT_TYPES.get(Path(filename).suffix.lower(), "image/jpeg")


def _vector_search(provider: str, query_vector: list[float], top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Cosine-similarity search via pgvector's <=> operator on the VectorField."""
    vec_field = VECTOR_FIELD_MAP[provider]
    cosine_dist = vec_field.cosine_distance(query_vector)

    filtered = (
        Embedding
        .select(Embedding.id, Embedding.image)
        .where(
            (Embedding.provider_name == provider) &
            (vec_field.is_null(False))
        )
    )

    results = (
        filtered
        .select(
            Embedding.image,
            cosine_dist.alias("distance"),
        )
        .order_by(cosine_dist.asc())
        .limit(top_k)
    )

    hits = list(results.execute())

    rows: list[dict[str, Any]] = []
    for hit in hits:
        img = Image.get_by_id(hit.image)
        similarity = 1.0 - float(hit.distance)
        rows.append({
            "image_id": img.id,
            "filename": img.filename,
            "similarity": round(similarity, 4),
        })
    return rows


@router.post("/search")
async def search(
    query_text: str | None = Form(default=None),
    query_image: UploadFile | None = File(default=None),
) -> JSONResponse:
    db.connect(reuse_if_open=True)
    try:
        if query_image and query_image.filename:
            query_type = "image"
            image_bytes = await query_image.read()
            query_value = image_bytes
            query_display = "image"
        elif query_text and query_text.strip():
            query_type = "text"
            query_value = query_text.strip()
            query_display = query_value
        else:
            return JSONResponse({"error": "Provide either query_text or query_image"}, status_code=400)

        embeddings = await embed_query(query_type, query_value)

        if query_type == "image" and query_image and query_image.filename:
            qh = QueryHistory.create(
                query_type=query_type,
                query_content="image",
                query_image_data=image_bytes,
                query_image_content_type=_content_type(query_image.filename),
            )
        else:
            qh = QueryHistory.create(query_type=query_type, query_content=query_display)

        provider_results: dict[str, list[dict[str, Any]]] = {}
        for emb in embeddings:
            hits = _vector_search(emb.provider, emb.vector)
            provider_results[emb.provider] = hits
            for rank_idx, hit in enumerate(hits):
                QueryResult.create(
                    query=qh.id,
                    provider_name=emb.provider,
                    image=hit["image_id"],
                    similarity_score=hit["similarity"],
                    rank=rank_idx + 1,
                )

        return JSONResponse({
            "query_id": qh.id,
            "query_type": query_type,
            "query_content": query_display,
            "results": provider_results,
        })
    finally:
        if not db.is_closed():
            db.close()
