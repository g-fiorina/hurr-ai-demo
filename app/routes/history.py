"""Query history route."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.database import db
from app.models import Image, QueryHistory, QueryResult

router = APIRouter()


@router.get("/api/history")
def get_history() -> JSONResponse:
    db.connect(reuse_if_open=True)
    try:
        queries = (
            QueryHistory
            .select()
            .order_by(QueryHistory.created_at.desc())
            .limit(50)
        )

        data: list[dict[str, Any]] = []
        for q in queries:
            results_by_provider: dict[str, list[dict[str, Any]]] = {}
            for qr in (
                QueryResult
                .select(QueryResult, Image)
                .join(Image, on=(QueryResult.image == Image.id))
                .where(QueryResult.query == q.id)
                .order_by(QueryResult.rank)
            ):
                entry = {
                    "image_id": qr.image.id,
                    "filename": qr.image.filename,
                    "similarity": round(float(qr.similarity_score), 4),
                    "rank": qr.rank,
                }
                results_by_provider.setdefault(qr.provider_name, []).append(entry)

            data.append({
                "id": q.id,
                "query_type": q.query_type,
                "query_content": q.query_content,
                "has_query_image": q.query_type == "image" and bool(q.query_image_data),
                "created_at": q.created_at.isoformat() if q.created_at else None,
                "results": results_by_provider,
            })

        return JSONResponse({"queries": data})
    finally:
        if not db.is_closed():
            db.close()
