"""Image serving from DB."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.database import db
from app.models import Image, QueryHistory

router = APIRouter()


@router.get("/images/{image_id}")
def get_image(image_id: int) -> Response:
    """Serve image from database."""
    db.connect(reuse_if_open=True)
    try:
        img = Image.get_or_none(Image.id == image_id)
        if not img or not img.image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        return Response(
            content=bytes(img.image_data),
            media_type=img.content_type or "image/jpeg",
        )
    finally:
        if not db.is_closed():
            db.close()


@router.get("/queries/{query_id}/query-image")
def get_query_image(query_id: int) -> Response:
    """Serve query image from database (for image searches)."""
    db.connect(reuse_if_open=True)
    try:
        q = QueryHistory.get_or_none(QueryHistory.id == query_id)
        if not q or not q.query_image_data or q.query_type != "image":
            raise HTTPException(status_code=404, detail="Query image not found")
        return Response(
            content=bytes(q.query_image_data),
            media_type=q.query_image_content_type or "image/jpeg",
        )
    finally:
        if not db.is_closed():
            db.close()
