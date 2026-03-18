"""Image serving from DB."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response

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


@router.get("/api/images")
def list_images() -> JSONResponse:
    """List all uploaded images with their embedding status."""
    db.connect(reuse_if_open=True)
    try:
        from peewee import prefetch
        from app.models import Embedding
        
        images = Image.select().order_by(Image.created_at.desc())
        embeddings = Embedding.select(Embedding.image, Embedding.provider_name)
        images_with_embeddings = prefetch(images, embeddings)
        
        results = []
        for img in images_with_embeddings:
            provs = {e.provider_name for e in img.embeddings}
            results.append({
                "id": img.id,
                "filename": img.filename,
                "content_type": img.content_type,
                "created_at": img.created_at.isoformat() if img.created_at else None,
                "embeddings": {
                    "voyage_2048": "voyage_2048" in provs,
                    "jina_2048": "jina_2048" in provs,
                    "cohere_1536": "cohere_1536" in provs,
                }
            })
            
        return JSONResponse({"images": results})
    finally:
        if not db.is_closed():
            db.close()


@router.delete("/api/images/{image_id}")
def delete_image(image_id: int) -> JSONResponse:
    """Delete an image and its embeddings."""
    db.connect(reuse_if_open=True)
    try:
        img = Image.get_or_none(Image.id == image_id)
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        
        img.delete_instance(recursive=True)
        return JSONResponse({"status": "ok"})
    finally:
        if not db.is_closed():
            db.close()

@router.delete("/api/images/cleanup/missing-any")
def cleanup_missing_any() -> JSONResponse:
    """Delete all images that are missing at least one embedding."""
    db.connect(reuse_if_open=True)
    try:
        from peewee import fn
        from app.models import Embedding, Image

        # Subquery: image IDs that have exactly 3 embeddings
        valid_image_ids = Embedding.select(Embedding.image).group_by(Embedding.image).having(fn.COUNT(Embedding.id) == 3)
        
        # Select images to delete
        images_to_delete = Image.select().where(Image.id.not_in(valid_image_ids))
        
        count = 0
        for img in images_to_delete:
            img.delete_instance(recursive=True)
            count += 1
            
        return JSONResponse({"status": "ok", "deleted": count})
    finally:
        if not db.is_closed():
            db.close()

@router.delete("/api/images/cleanup/missing-all")
def cleanup_missing_all() -> JSONResponse:
    """Delete all images that have no embeddings at all."""
    db.connect(reuse_if_open=True)
    try:
        from peewee import fn
        from app.models import Embedding, Image

        # Subquery: image IDs that have at least 1 embedding
        valid_image_ids = Embedding.select(Embedding.image).group_by(Embedding.image).having(fn.COUNT(Embedding.id) > 0)
        
        # Select images to delete
        images_to_delete = Image.select().where(Image.id.not_in(valid_image_ids))
        
        count = 0
        for img in images_to_delete:
            img.delete_instance(recursive=True)
            count += 1
            
        return JSONResponse({"status": "ok", "deleted": count})
    finally:
        if not db.is_closed():
            db.close()
