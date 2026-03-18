"""Upload / ingestion route."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.ai_clients import embed_image
from app.database import db
from app.models import Embedding, Image

logger = logging.getLogger(__name__)
router = APIRouter()


ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}

CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
}


def _is_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTS


def _content_type(filename: str) -> str:
    return CONTENT_TYPES.get(Path(filename).suffix.lower(), "image/jpeg")


async def _ingest_image(image_bytes: bytes, filename: str) -> dict:
    """Store image in DB and generate embeddings from all providers."""
    content_type = _content_type(filename)
    img = Image.create(filename=filename, image_data=image_bytes, content_type=content_type)

    results = await embed_image(image_bytes)

    for res in results:
        data = {
            "image": img.id,
            "provider_name": res.provider,
            # "voyage_1024_vector": None,  # disabled – max search quality
            "voyage_2048_vector": None,
            # "jina_1024_vector": None,  # disabled – max search quality
            "jina_2048_vector": None,
            # "cohere_1024_vector": None,  # disabled – max search quality
            "cohere_1536_vector": None,
        }
        data[f"{res.provider}_vector"] = res.vector
        Embedding.create(**data)

    return {
        "id": img.id,
        "filename": img.filename,
        "providers_ok": [r.provider for r in results],
    }


@router.post("/upload")
async def upload_images(request: Request, files: list[UploadFile] = File(...)) -> JSONResponse:
    """Accept multiple images or a ZIP archive of images. All stored in DB."""
    results: list[dict] = []
    errors: list[str] = []

    db.connect(reuse_if_open=True)
    try:
        for file in files:
            if file.filename and file.filename.lower().endswith(".zip"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = Path(tmpdir) / "upload.zip"
                    with open(zip_path, "wb") as f:
                        f.write(await file.read())

                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(tmpdir)

                    async def _ingest_one(img_file: Path) -> None:
                        try:
                            image_bytes = img_file.read_bytes()
                            info = await _ingest_image(image_bytes, img_file.name)
                            results.append(info)
                        except Exception as exc:
                            logger.exception("Failed to ingest %s", img_file.name)
                            errors.append(f"{img_file.name}: {exc}")

                    tasks = [
                        _ingest_one(img_file)
                        for img_file in sorted(Path(tmpdir).rglob("*"))
                        if img_file.is_file() and _is_image(img_file.name) and not img_file.name.startswith("._")
                    ]
                    await asyncio.gather(*tasks)
            else:
                try:
                    image_bytes = await file.read()
                    filename = file.filename or "image.jpg"
                    info = await _ingest_image(image_bytes, filename)
                    results.append(info)
                except Exception as exc:
                    logger.exception("Failed to ingest %s", file.filename)
                    errors.append(f"{file.filename or 'file'}: {exc}")

        return JSONResponse(
            {"status": "ok", "ingested": results, "errors": errors},
            status_code=200 if results else 422,
        )
    finally:
        if not db.is_closed():
            db.close()
