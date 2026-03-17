"""FastAPI application entry-point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR
from app.database import db, init_db
from app.routes.history import router as history_router
from app.routes.images import router as images_router
from app.routes.search import router as search_router
from app.routes.upload import router as upload_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    if not db.is_closed():
        db.close()
    yield
    if not db.is_closed():
        db.close()


app = FastAPI(title="Hurr AI Multimodal Embedding Benchmark", lifespan=lifespan)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.include_router(upload_router, tags=["upload"])
app.include_router(search_router, tags=["search"])
app.include_router(history_router, tags=["history"])
app.include_router(images_router, tags=["images"])


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request, "tab": "upload"})


@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request, "tab": "search"})


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request, "tab": "history"})
