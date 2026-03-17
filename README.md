# Hurr AI – Multimodal Embedding Benchmark

A web application for benchmarking multimodal embedding providers (Voyage AI, Jina AI, Cohere) side-by-side. Upload images, generate embeddings from all three APIs, then search by text or image to compare retrieval quality.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker & Docker Compose (for PostgreSQL + pgvector)

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```
VOYAGE_API_KEY=<your-voyage-key>
JINA_API_KEY=<your-jina-key>
COHERE_API_KEY=<your-cohere-key>
```

### 3. Start everything

```bash
docker compose up
```

This spins up PostgreSQL 18 with pgvector on port 5432 and the FastAPI app on port 8000.

Open [http://localhost:8000](http://localhost:8000) in your browser.

**Alternative (local dev with hot reload):** Run the database only with `docker compose up db`, then in another terminal run `uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`.

The database tables and pgvector extension are created automatically on first startup.

## Usage

### Upload Tab

Upload individual images (JPEG, PNG, WebP, etc.) or a ZIP archive. Each image is embedded concurrently by all three providers and stored in the database.

### Search Tab

Enter a text query or upload an image. The query is embedded by all three providers, and the top-X most similar catalogue images are returned per provider, displayed side-by-side for comparison.

### History Tab

Browse past queries. Click any entry to expand and see the comparative results that were generated at query time.

## Architecture

```
Hurr_ai/
├── app/
│   ├── main.py            # FastAPI app & routes
│   ├── config.py          # Settings & constants
│   ├── database.py        # Peewee + pgvector setup
│   ├── models.py          # ORM models
│   ├── ai_clients.py      # Voyage / Jina / Cohere wrappers
│   ├── routes/
│   │   ├── upload.py      # POST /upload
│   │   ├── search.py      # POST /search
│   │   ├── history.py     # GET /api/history
│   │   └── images.py      # GET /images/{id}, GET /queries/{id}/query-image
│   └── templates/
│       └── index.html     # SPA-style UI with TailwindCSS
├── docker-compose.yml     # PostgreSQL + pgvector
├── pyproject.toml         # Dependencies (uv)
└── .env.example           # Environment template
```

Images are stored in the database (PostgreSQL BYTEA), not on the filesystem. This simplifies deployment and backups.

## Embedding Models

| Provider   | Model                   | Dimensions |
|------------|-------------------------|------------|
| Voyage AI  | `voyage-multimodal-3.5`  | 2048       |
| Jina AI    | `jina-embeddings-v4`     | 2048       |
| Cohere     | `embed-v4.0`            | 1536       |

Max dimensions for each provider to optimize search quality.

## Tech Stack

- **Backend:** FastAPI, Peewee ORM, pgvector
- **Frontend:** Jinja2, TailwindCSS (CDN), Vanilla JS
- **Database:** PostgreSQL 18 + pgvector (images stored as BYTEA)
- **AI:** httpx (Voyage, Jina REST APIs), Cohere SDK
