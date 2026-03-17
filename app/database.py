"""Database connection and pgvector setup using peewee."""

from __future__ import annotations

from playhouse.postgres_ext import PostgresqlExtDatabase

from app.config import settings

db = PostgresqlExtDatabase(None, autoconnect=False)
db.init(settings.database_url)


def init_db() -> None:
    """Create pgvector extension and all tables."""
    db.connect(reuse_if_open=True)
    db.execute_sql("CREATE EXTENSION IF NOT EXISTS vector;")

    from app.models import Embedding, Image, QueryHistory, QueryResult

    cursor = db.execute_sql(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'images' AND column_name = 'filepath'"
    )
    if cursor.fetchone():
        db.execute_sql("DROP TABLE IF EXISTS query_results, embeddings, query_history, images CASCADE;")
    for old_col in ("voyage_vector", "jina_vector", "cohere_vector", "voyage_1024_vector", "jina_1024_vector", "cohere_1024_vector"):
        cursor = db.execute_sql(
            "SELECT 1 FROM information_schema.columns "
            f"WHERE table_name = 'embeddings' AND column_name = '{old_col}'"
        )
        if cursor.fetchone():
            db.execute_sql("DROP TABLE IF EXISTS query_results, embeddings CASCADE;")
            break

    db.create_tables([Image, Embedding, QueryHistory, QueryResult], safe=True)
