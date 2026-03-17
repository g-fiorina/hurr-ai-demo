"""Peewee ORM models for the Hurr AI benchmark database."""

from __future__ import annotations

import datetime

from peewee import (
    AutoField,
    BlobField,
    CharField,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
)

from pgvector.peewee import VectorField

from app.config import PROVIDER_DIMENSIONS
from app.database import db


class BaseModel(Model):
    class Meta:
        database = db


class Image(BaseModel):
    id = AutoField()
    filename = CharField(max_length=512)
    image_data = BlobField()
    content_type = CharField(max_length=64, default="image/jpeg")
    created_at = DateTimeField(default=datetime.datetime.utcnow)

    class Meta:
        table_name = "images"


class Embedding(BaseModel):
    """One row per (image, provider) pair.

    Each provider has its own vector column so dimensions stay explicit.
    Only the column matching ``provider_name`` will be populated per row.
    """

    id = AutoField()
    image = ForeignKeyField(Image, backref="embeddings", on_delete="CASCADE")
    provider_name = CharField(max_length=32)
    # voyage_1024_vector = ...  # disabled – max search quality
    voyage_2048_vector = VectorField(dimensions=PROVIDER_DIMENSIONS["voyage_2048"], null=True)
    # jina_1024_vector = ...  # disabled – max search quality
    jina_2048_vector = VectorField(dimensions=PROVIDER_DIMENSIONS["jina_2048"], null=True)
    # cohere_1024_vector = ...  # disabled – max search quality
    cohere_1536_vector = VectorField(dimensions=PROVIDER_DIMENSIONS["cohere_1536"], null=True)

    class Meta:
        table_name = "embeddings"
        indexes = ((("image", "provider_name"), True),)

    @property
    def vector_column(self) -> str:
        return f"{self.provider_name}_vector"


class QueryHistory(BaseModel):
    id = AutoField()
    query_type = CharField(max_length=16)  # 'text' or 'image'
    query_content = CharField(max_length=4096)
    query_image_data = BlobField(null=True)
    query_image_content_type = CharField(max_length=64, null=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow)

    class Meta:
        table_name = "query_history"


class QueryResult(BaseModel):
    id = AutoField()
    query = ForeignKeyField(QueryHistory, backref="results", on_delete="CASCADE")
    provider_name = CharField(max_length=32)
    image = ForeignKeyField(Image, backref="query_results", on_delete="CASCADE")
    similarity_score = FloatField()
    rank = IntegerField()

    class Meta:
        table_name = "query_results"
