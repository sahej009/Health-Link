"""
Simple KB -> Pinecone uploader.
Creates index if missing, chunks + uploads data/symptoms_kb.json, prints index name.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import get_settings
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


def record_to_text(record: dict) -> str:
    symptom = record.get("symptom", "")
    category = record.get("category", "")
    description = record.get("description", "")
    urgency = record.get("urgency", "")
    common_causes = ", ".join(record.get("common_causes", []))
    red_flags = ", ".join(record.get("red_flags", []))
    recommended_actions = ", ".join(record.get("recommended_actions", []))
    specialty = ", ".join(record.get("specialty", []))

    return (
        f"Symptom: {symptom}\n"
        f"Category: {category}\n"
        f"Description: {description}\n"
        f"Urgency: {urgency}\n"
        f"Specialty: {specialty}\n"
        f"Common Causes: {common_causes}\n"
        f"Red Flags: {red_flags}\n"
        f"Recommended Actions: {recommended_actions}"
    ).strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def main() -> None:
    kb_file = ROOT_DIR / "data" / "symptoms_kb.json"
    if not kb_file.exists():
        raise FileNotFoundError(f"Missing file: {kb_file}")

    settings = get_settings()
    with kb_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("symptoms_kb.json must be a JSON list of symptom records")

    model_name = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = SentenceTransformer(model_name)

    docs: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = record_to_text(row)
        if not text:
            continue
        chunks = chunk_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        for ch in chunks:
            docs.append(
                {
                    "content": ch,
                    "metadata": {
                        "symptom": row.get("symptom", ""),
                        "category": row.get("category", ""),
                    },
                }
            )

    if not docs:
        raise ValueError("No chunks created from symptoms_kb.json")

    texts = [d["content"] for d in docs]
    vectors = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    vector_dim = len(vectors[0])

    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing_indexes:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=vector_dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=settings.pinecone_environment or "us-east-1",
            ),
        )

    index = pc.Index(settings.pinecone_index_name)
    index_stats = index.describe_index_stats()
    if index_stats.dimension != vector_dim:
        raise ValueError(
            f"Index dimension mismatch: index={index_stats.dimension}, embedding_model={vector_dim}. "
            "Use a new PINECONE_INDEX_NAME for this embedding model."
        )

    pinecone_vectors = []
    for i, (doc, emb) in enumerate(zip(docs, vectors)):
        pinecone_vectors.append(
            {
                "id": f"symptom_chunk_{i}",
                "values": emb.tolist(),
                "metadata": {
                    "content": doc["content"],
                    **doc["metadata"],
                },
            }
        )

    batch_size = 100
    for i in range(0, len(pinecone_vectors), batch_size):
        index.upsert(vectors=pinecone_vectors[i : i + batch_size])

    stats = index.describe_index_stats()
    print(f"pinecone_index_name={settings.pinecone_index_name}")
    print(f"upserted_chunks={len(docs)}")
    print(f"total_vectors={stats.total_vector_count}")


if __name__ == "__main__":
    main()
