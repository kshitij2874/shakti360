"""
rag.py — PageIndex-style RAG retrieval pipeline.
Embeddings: gemini-embedding-001 via Vertex AI (fallback to mock).
Vector store: Firestore with vector search (fallback to local numpy cosine).
Retrieval: top-3 chunks filtered by pillar + age_band BEFORE similarity.
Hierarchical chunking: section headers → paragraphs → metadata.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from observability import observe

logger = logging.getLogger("shakti.rag")

# ── Config ──
DOCS_DIR = Path(__file__).parent / "docs"
MAX_RAG_TOKENS = 1500
TOP_K = 3

# ── Local vector store fallback ──
_local_vectors: list[dict[str, Any]] = []
_vectors_loaded = False

# ── Firestore ──
_firestore_client = None
_firestore_available = False

# ── Vertex AI Availability ──
_vertex_available = True
_embedding_model = None



def _get_firestore():
    global _firestore_client, _firestore_available
    if _firestore_client is not None:
        return _firestore_client
    try:
        from google.cloud import firestore
        _firestore_client = firestore.Client()
        _firestore_available = True
        return _firestore_client
    except Exception as e:
        logger.warning(f"Firestore unavailable for RAG: {e}")
        _firestore_available = False
        return None


# ═══════════════════════════════════════════════════════
# EMBEDDINGS
# ═══════════════════════════════════════════════════════

async def get_embedding(text: str) -> list[float]:
    """Generate embedding via Vertex AI or deterministic 768-dim mock."""
    global _vertex_available, _embedding_model
    if not text or not text.strip():
        text = "empty"
    
    if _vertex_available:
        try:
            if _embedding_model is None:
                from vertexai.language_models import TextEmbeddingModel
                _embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _embedding_model.get_embeddings, [text])
            return result[0].values
        except Exception as e:
            logger.warning(f"Vertex embedding failed, using mock: {e}")
            _vertex_available = False
            
    embedding = []
    seed = text.lower().strip()
    while len(embedding) < 768:
        seed_hash = hashlib.sha256(seed.encode()).hexdigest()
        for i in range(0, len(seed_hash) - 1, 2):
            if len(embedding) >= 768:
                break
            hex_pair = seed_hash[i:i+2]
            if len(hex_pair) == 2:
                embedding.append(int(hex_pair, 16) / 255.0)
        seed = seed_hash
    return embedding[:768]

async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts in a single batch call via Vertex AI or fallback to mock."""
    global _vertex_available, _embedding_model
    if not texts:
        return []
        
    if _vertex_available:
        try:
            if _embedding_model is None:
                from vertexai.language_models import TextEmbeddingModel
                _embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
            
            results = []
            batch_size = 250
            for i in range(0, len(texts), batch_size):
                batch_texts = [t or "empty" for t in texts[i:i+batch_size]]
                loop = asyncio.get_event_loop()
                batch_result = await loop.run_in_executor(
                    None, 
                    _embedding_model.get_embeddings, 
                    batch_texts
                )
                results.extend([r.values for r in batch_result])
            return results
        except Exception as e:
            logger.warning(f"Vertex batch embedding failed, falling back to mock: {e}")
            _vertex_available = False
            
    # Mock fallback
    results = []
    for text in texts:
        embedding = []
        seed = (text or "empty").lower().strip()
        while len(embedding) < 768:
            seed_hash = hashlib.sha256(seed.encode()).hexdigest()
            for i in range(0, len(seed_hash) - 1, 2):
                if len(embedding) >= 768:
                    break
                hex_pair = seed_hash[i:i+2]
                if len(hex_pair) == 2:
                    embedding.append(int(hex_pair, 16) / 255.0)
            seed = seed_hash
        results.append(embedding[:768])
    return results

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ═══════════════════════════════════════════════════════
# HIERARCHICAL CHUNKING (PageIndex-style)
# ═══════════════════════════════════════════════════════

def chunk_document(
    text: str,
    pillar: str,
    age_band: str,
    source_file: str,
) -> list[dict[str, Any]]:
    """
    PageIndex-style hierarchical chunking:
    1. Split by section headers (lines ending with ':' or all-caps lines)
    2. Within sections, split by paragraphs
    3. Store hierarchical metadata (document > section > paragraph)
    """
    chunks = []
    lines = text.strip().split("\n")

    # Extract document title (first non-empty line)
    doc_title = ""
    for line in lines:
        if line.strip():
            doc_title = line.strip()
            break

    # Extract source line
    source_ref = ""
    for line in reversed(lines):
        if line.strip().startswith("[Source:"):
            source_ref = line.strip()
            break

    # Split into sections by headers
    sections: list[dict[str, Any]] = []
    current_section = {"title": doc_title, "content": []}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect section headers
        is_header = (
            (stripped.endswith(":") and len(stripped) < 80 and not stripped.startswith("-"))
            or (stripped.isupper() and len(stripped) > 3 and len(stripped) < 80)
            or re.match(r"^\d+\.\s+", stripped)
        )

        if is_header and current_section["content"]:
            sections.append(current_section)
            current_section = {"title": stripped.rstrip(":"), "content": []}
        else:
            current_section["content"].append(stripped)

    if current_section["content"]:
        sections.append(current_section)

    # Create chunks from sections
    for section_idx, section in enumerate(sections):
        section_text = "\n".join(section["content"])

        # Split long sections into paragraph-level chunks
        paragraphs = re.split(r'\n\n+', section_text)
        if len(paragraphs) == 1 and len(section_text) > 500:
            # Further split by sentence groups (roughly 200-300 chars)
            sentences = re.split(r'(?<=[.!?])\s+', section_text)
            para_group: list[str] = []
            current_len = 0
            paragraphs = []
            for sent in sentences:
                if current_len + len(sent) > 300 and para_group:
                    paragraphs.append(" ".join(para_group))
                    para_group = []
                    current_len = 0
                para_group.append(sent)
                current_len += len(sent)
            if para_group:
                paragraphs.append(" ".join(para_group))

        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) < 20:
                continue

            chunk_id = hashlib.md5(
                f"{source_file}:{section_idx}:{para_idx}".encode()
            ).hexdigest()[:12]

            chunks.append({
                "id": chunk_id,
                "content": paragraph,
                "pillar": pillar,
                "age_band": age_band,
                "source": source_file,
                "source_ref": source_ref,
                "doc_title": doc_title,
                "section_title": section["title"],
                "section_index": section_idx,
                "paragraph_index": para_idx,
                "hierarchy": f"{doc_title} > {section['title']} > P{para_idx}",
            })

    return chunks


# ═══════════════════════════════════════════════════════
# DOCUMENT INGESTION
# ═══════════════════════════════════════════════════════

@observe("rag_ingest")
async def ingest_documents() -> int:
    """Load all documents from docs/ directory, chunk, embed, and store."""
    global _local_vectors, _vectors_loaded

    if _vectors_loaded:
        return len(_local_vectors)

    total_chunks = 0

    for pillar_dir in DOCS_DIR.iterdir():
        if not pillar_dir.is_dir():
            continue
        pillar = pillar_dir.name.upper()  # health → HEALTH

        for age_dir in pillar_dir.iterdir():
            if not age_dir.is_dir():
                continue
            age_band = age_dir.name  # 11-24, 25-40, 41+

            for doc_file in age_dir.glob("*.txt"):
                try:
                    text = doc_file.read_text(encoding="utf-8")
                    chunks = chunk_document(
                        text=text,
                        pillar=pillar,
                        age_band=age_band,
                        source_file=doc_file.name,
                    )

                    # Generate embeddings in a single batch call for the document
                    texts = [chunk["content"] for chunk in chunks]
                    embeddings = await get_embeddings_batch(texts)
                    for chunk, emb in zip(chunks, embeddings):
                        chunk["embedding"] = emb
                        _local_vectors.append(chunk)

                    total_chunks += len(chunks)
                    logger.info(f"Ingested {doc_file.name}: {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Failed to ingest {doc_file}: {e}")

    # Also try storing in Firestore
    db = _get_firestore()
    if db:
        try:
            for chunk in _local_vectors:
                db.collection("rag_chunks").document(chunk["id"]).set(chunk)
            logger.info(f"Stored {len(_local_vectors)} chunks in Firestore")
        except Exception as e:
            logger.warning(f"Firestore chunk storage failed: {e}")

    _vectors_loaded = True
    logger.info(f"RAG ingestion complete: {total_chunks} total chunks from {DOCS_DIR}")
    return total_chunks


# ═══════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════

@observe("rag_retrieve")
async def retrieve(
    query: str,
    pillar: str,
    age_band: str,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k relevant chunks.
    MUST filter by pillar + age_band BEFORE similarity search.
    Truncates total context to MAX_RAG_TOKENS.
    """
    # Ensure documents are ingested
    if not _vectors_loaded:
        await ingest_documents()

    query_embedding = await get_embedding(query)

    # Step 1: Filter by pillar + age_band (metadata pre-filter)
    filtered = [
        chunk for chunk in _local_vectors
        if chunk["pillar"].upper() == pillar.upper()
        and chunk["age_band"] == age_band
    ]

    if not filtered:
        logger.warning(f"No chunks found for pillar={pillar}, age_band={age_band}")
        return []

    # Step 2: Compute cosine similarity
    scored = []
    for chunk in filtered:
        sim = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append({**chunk, "similarity": round(sim, 4)})

    # Step 3: Sort by similarity descending, take top-k
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top_chunks = scored[:top_k]

    # Step 4: Truncate to MAX_RAG_TOKENS
    total_chars = 0
    truncated = []
    for chunk in top_chunks:
        content = chunk["content"]
        remaining = MAX_RAG_TOKENS * 4 - total_chars  # ~4 chars per token
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[:remaining] + "..."
            chunk = {**chunk, "content": content}
        truncated.append(chunk)
        total_chars += len(content)

    logger.info(
        f"RAG retrieved {len(truncated)} chunks for "
        f"pillar={pillar}, age_band={age_band}, "
        f"total_chars={total_chars}"
    )
    return truncated


async def web_search_fallback(query: str, pillar: str) -> list[dict[str, Any]]:
    """When the local RAG corpus has nothing, search the web (Tavily) and return
    results shaped like RAG chunks so the rest of the pipeline is unchanged.
    Clearly marked as web sources (not official docs) for transparency.
    Returns [] if web search is unavailable."""
    try:
        from advanced_tools import (
            tavily_search_health, tavily_search_finance, tavily_search_career,
        )
        fn = {
            "HEALTH": tavily_search_health,
            "FINANCE": tavily_search_finance,
            "CAREER": tavily_search_career,
        }.get(pillar.upper())
        if not fn:
            return []

        result = await fn(query=query, max_results=4)
        if not result.get("success"):
            return []

        chunks: list[dict[str, Any]] = []

        # Tavily's synthesized answer makes a strong first chunk
        answer = (result.get("answer") or "").strip()
        if answer:
            chunks.append({
                "id": "web_answer",
                "content": answer,
                "pillar": pillar.upper(),
                "age_band": "",
                "source": "web_search",
                "source_ref": "[Web search — verify before acting]",
                "doc_title": "Web search",
                "section_title": "Summary",
                "is_web": True,
            })

        for i, r in enumerate(result.get("results", [])[:4]):
            snippet = (r.get("snippet") or "").strip()
            if not snippet:
                continue
            url = r.get("url", "")
            domain = url.split("/")[2] if "://" in url else url
            chunks.append({
                "id": f"web_{i}",
                "content": snippet,
                "pillar": pillar.upper(),
                "age_band": "",
                "source": domain or "web",
                "source_ref": f"[Web: {url}]",
                "doc_title": r.get("title", "Web result"),
                "section_title": r.get("title", "")[:80],
                "is_web": True,
            })

        logger.info(f"Web fallback returned {len(chunks)} chunks for pillar={pillar}")
        return chunks
    except Exception as e:
        logger.warning(f"Web search fallback failed: {e}")
        return []


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    if not chunks:
        return "No relevant context found."

    parts = ["[Retrieved Context]"]
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"\n--- Source {i}: {chunk.get('source', 'unknown')} "
            f"(Section: {chunk.get('section_title', 'N/A')}) ---\n"
            f"{chunk['content']}"
        )
        if chunk.get("source_ref"):
            parts.append(f"\n{chunk['source_ref']}")

    return "\n".join(parts)
